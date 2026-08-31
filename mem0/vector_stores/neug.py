"""NeuG vector store backend for mem0.

NeuG is a graph database engine with native HNSW vector indexes, BM25
full-text indexes, and property-graph edges. Each mem0 collection maps to one
NeuG node table plus one relation table:

    Node table:  id STRING PRIMARY KEY, vector FLOAT[dim], text VARCHAR(65535),
                 payload VARCHAR(65535), user_id STRING, agent_id STRING,
                 run_id STRING, actor_id STRING
    Rel table:   <node_table>_mem0_links (FROM <node_table> TO <node_table>,
                 relation STRING, weight DOUBLE)

- ``text`` holds ``payload["text_lemmatized"] or payload["data"]`` and carries
  the FTS index, so ``keyword_search`` uses native BM25 (no sparse encoder).
- ``payload`` holds the full JSON payload; ``user_id``/``agent_id``/``run_id``/
  ``actor_id`` are promoted to columns for server-side filtering. Filters on
  any other key are applied client-side after over-fetching.
- Graph traversal uses the relation table (``add_edge``/``remove_edge``/
  ``traverse``); ``delete``/``reset`` clean edges via DETACH DELETE / DROP.

Dialect notes (verified against NeuG main @ 5300cb3, previously 01d88ef):
- ``vector_search`` / ``fts`` extensions must be LOADed per connection.
- ``conn.execute`` takes ``parameters`` as a keyword dict.
- ``IN $list`` bound parameters and bound ``LIMIT`` are not usable, so IN is
  expanded to OR-ed equality and LIMIT is inlined as an integer literal.
- ``DROP TABLE`` now tolerates attached HNSW indexes (fixed upstream), but
  indexes (and the relation table) are still dropped before the node table.
- Updates go through ``SET`` (which keeps row data and the FTS index in sync).
- The engine allows duplicate edges between the same node pair, so
  ``add_edge`` checks for an existing edge first.
- Performance: ``search`` uses the HNSW IndexScan plan (ORDER BY distance ASC
  LIMIT) with a bound ``$q`` query vector and projects the metric distance
  directly. Since 5300cb3 (alibaba/neug#968, closing alibaba/neug#931) the
  ANN rewrite path reports true distances — including rows updated via SET —
  so no vector transfer or client-side re-ranking is needed on this path.
  Pass ``exact=True`` to force an O(n) full scan.
- A NeuG database directory can only be opened once per process (error 1004)
  and only supports ONE read-write connection per Database (error 4001), so
  stores on the same ``db_path`` share one Database handle and one connection
  with reference counting and a lock; mem0's telemetry store and entity store
  open the same path.
"""

import atexit
import hashlib
import json
import logging
import math
import os
import re
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)

# Columns promoted out of the payload for server-side filtering.
PROMOTED_COLUMNS = ("user_id", "agent_id", "run_id", "actor_id")

# mem0 distance value -> (HNSW metric, distance function).
# Only cosine and l2 are verified on NeuG 0.2.0; inner-product support is not
# usable yet (no valid metric name + query function pair), so it is rejected.
_DISTANCE_METRICS = {
    "cosine": ("cosine", "vector_distance_cosine"),
    "l2": ("l2", "vector_distance_l2"),
    "euclidean": ("l2", "vector_distance_l2"),
}

# Over-fetch factor applied when results still need client-side filtering.
_OVERFETCH_FACTOR = 5
_OVERFETCH_MIN = 50

# Rows per multi-row CREATE statement in insert(). Bound UNWIND lists are
# unsupported (parameter serialization rejects them), so batching happens as
# one CREATE with N node patterns; this cuts per-statement overhead from
# ~70ms/row to ~1ms/row while keeping HNSW/FTS index maintenance intact.
_INSERT_BATCH_SIZE = 128

# Process-level Database/connection registry: NeuG refuses a second open of
# the same directory within one process (error 1004) and a second read-write
# connection on the same Database (error 4001), but mem0 creates multiple
# stores on the same db_path (telemetry "mem0migrations", entity store).
# Each entry is [database, connection, refs, lock].
_DATABASES: Dict[Tuple[str, int], List[Any]] = {}
_DATABASES_LOCK = threading.Lock()


def _acquire_connection(db_path: str, database_cls: Any) -> Tuple[Any, Any, threading.Lock]:
    """Return the shared (database, connection, lock) for db_path, creating on first use."""
    key = (os.path.abspath(db_path), id(database_cls))
    with _DATABASES_LOCK:
        entry = _DATABASES.get(key)
        if entry is None:
            db = database_cls(db_path)
            entry = [db, db.connect(), 0, threading.Lock()]
            _DATABASES[key] = entry
        entry[2] += 1
        return entry[0], entry[1], entry[3]


def _release_connection(db_path: str, database_cls: Any):
    key = (os.path.abspath(db_path), id(database_cls))
    with _DATABASES_LOCK:
        entry = _DATABASES.get(key)
        if entry is None:
            return
        entry[2] -= 1
        if entry[2] > 0:
            return
        # Last reference: unregister first so new acquirers open a fresh
        # Database, then close outside the global lock, under the connection
        # lock, so any in-flight _execute finishes before the handles close.
        _DATABASES.pop(key, None)
        db, conn, conn_lock = entry[0], entry[1], entry[3]
    with conn_lock:
        try:
            conn.close()
        except Exception:
            pass
        try:
            db.close()
        except Exception:
            pass


class NeuGOutput:
    """Search result object matching the id/score/payload shape mem0 expects."""

    def __init__(self, id: Any, score: Optional[float] = None, payload: Optional[Dict] = None):
        self.id = id
        self.score = score
        self.payload = payload

    def __repr__(self):
        return f"NeuGOutput(id={self.id!r}, score={self.score!r})"


def _sanitize_table_name(name: str) -> str:
    """Map a collection name to a safe, collision-free NeuG node table name.

    Names that needed sanitizing get a short hash suffix so e.g.
    ``my-collection`` and ``my_collection`` cannot silently share one table.
    """
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"m_{sanitized}"
    if sanitized != name:
        suffix = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
        sanitized = f"{sanitized}_{suffix}"
    return sanitized


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Exact cosine similarity in [0, 1] (negative values clamped)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))


def _l2_similarity(a: List[float], b: List[float]) -> float:
    """L2 distance converted to the base-class similarity contract."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return 1.0 / (1.0 + math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def _is_duplicate_pk_error(error: Exception) -> bool:
    """Detect NeuG primary-key conflict errors (code 1009)."""
    message = str(error).lower()
    return "primary key" in message or "error code: 1009" in message or "code: 1009" in message


def _score_from_distance(metric: str, dist: Any) -> float:
    """Convert an engine-reported distance into mem0's [0, 1] similarity.

    ``vector_distance_cosine`` returns 1 - cosine similarity and
    ``vector_distance_l2`` returns the SQUARED L2 distance (verified on
    NeuG main @ 5300cb3).
    """
    try:
        d = float(dist)
    except (TypeError, ValueError):
        return 0.0
    if metric.endswith("cosine"):
        return max(0.0, min(1.0, 1.0 - d))
    return 1.0 / (1.0 + math.sqrt(max(0.0, d)))


def _sanitize_fts_query(query: str) -> str:
    """Quote each whitespace token so NeuG FTS (SQLite FTS5) parses it literally.

    An unquoted token containing a hyphen (e.g. ``dutch-made``) is parsed as a
    column expression by FTS5 and fails with ``no such column``, which used to
    abort the whole keyword channel (verified on NeuG main @ 5300cb3).
    Per-token quoting keeps the AND semantics of multi-token queries.
    """
    tokens = query.split()
    if not tokens:
        return ""
    return " ".join('"{}"'.format(t.replace('"', '""')) for t in tokens)


class NeuG(VectorStoreBase):
    def __init__(
        self,
        collection_name: str = "mem0",
        embedding_model_dims: int = 1536,
        db_path: str = "/tmp/neug_mem0",
        distance: str = "cosine",
    ):
        """
        Initialize the NeuG vector store.

        Args:
            collection_name (str): Name of the collection (mapped to a node table).
            embedding_model_dims (int): Dimensions of the embedding model.
            db_path (str): Path to the NeuG database directory.
            distance (str): Distance metric, 'cosine' or 'l2'.
        """
        try:
            from neug import Database
        except ImportError as e:
            raise ImportError(
                "NeuG Python bindings are not installed. Build the neug python_bind "
                "package for your interpreter and make it importable as `neug`."
            ) from e

        if distance not in _DISTANCE_METRICS:
            raise ValueError(
                f"Unsupported distance metric for NeuG: {distance!r}. Supported: {sorted(_DISTANCE_METRICS)}"
            )

        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.db_path = db_path
        self.distance = distance
        self._metric, self._distance_func = _DISTANCE_METRICS[distance]
        self.table_name = _sanitize_table_name(collection_name)
        self._edge_table = f"{self.table_name}_mem0_links"
        self._hnsw_index = f"{self.table_name}_mem0_hnsw"
        self._fts_index = f"{self.table_name}_mem0_fts"
        self._database_cls = Database

        self._db, self._conn, self._conn_lock = _acquire_connection(db_path, Database)
        self._closed = False
        try:
            for ext in ("vector_search", "fts"):
                self._execute(f"LOAD {ext}")
            self.create_col(collection_name, embedding_model_dims, distance)
        except Exception:
            # Give back the reference so a failed init cannot pin the
            # shared Database open for the rest of the process lifetime.
            self.close()
            raise
        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def close(self):
        """Release this store's share of the connection; the last one closes it."""
        # Flip the flag under the connection lock so concurrent close() calls
        # (e.g. explicit close racing atexit) can only release once.
        with self._conn_lock:
            if self._closed:
                return
            self._closed = True
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
        _release_connection(self.db_path, self._database_cls)

    def _execute(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Run a query and return the rows as a list of dicts (AS aliases as keys)."""
        with self._conn_lock:
            result = self._conn.execute(query, parameters=params) if params else self._conn.execute(query)
            raw = result.get_bolt_response()
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            return []
        return raw.get("table") or []

    def _get_schema(self) -> Dict:
        try:
            with self._conn_lock:
                return json.loads(self._conn.get_schema())
        except Exception as e:
            logger.debug(f"get_schema failed: {e}")
            return {}

    def _table_exists(self, name: str) -> bool:
        schema = self._get_schema()
        if not schema:
            return False
        return any(vt.get("type_name") == name for vt in schema.get("schema", {}).get("vertex_types", []))

    def _build_server_filters(self, filters: Optional[Dict]) -> Tuple[str, Dict, Dict, bool]:
        """Split filters into a server-side WHERE clause and leftover client filters.

        Only equality-style conditions on promoted columns are pushed to the
        engine; everything else is evaluated client-side after over-fetching.

        Returns:
            (where_clause, params, client_filters, match_nothing) where
            ``match_nothing`` is True when the filters cannot match any row
            (e.g. an empty ``in`` list) and the caller should return [].
        """
        if not filters:
            return "", {}, {}, False

        clauses = []
        params = {}
        client_filters = {}

        for key, value in filters.items():
            if key not in PROMOTED_COLUMNS:
                client_filters[key] = value
                continue
            column = f"m.{key}"
            if isinstance(value, dict):
                ops = set(value.keys())
                if ops == {"eq"}:
                    params[key] = value["eq"]
                    clauses.append(f"{column} = ${key}")
                elif ops == {"ne"}:
                    params[key] = value["ne"]
                    clauses.append(f"{column} <> ${key}")
                elif ops == {"in"}:
                    values = value["in"]
                    if not isinstance(values, list):
                        raise ValueError(f"Filter operator 'in' for key {key!r} requires a list value")
                    if not values:
                        return "", {}, {}, True  # IN [] matches nothing
                    clauses.append(self._expand_in(column, values, key, params))
                elif ops == {"nin"}:
                    values = value["nin"]
                    if not isinstance(values, list):
                        raise ValueError(f"Filter operator 'nin' for key {key!r} requires a list value")
                    if values:
                        clauses.append(self._expand_nin(column, values, key, params))
                else:
                    # Range/contains ops on promoted columns fall back to client-side.
                    client_filters[key] = value
            elif isinstance(value, list):
                if not value:
                    return "", {}, {}, True  # equality against [] matches nothing
                clauses.append(self._expand_in(column, value, key, params))
            elif value == "*":
                continue  # wildcard: match any, nothing to filter
            else:
                params[key] = value
                clauses.append(f"{column} = ${key}")

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params, client_filters, False

    @staticmethod
    def _expand_in(column: str, values: List, prefix: str, params: Dict) -> str:
        """Expand an IN condition to OR-ed equality (bound IN lists are unsupported)."""
        parts = []
        for i, v in enumerate(values):
            pname = f"{prefix}_{i}"
            params[pname] = v
            parts.append(f"{column} = ${pname}")
        return "(" + " OR ".join(parts) + ")" if parts else ""

    @staticmethod
    def _expand_nin(column: str, values: List, prefix: str, params: Dict) -> str:
        parts = []
        for i, v in enumerate(values):
            pname = f"{prefix}_{i}"
            params[pname] = v
            parts.append(f"{column} <> ${pname}")
        return "(" + " AND ".join(parts) + ")" if parts else ""

    @staticmethod
    def _matches_client_filters(payload: Dict, filters: Dict) -> bool:
        """Evaluate leftover filter conditions against a payload dict."""

        def match_condition(key: str, value: Any) -> bool:
            payload_value = payload.get(key)
            if isinstance(value, dict):
                for op, expected in value.items():
                    try:
                        if op == "eq":
                            if payload_value != expected:
                                return False
                        elif op == "ne":
                            if payload_value == expected:
                                return False
                        elif op == "gt":
                            if not (payload_value is not None and payload_value > expected):
                                return False
                        elif op == "gte":
                            if not (payload_value is not None and payload_value >= expected):
                                return False
                        elif op == "lt":
                            if not (payload_value is not None and payload_value < expected):
                                return False
                        elif op == "lte":
                            if not (payload_value is not None and payload_value <= expected):
                                return False
                        elif op == "in":
                            if payload_value not in expected:
                                return False
                        elif op == "nin":
                            if payload_value in expected:
                                return False
                        elif op == "contains":
                            if not (isinstance(payload_value, str) and expected in payload_value):
                                return False
                        elif op == "icontains":
                            if not (isinstance(payload_value, str) and expected.lower() in payload_value.lower()):
                                return False
                        else:
                            raise ValueError(f"Unsupported filter operator for NeuG store: {op}")
                    except TypeError:
                        # Incomparable types (e.g. str vs int) never match.
                        return False
                return True
            if value == "*":
                return True
            return payload_value == value

        for key, value in filters.items():
            if key in ("AND", "$and"):
                if not all(match_condition(k, v) for item in value for k, v in item.items()):
                    return False
            elif key in ("OR", "$or"):
                if not any(all(match_condition(k, v) for k, v in item.items()) for item in value):
                    return False
            elif key in ("NOT", "$not"):
                if any(all(match_condition(k, v) for k, v in item.items()) for item in value):
                    return False
            elif not match_condition(key, value):
                return False
        return True

    @staticmethod
    def _parse_payload(raw: Any) -> Dict:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}

    def _similarity_fn(self):
        return _cosine_similarity if self._distance_func.endswith("cosine") else _l2_similarity

    # ------------------------------------------------------------------
    # VectorStoreBase interface
    # ------------------------------------------------------------------

    def create_col(self, name: str, vector_size: int, distance: str = "cosine"):
        """Create the node table, relation table, and HNSW/FTS indexes."""
        table = _sanitize_table_name(name)
        if self._table_exists(table):
            logger.debug(f"NeuG table {table} already exists. Skipping creation.")
            return

        metric, _ = _DISTANCE_METRICS.get(distance, _DISTANCE_METRICS["cosine"])
        self._execute(
            f"CREATE NODE TABLE {table} ("
            "id STRING PRIMARY KEY, "
            f"vector FLOAT[{int(vector_size)}], "
            "text VARCHAR(65535), "
            "payload VARCHAR(65535), "
            "user_id STRING, agent_id STRING, run_id STRING, actor_id STRING)"
        )
        self._execute(
            f"CREATE INDEX {table}_mem0_hnsw ON {table} USING HNSW (vector) WITH (metric = '{metric}')"
        )
        self._execute(f"CREATE INDEX {table}_mem0_fts ON {table} USING FTS (text)")
        self._execute(f"CREATE REL TABLE {table}_mem0_links (FROM {table} TO {table}, relation STRING, weight DOUBLE)")

    def insert(self, vectors: List, payloads: List = None, ids: List = None):
        """Insert vectors (upsert semantics: existing IDs are updated).

        Rows are written as multi-row CREATE statements for bulk throughput.
        A batch failing on a duplicate primary key rolls back atomically and
        is retried row by row so per-row upsert semantics are preserved.
        """
        if payloads is not None and len(payloads) != len(vectors):
            raise ValueError(f"payloads length ({len(payloads)}) must match vectors length ({len(vectors)})")
        if ids is not None and len(ids) != len(vectors):
            raise ValueError(f"ids length ({len(ids)}) must match vectors length ({len(vectors)})")
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        for start in range(0, len(vectors), _INSERT_BATCH_SIZE):
            end = min(start + _INSERT_BATCH_SIZE, len(vectors))
            batch_ids = [str(ids[i]) if ids is not None else str(uuid.uuid4()) for i in range(start, end)]
            try:
                self._bulk_create(
                    batch_ids,
                    vectors[start:end],
                    payloads[start:end] if payloads is not None else [{}] * (end - start),
                )
            except Exception as e:
                if not _is_duplicate_pk_error(e):
                    raise
                # Duplicate primary key somewhere in the batch (it rolled back
                # atomically): replay it one row at a time via upsert.
                for i, vector_id in enumerate(batch_ids):
                    self._upsert(vector_id, vectors[start + i], payloads[start + i] if payloads else {})

    def _bulk_create(self, vector_ids: List[str], vectors: List, payloads: List):
        """Write a batch of rows as one multi-row CREATE with bound parameters."""
        patterns = []
        params: Dict[str, Any] = {}
        for j, (vector_id, vector, payload) in enumerate(zip(vector_ids, vectors, payloads)):
            text = payload.get("text_lemmatized") or payload.get("data", "")
            patterns.append(
                f"(:{self.table_name} {{id: $id{j}, vector: $vec{j}, text: $text{j}, "
                f"payload: $payload{j}, user_id: $user_id{j}, agent_id: $agent_id{j}, "
                f"run_id: $run_id{j}, actor_id: $actor_id{j}}})"
            )
            params[f"id{j}"] = vector_id
            params[f"vec{j}"] = [float(x) for x in vector]
            params[f"text{j}"] = text
            params[f"payload{j}"] = json.dumps(payload)
            for col in PROMOTED_COLUMNS:
                params[f"{col}{j}"] = payload.get(col)
        self._execute("CREATE " + ", ".join(patterns), params)

    def _upsert(self, vector_id: str, vector: List, payload: Dict):
        text = payload.get("text_lemmatized") or payload.get("data", "")
        params = {
            "id": vector_id,
            "vec": [float(x) for x in vector],
            "text": text,
            "payload": json.dumps(payload),
        }
        for col in PROMOTED_COLUMNS:
            params[col] = payload.get(col)
        try:
            self._execute(
                f"CREATE (:{self.table_name} {{id: $id, vector: $vec, text: $text, payload: $payload, "
                "user_id: $user_id, agent_id: $agent_id, run_id: $run_id, actor_id: $actor_id})",
                params,
            )
        except Exception as e:
            if not _is_duplicate_pk_error(e):
                raise
            # Duplicate primary key: fall back to SET, then verify the row was
            # actually written (a MATCH...SET on a missing row is a silent no-op).
            self.update(vector_id, vector=vector, payload=payload)
            if self.get(vector_id) is None:
                raise

    def search(self, query: str, vectors: List, top_k: int = 5, filters: Dict = None, ann: bool = True, exact: bool = False) -> List:
        """Search for similar vectors.

        The default path is the HNSW IndexScan plan (ORDER BY distance ASC
        LIMIT) with a bound ``$q`` query vector; the engine projects the
        metric distance, which the adapter converts to mem0's [0, 1]
        similarity contract (verified on NeuG main @ 5300cb3, where the
        former alibaba/neug#931 raw-score defect is fixed). ``ann`` is
        accepted for compatibility (ANN is the default); pass ``exact=True``
        to force an O(n) full scan ranked client-side.
        """
        where, params, client_filters, match_nothing = self._build_server_filters(filters)
        if match_nothing:
            return []
        query_vec = [float(x) for x in vectors]
        similarity = self._similarity_fn()

        if exact:
            # Full scan: no ORDER BY/LIMIT (an unordered scan truncates rows
            # arbitrarily); ranking happens entirely client-side.
            query_sql = (
                f"MATCH (m:{self.table_name}){where} "
                "RETURN m.id AS id, m.payload AS payload, m.vector AS vector"
            )
            exec_params = params or None
        else:
            # ANN IndexScan path. Bound $q works on this plan. Over-fetch only
            # when rows may be dropped client-side: with purely server-side
            # filters the engine applies them during the traversal, and a
            # LIMIT-sized pool re-ranks identically to a larger one (verified
            # at benchmark scale), so skip transferring vectors that would be
            # discarded anyway.
            fetch_k = max(top_k * _OVERFETCH_FACTOR, _OVERFETCH_MIN) if client_filters else top_k
            params["q"] = query_vec
            query_sql = (
                f"MATCH (m:{self.table_name}){where} "
                "RETURN m.id AS id, m.payload AS payload, "
                f"{self._distance_func}(m.vector, $q) AS d "
                f"ORDER BY d ASC LIMIT {int(fetch_k)}"
            )
            exec_params = params

        rows = self._execute(query_sql, exec_params)
        if exact:
            results = [
                NeuGOutput(
                    id=row["id"],
                    score=similarity([float(x) for x in row.get("vector") or []], query_vec),
                    payload=self._parse_payload(row.get("payload")),
                )
                for row in rows
            ]
        else:
            results = [
                NeuGOutput(
                    id=row["id"],
                    score=_score_from_distance(self._metric, row.get("d")),
                    payload=self._parse_payload(row.get("payload")),
                )
                for row in rows
            ]
        if client_filters:
            results = [r for r in results if self._matches_client_filters(r.payload, client_filters)]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def keyword_search(self, query: str, top_k: int = 5, filters: Dict = None):
        """Full-text BM25 search over the FTS index on the text column.

        NeuG's bm25() returns negative scores where lower is better; scores are
        negated so higher raw values indicate better matches, which mem0's
        sigmoid normalization (utils/scoring.py) expects.
        """
        if not query:
            return None
        fts_query = _sanitize_fts_query(query)
        if not fts_query:
            return None
        where, params, client_filters, match_nothing = self._build_server_filters(filters)
        if match_nothing:
            return []
        fetch_k = max(top_k * _OVERFETCH_FACTOR, _OVERFETCH_MIN) if client_filters else top_k
        params["q"] = fts_query
        try:
            rows = self._execute(
                f"MATCH (m:{self.table_name}){where} "
                "RETURN m.id AS id, m.payload AS payload, bm25(m.text, $q) AS score "
                f"ORDER BY score ASC LIMIT {int(fetch_k)}",
                params,
            )
        except Exception as e:
            logger.debug(f"NeuG keyword search failed: {e}")
            return None
        results = [
            NeuGOutput(id=row["id"], score=-float(row["score"]), payload=self._parse_payload(row.get("payload")))
            for row in rows
        ]
        if client_filters:
            results = [r for r in results if self._matches_client_filters(r.payload, client_filters)]
        return results[:top_k]

    def delete(self, vector_id: str):
        """Delete a vector by ID (DETACH DELETE also removes its edges)."""
        self._execute(f"MATCH (m:{self.table_name} {{id: $id}}) DETACH DELETE m", {"id": str(vector_id)})

    def update(self, vector_id: str, vector: List = None, payload: Dict = None):
        """Update a vector and/or its payload in place via SET."""
        sets = []
        params = {"id": str(vector_id)}
        if vector is not None:
            params["vec"] = [float(x) for x in vector]
            sets.append("m.vector = $vec")
        if payload is not None:
            text = payload.get("text_lemmatized") or payload.get("data", "")
            params["text"] = text
            params["payload"] = json.dumps(payload)
            sets.append("m.text = $text")
            sets.append("m.payload = $payload")
            for col in PROMOTED_COLUMNS:
                # NeuG rejects SET col = NULL (ERR_NOT_SUPPORTED 1011), so only
                # assign promoted columns the payload actually carries.
                if payload.get(col) is not None:
                    params[col] = payload.get(col)
                    sets.append(f"m.{col} = ${col}")
        if not sets:
            return
        self._execute(
            f"MATCH (m:{self.table_name} {{id: $id}}) SET {', '.join(sets)}",
            params,
        )

    def get(self, vector_id: str):
        """Retrieve a vector by ID."""
        rows = self._execute(
            f"MATCH (m:{self.table_name} {{id: $id}}) RETURN m.id AS id, m.payload AS payload LIMIT 1",
            {"id": str(vector_id)},
        )
        if not rows:
            return None
        return NeuGOutput(id=rows[0]["id"], score=1.0, payload=self._parse_payload(rows[0].get("payload")))

    def list_cols(self) -> List[str]:
        """List all node tables known to the engine."""
        schema = self._get_schema()
        if not schema:
            return []
        return [vt.get("type_name") for vt in schema.get("schema", {}).get("vertex_types", [])]

    def delete_col(self):
        """Delete the collection: indexes and the relation table first, then the node table."""
        for stmt in (
            f"DROP INDEX {self._fts_index}",
            f"DROP INDEX {self._hnsw_index}",
            f"DROP TABLE {self._edge_table}",
        ):
            try:
                self._execute(stmt)
            except Exception as e:
                logger.debug(f"{stmt} skipped: {e}")
        self._execute(f"DROP TABLE {self.table_name}")

    def col_info(self) -> Dict:
        """Get information about the collection."""
        return {
            "name": self.collection_name,
            "table_name": self.table_name,
            "edge_table": self._edge_table,
            "vector_size": self.embedding_model_dims,
            "distance": self.distance,
        }

    def list(self, filters: Dict = None, top_k: int = None) -> List:
        """List all memories, optionally filtered.

        Returns the mem0 wrapped shape ``[[NeuGOutput, ...]]`` expected by
        ``Memory.delete_all`` and ``Memory.get_all``.
        """
        where, params, client_filters, match_nothing = self._build_server_filters(filters)
        if match_nothing:
            return [[]]
        rows = self._execute(
            f"MATCH (m:{self.table_name}){where} RETURN m.id AS id, m.payload AS payload",
            params or None,
        )
        results = [NeuGOutput(id=row["id"], score=None, payload=self._parse_payload(row.get("payload"))) for row in rows]
        if client_filters:
            results = [r for r in results if self._matches_client_filters(r.payload, client_filters)]
        if top_k is not None:
            results = results[: int(top_k)]
        return [results]

    def reset(self):
        """Reset by deleting the collection and recreating it."""
        logger.warning(f"Resetting NeuG collection {self.collection_name}...")
        self.delete_col()
        self.create_col(self.collection_name, self.embedding_model_dims, self.distance)

    # ------------------------------------------------------------------
    # Graph API (native NeuG relation table)
    # ------------------------------------------------------------------

    def add_edge(self, source_id: str, target_id: str, relation: str = "related_to", weight: float = 1.0) -> bool:
        """Create an edge between two stored memories (no-op if it already exists).

        The existence check is best-effort: concurrent calls for the same
        (source, target, relation) may still create duplicate edges, which
        NeuG allows; ``traverse`` dedups nodes and ``remove_edge`` deletes
        all matching edges, so duplicates do not affect results.
        """
        params = {"src": str(source_id), "dst": str(target_id), "rel": relation}
        existing = self._execute(
            f"MATCH (x:{self.table_name} {{id: $src}})-[r:{self._edge_table} {{relation: $rel}}]->"
            f"(y:{self.table_name} {{id: $dst}}) RETURN count(r) AS n",
            params,
        )
        if existing and int(existing[0].get("n") or 0) > 0:
            return False
        self._execute(
            f"MATCH (x:{self.table_name} {{id: $src}}), (y:{self.table_name} {{id: $dst}}) "
            f"CREATE (x)-[:{self._edge_table} {{relation: $rel, weight: $w}}]->(y)",
            {**params, "w": float(weight)},
        )
        return True

    def remove_edge(self, source_id: str, target_id: str, relation: str = None):
        """Delete edges between two memories, optionally restricted to one relation."""
        rel_filter = " {relation: $rel}" if relation is not None else ""
        params = {"src": str(source_id), "dst": str(target_id)}
        if relation is not None:
            params["rel"] = relation
        self._execute(
            f"MATCH (x:{self.table_name} {{id: $src}})-[r:{self._edge_table}{rel_filter}]->"
            f"(y:{self.table_name} {{id: $dst}}) DELETE r",
            params,
        )

    def traverse(self, start_id: str, depth: int = 1, direction: str = "out", relation: str = None, limit: int = 100) -> List[NeuGOutput]:
        """Multi-hop traversal from a memory node, returning reachable memories.

        Depth > 1 is expanded hop by hop: NeuG variable-length patterns return
        PATH values that cannot be filtered on edge properties, so filtered
        traversals iterate single-hop queries (each one index-friendly) with a
        visited set instead.

        Args:
            start_id (str): ID of the starting memory.
            depth (int): Maximum number of hops (>= 1).
            direction (str): 'out', 'in', or 'both'.
            relation (str): Optional relation type to restrict edges to.
            limit (int): Maximum number of distinct nodes to return. It also
                caps the neighbors fetched per hop, so very high-degree nodes
                may yield fewer than ``limit`` results overall.
        """
        depth = max(1, int(depth))
        if direction == "out":
            edge_pattern = f"(x:{self.table_name} {{id: $src}})-[r:{self._edge_table}{{rel_filter}}]->(y:{self.table_name})"
        elif direction == "in":
            edge_pattern = f"(x:{self.table_name} {{id: $src}})<-[r:{self._edge_table}{{rel_filter}}]-(y:{self.table_name})"
        elif direction == "both":
            edge_pattern = f"(x:{self.table_name} {{id: $src}})-[r:{self._edge_table}{{rel_filter}}]-(y:{self.table_name})"
        else:
            raise ValueError(f"Unsupported traversal direction: {direction!r}")

        rel_filter = " {relation: $rel}" if relation is not None else ""
        hop_sql = (
            "MATCH "
            + edge_pattern.replace("{rel_filter}", rel_filter)
            + f" RETURN y.id AS id, y.payload AS payload LIMIT {int(limit)}"
        )

        visited = {str(start_id)}
        found: Dict[str, Dict] = {}
        frontier = [str(start_id)]
        for _ in range(depth):
            if not frontier or len(found) >= limit:
                break
            next_frontier = []
            for node_id in frontier:
                params: Dict[str, Any] = {"src": node_id}
                if relation is not None:
                    params["rel"] = relation
                for row in self._execute(hop_sql, params):
                    neighbor = row["id"]
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    if neighbor not in found:
                        found[neighbor] = row
                        next_frontier.append(neighbor)
                        if len(found) >= limit:
                            break
            frontier = next_frontier
        return [
            NeuGOutput(id=node_id, score=None, payload=self._parse_payload(row.get("payload")))
            for node_id, row in found.items()
        ]
