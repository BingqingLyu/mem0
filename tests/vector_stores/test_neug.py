"""Unit tests for the NeuG vector store adapter (neug bindings are mocked)."""

import json
import sys
import types

import pytest


class FakeQueryResult:
    def __init__(self, rows=None):
        # The real binding returns a JSON string; mirror that contract.
        self._payload = json.dumps({"table": rows or []})

    def get_bolt_response(self):
        return self._payload


class FakeConnection:
    # One-shot hook: raise on the next query whose prefix matches.
    fail_query_prefix = None

    def __init__(self, schema_tables=None):
        self.calls = []
        self.closed = False
        self._schema_tables = schema_tables or []
        self.next_result = None
        self.result_queue = []  # FIFO of canned results (consumed before next_result)
        # Fail the N-th statement starting with "CREATE (:" (0-based, across
        # batched and replayed writes) with the given error message.
        self.fail_create_indexes = {}
        self._create_calls = 0

    def execute(self, query, access_mode="", parameters=None):
        self.calls.append((query, parameters))
        if self.fail_query_prefix is not None and query.startswith(self.fail_query_prefix):
            prefix = self.fail_query_prefix
            self.fail_query_prefix = None
            raise RuntimeError(f"forced failure on {prefix.strip()}")
        if query.startswith("CREATE (:"):
            index = self._create_calls
            self._create_calls += 1
            if index in self.fail_create_indexes:
                raise RuntimeError(self.fail_create_indexes[index])
        if self.result_queue:
            return self.result_queue.pop(0)
        if self.next_result is not None:
            result, self.next_result = self.next_result, None
            return result
        return FakeQueryResult()

    def get_schema(self):
        return json.dumps(
            {"schema": {"vertex_types": [{"type_name": t} for t in self._schema_tables]}}
        )

    def is_open(self):
        return not self.closed

    def close(self):
        self.closed = True


class FakeDatabase:
    instances = []

    def __init__(self, path):
        self.path = path
        self.conn = FakeConnection()
        self.closed = False
        FakeDatabase.instances.append(self)

    def connect(self):
        return self.conn

    def close(self):
        self.closed = True


def _install_fake_neug():
    fake_module = types.ModuleType("neug")
    fake_module.Database = FakeDatabase
    sys.modules["neug"] = fake_module


@pytest.fixture
def neug_store():
    """A NeuG store backed by a fake `neug` module."""
    saved = sys.modules.get("neug")
    FakeDatabase.instances.clear()
    _install_fake_neug()
    store = None
    try:
        # Import after the fake module is installed.
        from mem0.vector_stores.neug import NeuG

        store = NeuG(collection_name="test_col", embedding_model_dims=4, db_path="/tmp/fake_neug")
        store._conn.calls.clear()
        yield store
    finally:
        if store is not None:
            store.close()
        if saved is not None:
            sys.modules["neug"] = saved
        else:
            sys.modules.pop("neug", None)


def _last_call(store):
    return store._conn.calls[-1]


def test_create_col_creates_table_indexes_and_rel_table(neug_store):
    neug_store.create_col("new_col", 8, "cosine")
    queries = [q for q, _ in neug_store._conn.calls]
    assert any("CREATE NODE TABLE new_col" in q and "vector FLOAT[8]" in q for q in queries)
    assert any("USING HNSW (vector) WITH (metric = 'cosine')" in q for q in queries)
    assert any("USING FTS (text)" in q for q in queries)
    assert any("CREATE REL TABLE new_col_mem0_links (FROM new_col TO new_col" in q for q in queries)


def test_create_col_skips_existing_table(neug_store):
    neug_store._conn._schema_tables = ["existing_col"]
    neug_store.create_col("existing_col", 4, "cosine")
    assert neug_store._conn.calls == []


def test_create_col_l2_metric(neug_store):
    neug_store.create_col("l2_col", 4, "l2")
    queries = [q for q, _ in neug_store._conn.calls]
    assert any("metric = 'l2'" in q for q in queries)


def test_insert_builds_create_with_params(neug_store):
    payload = {"data": "hello world", "user_id": "u1", "hash": "h1"}
    neug_store.insert([[0.1, 0.2, 0.3, 0.4]], payloads=[payload], ids=["m1"])
    query, params = _last_call(neug_store)
    # Batched write: one multi-row CREATE with per-row suffixed parameters.
    assert query.startswith("CREATE (:test_col")
    assert params["id0"] == "m1"
    assert params["vec0"] == [0.1, 0.2, 0.3, 0.4]
    assert params["text0"] == "hello world"
    assert json.loads(params["payload0"]) == payload
    assert params["user_id0"] == "u1"
    assert params["agent_id0"] is None


def test_insert_batches_rows_into_multi_row_creates(neug_store):
    from mem0.vector_stores import neug as neug_mod

    n = neug_mod._INSERT_BATCH_SIZE + 3  # two full/partial batches
    neug_store.insert(
        [[0.1, 0.2, 0.3, 0.4]] * n,
        payloads=[{"data": f"d{i}"} for i in range(n)],
        ids=[f"m{i}" for i in range(n)],
    )
    creates = [(q, p) for q, p in neug_store._conn.calls if q.startswith("CREATE (")]
    assert len(creates) == 2
    assert creates[0][1]["id0"] == "m0"
    assert creates[0][1][f"id{neug_mod._INSERT_BATCH_SIZE - 1}"] == f"m{neug_mod._INSERT_BATCH_SIZE - 1}"
    assert creates[1][1]["id0"] == f"m{neug_mod._INSERT_BATCH_SIZE}"
    assert creates[1][1]["id2"] == f"m{n - 1}"


def test_insert_prefers_text_lemmatized(neug_store):
    payload = {"data": "raw text", "text_lemmatized": "lemmatized text"}
    neug_store.insert([[0.0, 0.0, 0.0, 0.0]], payloads=[payload], ids=["m1"])
    _, params = _last_call(neug_store)
    assert params["text0"] == "lemmatized text"


def test_insert_without_payloads_or_ids(neug_store):
    neug_store.insert([[0.1, 0.2, 0.3, 0.4]])
    _, params = _last_call(neug_store)
    assert params["payload0"] == "{}"
    assert params["id0"]  # auto-generated uuid


def test_insert_validates_lengths(neug_store):
    with pytest.raises(ValueError, match="payloads length"):
        neug_store.insert([[0.1, 0.2, 0.3, 0.4]], payloads=[])
    with pytest.raises(ValueError, match="ids length"):
        neug_store.insert([[0.1, 0.2, 0.3, 0.4]], payloads=[{"data": "x"}], ids=["a", "b"])


def test_insert_duplicate_falls_back_to_set_and_verifies(neug_store):
    # CREATE #0 is the batch (fails, rolls back atomically); the replay then
    # succeeds for m1 (#1) and hits the conflict again for m2 (#2), which
    # falls back to SET. Every successful statement consumes one queued
    # result, so queue one for the m1 replay, the SET, and the verification.
    neug_store._conn.fail_create_indexes = {
        0: "Error code: 1009, primary key conflict",
        2: "Error code: 1009, primary key conflict",
    }
    neug_store._conn.result_queue = [
        FakeQueryResult(),  # replayed CREATE for m1 succeeds
        FakeQueryResult(),  # SET returns no rows
        FakeQueryResult([{"id": "m2", "payload": "{}"}]),  # verification get
    ]
    neug_store.insert(
        [[0.1, 0.2, 0.3, 0.4]] * 2,
        payloads=[{"data": "a"}, {"data": "dup"}],
        ids=["m1", "m2"],
    )
    queries = [q for q, _ in neug_store._conn.calls]
    assert any("SET" in q for q in queries)


def test_insert_duplicate_fallback_raises_when_row_missing(neug_store):
    # Batch CREATE (#0) fails, the replayed single-row CREATE (#1) hits the
    # conflict again, and the verification `get` finds nothing -> the
    # original error must propagate.
    neug_store._conn.fail_create_indexes = {
        0: "Error code: 1009, primary key conflict",
        1: "Error code: 1009, primary key conflict",
    }
    with pytest.raises(RuntimeError, match="1009"):
        neug_store.insert([[0.1, 0.2, 0.3, 0.4]], payloads=[{"data": "dup"}], ids=["m1"])


def test_search_default_is_ann_index_scan(neug_store):
    neug_store._conn.next_result = FakeQueryResult(
        [{"id": "m1", "payload": json.dumps({"data": "hi", "user_id": "u1"}), "d": 0.0}]
    )
    results = neug_store.search("hi", [0.1, 0.2, 0.3, 0.4], top_k=3)
    query, params = _last_call(neug_store)
    # Default path: HNSW IndexScan with bound query vector; the engine projects
    # the distance, no stored-vector transfer.
    assert "vector_distance_cosine(m.vector, $q) AS d" in query
    assert "ORDER BY d ASC LIMIT 3" in query
    assert "m.vector AS vector" not in query
    assert params["q"] == [0.1, 0.2, 0.3, 0.4]
    assert len(results) == 1
    assert results[0].id == "m1"
    assert results[0].score == pytest.approx(1.0)  # 1 - engine distance
    assert results[0].payload["data"] == "hi"


def test_search_overfetches_only_for_client_filters(neug_store):
    neug_store.search("q", [0.0, 0.0, 0.0, 0.0], top_k=3, filters={"custom": "y"})
    query, _ = _last_call(neug_store)
    assert "LIMIT 50" in query  # client filters may drop rows, so widen the pool

    neug_store._conn.calls.clear()
    neug_store.search("q", [0.0, 0.0, 0.0, 0.0], top_k=3, filters={"user_id": "u1"})
    query, _ = _last_call(neug_store)
    assert "LIMIT 3" in query  # server-side filters need no over-fetch


def test_search_exact_uses_full_scan(neug_store):
    neug_store._conn.next_result = FakeQueryResult(
        [{"id": "m1", "payload": json.dumps({"data": "hi"}), "vector": [0.1, 0.2, 0.3, 0.4]}]
    )
    neug_store.search("hi", [0.1, 0.2, 0.3, 0.4], top_k=3, filters={"user_id": "u1"}, exact=True)
    query, params = _last_call(neug_store)
    # Exact path: full scan, no engine-side ordering or truncation, no bound vector.
    assert "ORDER BY" not in query
    assert "LIMIT" not in query
    assert "WHERE m.user_id = $user_id" in query
    assert params == {"user_id": "u1"}


def test_search_score_comes_from_engine_distance(neug_store):
    # Since alibaba/neug#931 is fixed (5300cb3), ANN scores derive from the
    # engine-projected distance: cosine similarity = 1 - distance.
    neug_store._conn.next_result = FakeQueryResult(
        [
            {"id": "m1", "payload": "{}", "d": 0.3},
            {"id": "m2", "payload": "{}", "d": 0.0},
        ]
    )
    results = neug_store.search("q", [-1.0, -1.0], top_k=2)
    assert [r.id for r in results] == ["m2", "m1"]  # sorted by similarity desc
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.7)


def test_search_l2_missing_vector_scores_zero():
    from mem0.vector_stores.neug import _l2_similarity

    assert _l2_similarity([], [1.0, 0.0]) == 0.0
    assert _l2_similarity([1.0], [1.0, 0.0]) == 0.0  # dim mismatch
    assert _l2_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_search_applies_client_filters(neug_store):
    neug_store._conn.next_result = FakeQueryResult(
        [
            {"id": "m1", "payload": json.dumps({"data": "a", "custom": "x"}), "d": 0.1},
            {"id": "m2", "payload": json.dumps({"data": "b", "custom": "y"}), "d": 0.2},
        ]
    )
    results = neug_store.search("q", [0.0, 0.0, 0.0, 0.0], top_k=1, filters={"custom": "y"})
    assert len(results) == 1
    assert results[0].id == "m2"


def test_search_client_filter_type_mismatch_is_not_an_error():
    from mem0.vector_stores.neug import NeuG

    # Incomparable types must be treated as non-matching, not raise TypeError.
    assert not NeuG._matches_client_filters({"category": "text"}, {"category": {"gt": 3}})


def test_search_empty_in_filter_returns_empty(neug_store):
    assert neug_store.search("q", [0.0, 0.0, 0.0, 0.0], filters={"user_id": {"in": []}}) == []
    assert neug_store.search("q", [0.0, 0.0, 0.0, 0.0], filters={"user_id": []}) == []
    assert neug_store._conn.calls == []  # short-circuited before any query


def test_search_in_filter_must_be_list(neug_store):
    with pytest.raises(ValueError, match="requires a list"):
        neug_store.search("q", [0.0, 0.0, 0.0, 0.0], filters={"user_id": {"in": "abc"}})


def test_search_in_filter_expands_to_or(neug_store):
    neug_store.search("q", [0.0, 0.0, 0.0, 0.0], top_k=2, filters={"user_id": ["u1", "u2"]})
    query, params = _last_call(neug_store)
    assert "m.user_id = $user_id_0 OR m.user_id = $user_id_1" in query
    assert params["user_id_0"] == "u1"
    assert params["user_id_1"] == "u2"


def test_keyword_search_negates_bm25_score(neug_store):
    neug_store._conn.next_result = FakeQueryResult(
        [{"id": "m1", "payload": json.dumps({"data": "hello"}), "score": -3.5}]
    )
    results = neug_store.keyword_search("hello", top_k=5)
    query, params = _last_call(neug_store)
    assert "bm25(m.text, $q)" in query
    assert "ORDER BY score ASC LIMIT 5" in query
    assert params["q"] == "hello"
    assert results[0].score == pytest.approx(3.5)


def test_keyword_search_empty_query_returns_none(neug_store):
    assert neug_store.keyword_search("", top_k=5) is None
    assert neug_store._conn.calls == []


def test_keyword_search_empty_in_filter_returns_empty(neug_store):
    assert neug_store.keyword_search("hello", filters={"user_id": []}) == []


def test_update_sets_vector_and_payload(neug_store):
    payload = {"data": "updated", "user_id": "u9"}
    neug_store.update("m1", vector=[1.0, 0.0, 0.0, 0.0], payload=payload)
    query, params = _last_call(neug_store)
    assert "SET m.vector = $vec, m.text = $text, m.payload = $payload" in query
    assert "m.user_id = $user_id" in query
    assert params["id"] == "m1"
    assert params["user_id"] == "u9"


def test_update_payload_only(neug_store):
    neug_store.update("m1", payload={"data": "text only"})
    query, _ = _last_call(neug_store)
    assert "m.vector" not in query
    assert "m.payload = $payload" in query


def test_update_skips_null_promoted_columns(neug_store):
    # NeuG rejects SET col = NULL (error 1011): promoted columns absent from
    # the payload must be omitted from the SET, not bound as NULL.
    neug_store.update("m1", payload={"data": "x", "user_id": "u1"})
    query, params = _last_call(neug_store)
    assert "m.user_id = $user_id" in query
    assert "m.agent_id" not in query
    assert "agent_id" not in params


def test_update_noop_without_args(neug_store):
    neug_store.update("m1")
    assert neug_store._conn.calls == []


def test_get_returns_output_or_none(neug_store):
    neug_store._conn.next_result = FakeQueryResult(
        [{"id": "m1", "payload": json.dumps({"data": "stored"})}]
    )
    result = neug_store.get("m1")
    assert result.id == "m1"
    assert result.payload["data"] == "stored"

    neug_store._conn.next_result = FakeQueryResult([])
    assert neug_store.get("missing") is None


def test_delete_uses_detach_delete(neug_store):
    neug_store.delete("m1")
    query, params = _last_call(neug_store)
    assert "DETACH DELETE" in query
    assert params == {"id": "m1"}


def test_list_returns_wrapped_shape(neug_store):
    rows = [
        {"id": "m1", "payload": json.dumps({"data": "a", "user_id": "u1"})},
        {"id": "m2", "payload": json.dumps({"data": "b", "user_id": "u1"})},
    ]
    neug_store._conn.next_result = FakeQueryResult(rows)
    results = neug_store.list(filters={"user_id": "u1"}, top_k=1)
    query, params = _last_call(neug_store)
    assert "WHERE m.user_id = $user_id" in query
    assert params["user_id"] == "u1"
    # mem0's delete_all/get_all expect the wrapped [[...]] shape.
    assert isinstance(results, list) and isinstance(results[0], list)
    assert len(results[0]) == 1
    assert results[0][0].score is None


def test_list_default_top_k_is_unlimited(neug_store):
    rows = [{"id": f"m{i}", "payload": "{}"} for i in range(3)]
    neug_store._conn.next_result = FakeQueryResult(rows)
    results = neug_store.list()
    assert len(results[0]) == 3  # no silent truncation to 100


def test_list_passes_parameters_keyword_without_filters(neug_store):
    # With no server filters params is empty; the query must still run.
    neug_store._conn.next_result = FakeQueryResult([{"id": "m1", "payload": "{}"}])
    results = neug_store.list()
    assert len(results[0]) == 1


def test_list_empty_in_filter_returns_wrapped_empty(neug_store):
    assert neug_store.list(filters={"user_id": []}) == [[]]
    assert neug_store._conn.calls == []


def test_list_cols_reads_schema(neug_store):
    neug_store._conn._schema_tables = ["mem0", "other_table"]
    assert neug_store.list_cols() == ["mem0", "other_table"]


def test_delete_col_drops_indexes_rel_table_then_table(neug_store):
    neug_store.delete_col()
    queries = [q for q, _ in neug_store._conn.calls]
    assert queries == [
        "DROP INDEX test_col_mem0_fts",
        "DROP INDEX test_col_mem0_hnsw",
        "DROP TABLE test_col_mem0_links",
        "DROP TABLE test_col",
    ]


def test_reset_recreates_collection(neug_store):
    neug_store.reset()
    queries = [q for q, _ in neug_store._conn.calls]
    assert "DROP TABLE test_col" in queries
    assert any(q.startswith("CREATE NODE TABLE test_col") for q in queries)


def test_sanitize_table_name():
    from mem0.vector_stores.neug import _sanitize_table_name

    assert _sanitize_table_name("mem0") == "mem0"
    # Names needing sanitization get a stable hash suffix to avoid collisions.
    dashed = _sanitize_table_name("my-collection.v2")
    underscored = _sanitize_table_name("my_collection_v2")
    assert dashed.startswith("my_collection_v2_")
    assert dashed != underscored
    assert _sanitize_table_name("my-collection.v2") == dashed  # deterministic
    assert _sanitize_table_name("123abc").startswith("m_123abc")


def test_client_filter_operators():
    from mem0.vector_stores.neug import NeuG

    match = NeuG._matches_client_filters
    payload = {"score": 5, "name": "Hello World", "tag": "a"}
    assert match(payload, {"score": {"gt": 4}})
    assert not match(payload, {"score": {"gt": 5}})
    assert match(payload, {"score": {"gte": 5, "lte": 5}})
    assert match(payload, {"tag": {"in": ["a", "b"]}})
    assert match(payload, {"tag": {"nin": ["b"]}})
    assert match(payload, {"name": {"contains": "World"}})
    assert match(payload, {"name": {"icontains": "world"}})
    assert match(payload, {"tag": {"ne": "b"}})
    assert match(payload, {"missing": "*"})
    assert not match(payload, {"missing": "x"})
    assert match(payload, {"AND": [{"score": {"gt": 1}}, {"tag": "a"}]})
    assert match(payload, {"OR": [{"tag": "z"}, {"tag": "a"}]})
    assert not match(payload, {"NOT": [{"tag": "a"}]})


def test_add_edge_deduplicates(neug_store):
    # First call: count query returns 0 -> CREATE edge.
    neug_store._conn.next_result = FakeQueryResult([{"n": 0}])
    assert neug_store.add_edge("m1", "m2", relation="about", weight=0.7) is True
    queries = [q for q, _ in neug_store._conn.calls]
    assert any("count(r) AS n" in q for q in queries)
    assert any("CREATE (x)-[:test_col_mem0_links" in q for q in queries)

    # Second call: edge already exists -> no CREATE.
    neug_store._conn.calls.clear()
    neug_store._conn.next_result = FakeQueryResult([{"n": 1}])
    assert neug_store.add_edge("m1", "m2", relation="about") is False
    queries = [q for q, _ in neug_store._conn.calls]
    assert not any("CREATE (x)-[" in q for q in queries)


def test_remove_edge(neug_store):
    neug_store.remove_edge("m1", "m2", relation="about")
    query, params = _last_call(neug_store)
    assert "DELETE r" in query
    assert "{relation: $rel}" in query
    assert params == {"src": "m1", "dst": "m2", "rel": "about"}

    neug_store._conn.calls.clear()
    neug_store.remove_edge("m1", "m2")
    query, params = _last_call(neug_store)
    assert "{relation: $rel}" not in query
    assert params == {"src": "m1", "dst": "m2"}


def test_traverse_multi_hop_bfs(neug_store):
    # Hop 1 from m1 -> m2; hop 2 from m2 -> m3.
    neug_store._conn.next_result = FakeQueryResult([{"id": "m2", "payload": json.dumps({"data": "two"})}])
    results = neug_store.traverse("m1", depth=2)
    assert len(results) == 1
    assert results[0].id == "m2"
    assert neug_store._conn.next_result is None  # consumed exactly one hop result so far
    # Continue: store has no more canned results, so the frontier yields nothing.
    assert [r.id for r in results] == ["m2"]


def test_traverse_direction_validation(neug_store):
    with pytest.raises(ValueError, match="direction"):
        neug_store.traverse("m1", direction="sideways")


def test_traverse_in_direction_pattern(neug_store):
    neug_store._conn.next_result = FakeQueryResult([])
    neug_store.traverse("m1", direction="in")
    query, _ = _last_call(neug_store)
    assert ")<-[r:test_col_mem0_links]-(" in query


def test_shared_connection_across_stores():
    """Two stores on the same db_path must share one Database+connection.

    NeuG allows one open per directory (error 1004) and one read-write
    connection per Database (error 4001).
    """
    saved = sys.modules.get("neug")
    FakeDatabase.instances.clear()
    _install_fake_neug()
    try:
        from mem0.vector_stores.neug import NeuG

        first = NeuG(collection_name="col_a", embedding_model_dims=4, db_path="/tmp/shared_neug")
        second = NeuG(collection_name="col_b", embedding_model_dims=4, db_path="/tmp/shared_neug")
        assert first._db is second._db
        assert first._conn is second._conn
        assert len(FakeDatabase.instances) == 1

        first.close()
        assert not first._db.closed  # still referenced by `second`
        second.close()
        assert first._db.closed  # last release closes the shared handle
    finally:
        if saved is not None:
            sys.modules["neug"] = saved
        else:
            sys.modules.pop("neug", None)


def test_failed_init_releases_shared_connection():
    """A store whose init fails must give back its reference to the registry."""
    saved = sys.modules.get("neug")
    FakeDatabase.instances.clear()
    _install_fake_neug()
    try:
        from mem0.vector_stores import neug as neug_module
        from mem0.vector_stores.neug import NeuG

        FakeConnection.fail_query_prefix = "LOAD "
        try:
            with pytest.raises(RuntimeError, match="forced failure"):
                NeuG(collection_name="col_a", embedding_model_dims=4, db_path="/tmp/failed_init")
        finally:
            FakeConnection.fail_query_prefix = None

        # The failed init must not pin the shared Database open.
        assert FakeDatabase.instances[0].closed
        assert not neug_module._DATABASES

        # A later store on the same db_path opens a fresh Database.
        store = NeuG(collection_name="col_a", embedding_model_dims=4, db_path="/tmp/failed_init")
        assert len(FakeDatabase.instances) == 2
        store.close()
        assert FakeDatabase.instances[1].closed
    finally:
        if saved is not None:
            sys.modules["neug"] = saved
        else:
            sys.modules.pop("neug", None)


def test_double_close_does_not_release_twice():
    """Closing one store twice must not steal a reference from another store."""
    saved = sys.modules.get("neug")
    FakeDatabase.instances.clear()
    _install_fake_neug()
    try:
        from mem0.vector_stores.neug import NeuG

        first = NeuG(collection_name="col_a", embedding_model_dims=4, db_path="/tmp/shared_neug")
        second = NeuG(collection_name="col_b", embedding_model_dims=4, db_path="/tmp/shared_neug")
        first.close()
        first.close()
        assert not first._db.closed  # `second` still holds its reference
        second.close()
        assert first._db.closed
    finally:
        if saved is not None:
            sys.modules["neug"] = saved
        else:
            sys.modules.pop("neug", None)


def test_unsupported_distance_raises():
    saved = sys.modules.get("neug")
    _install_fake_neug()
    try:
        from mem0.vector_stores.neug import NeuG

        with pytest.raises(ValueError, match="Unsupported distance"):
            NeuG(collection_name="c", embedding_model_dims=4, distance="dot")
    finally:
        if saved is not None:
            sys.modules["neug"] = saved
        else:
            sys.modules.pop("neug", None)


def test_config_registers_neug_provider():
    from mem0.vector_stores.configs import VectorStoreConfig

    config = VectorStoreConfig(provider="neug", config={"collection_name": "mem0", "embedding_model_dims": 8})
    assert config.config.collection_name == "mem0"
    assert config.config.distance == "cosine"

    with pytest.raises(ValueError):
        VectorStoreConfig(provider="neug", config={"distance": "dot"})


def test_factory_maps_neug_provider():
    from mem0.utils.factory import VectorStoreFactory

    assert VectorStoreFactory.provider_to_class["neug"] == "mem0.vector_stores.neug.NeuG"


def test_close_is_idempotent(neug_store):
    neug_store.close()
    neug_store.close()
    assert neug_store._conn.closed
