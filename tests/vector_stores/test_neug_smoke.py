"""Real-engine smoke test for the NeuG vector store adapter.

Requires the NeuG Python bindings (importable as `neug`) with the
vector_search and fts extensions. Skipped automatically when unavailable
(e.g. in CI), run it in an environment with a local NeuG build:

    pytest tests/vector_stores/test_neug_smoke.py -v
"""

import uuid

import pytest

neug = pytest.importorskip("neug", reason="NeuG bindings not available")

from mem0.vector_stores.neug import NeuG  # noqa: E402

DIM = 8


def _vec(seed: float) -> list:
    """Deterministic pseudo-embedding; distinct for each seed, entries in [-1, 1]."""
    return [((seed * 0.37 + i * 0.61) % 2.0) - 1.0 for i in range(DIM)]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    db_path = str(tmp_path_factory.mktemp("neug_smoke"))
    s = NeuG(collection_name="smoke_col", embedding_model_dims=DIM, db_path=db_path, distance="cosine")
    yield s
    s.close()


@pytest.fixture(scope="module")
def seeded_store(store):
    """Insert 20 memories split across two users."""
    vectors, payloads, ids = [], [], []
    for i in range(20):
        memory_id = str(uuid.uuid4())
        user = "alice" if i % 2 == 0 else "bob"
        payloads.append(
            {
                "data": f"memory number {i} about graph databases and vector search",
                "hash": f"hash-{i}",
                "user_id": user,
                "idx": i,
                "created_at": "2026-01-01T00:00:00",
            }
        )
        vectors.append(_vec(i))
        ids.append(memory_id)
    store.insert(vectors, payloads=payloads, ids=ids)
    store._smoke_ids = ids
    store._smoke_payloads = payloads
    return store


def test_collection_created(seeded_store):
    assert seeded_store.table_name in seeded_store.list_cols()
    info = seeded_store.col_info()
    assert info["vector_size"] == DIM


def test_get_roundtrip(seeded_store):
    memory_id = seeded_store._smoke_ids[0]
    result = seeded_store.get(memory_id)
    assert result is not None
    assert result.id == memory_id
    assert result.payload["data"] == seeded_store._smoke_payloads[0]["data"]
    assert result.payload["user_id"] == "alice"
    assert seeded_store.get("does-not-exist") is None


def test_ann_search_similarity_contract(seeded_store):
    # Default search is the HNSW IndexScan path with client-side re-ranking.
    query_vec = _vec(0)  # identical to the first inserted vector
    results = seeded_store.search("memory zero", query_vec, top_k=5, filters={"user_id": "alice"})
    assert len(results) > 0
    top = results[0]
    assert top.id == seeded_store._smoke_ids[0]  # exact match must rank first
    assert top.score == pytest.approx(1.0, abs=1e-3)  # cosine self-similarity
    assert all(r.score <= top.score for r in results)  # sorted desc
    assert all(0.0 <= r.score <= 1.0 for r in results)
    assert all(r.payload.get("user_id") == "alice" for r in results)

    # The exact full-scan path must agree on the top hit.
    exact = seeded_store.search("memory zero", query_vec, top_k=5, filters={"user_id": "alice"}, exact=True)
    assert exact[0].id == top.id


def test_search_client_side_metadata_filter(seeded_store):
    results = seeded_store.search(
        "graph", _vec(3), top_k=5, filters={"user_id": "alice", "idx": 3}
    )
    # idx=3 belongs to bob, so no row can satisfy both filters
    assert results == []

    results = seeded_store.search("graph", _vec(3), top_k=5, filters={"idx": {"gte": 10}})
    assert len(results) > 0
    assert all(r.payload["idx"] >= 10 for r in results)


def test_search_empty_in_filter_returns_empty(seeded_store):
    assert seeded_store.search("graph", _vec(0), filters={"user_id": []}) == []


def test_keyword_search_bm25(seeded_store):
    results = seeded_store.keyword_search("vector search", top_k=5, filters={"user_id": "alice"})
    assert results is not None and len(results) > 0
    for r in results:
        assert r.score > 0  # negated bm25 -> positive raw score for mem0 sigmoid
        assert r.payload.get("user_id") == "alice"
        assert "vector search" in r.payload["data"]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_keyword_search_hyphenated_tokens(seeded_store):
    # Regression: an unquoted hyphenated token made NeuG FTS (SQLite FTS5)
    # raise "no such column" and keyword_search return None. The adapter must
    # quote tokens so such queries hit instead of aborting.
    seeded_store.insert(
        [_vec(1)],
        payloads=[{"data": "self-hosted graph-database benchmark runs end-to-end", "user_id": "alice"}],
        ids=["smoke-hyphen"],
    )
    results = seeded_store.keyword_search("graph-database self-hosted", top_k=5)
    assert results is not None
    assert any(r.id == "smoke-hyphen" for r in results)
    seeded_store.delete("smoke-hyphen")  # module-scoped store: keep row counts stable


def test_list_with_filters_returns_wrapped_shape(seeded_store):
    alice = seeded_store.list(filters={"user_id": "alice"}, top_k=100)
    assert isinstance(alice, list) and len(alice) == 1  # mem0 wrapped [[...]] shape
    rows = alice[0]
    assert len(rows) == 10
    assert all(m.payload["user_id"] == "alice" for m in rows)
    assert all(m.id in seeded_store._smoke_ids for m in rows)


def test_update_payload_and_vector(seeded_store):
    memory_id = seeded_store._smoke_ids[1]
    new_payload = dict(seeded_store._smoke_payloads[1])
    new_payload["data"] = "updated memory about hnsw indexes"
    new_payload["user_id"] = "alice"
    new_vec = _vec(100)
    seeded_store.update(memory_id, vector=new_vec, payload=new_payload)

    result = seeded_store.get(memory_id)
    assert result.payload["data"] == "updated memory about hnsw indexes"
    assert result.payload["user_id"] == "alice"

    # The default ANN path over-fetches and re-ranks client-side, so the
    # updated vector must surface correctly despite the engine's stale HNSW
    # ranking (20 rows fit entirely inside the 50-row over-fetch window).
    hits = seeded_store.search("hnsw", new_vec, top_k=1, filters={"user_id": "alice"})
    assert hits and hits[0].id == memory_id
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)

    # BM25 must follow the SET text.
    kw = seeded_store.keyword_search("hnsw indexes", top_k=10)
    assert any(r.id == memory_id for r in kw)


def test_graph_edges_and_traversal(seeded_store):
    a, b, c = seeded_store._smoke_ids[4], seeded_store._smoke_ids[6], seeded_store._smoke_ids[8]
    assert seeded_store.add_edge(a, b, relation="supports") is True
    assert seeded_store.add_edge(b, c, relation="supports") is True
    assert seeded_store.add_edge(a, b, relation="supports") is False  # dedupe

    one_hop = seeded_store.traverse(a, depth=1)
    assert [r.id for r in one_hop] == [b]
    two_hop = seeded_store.traverse(a, depth=2)
    assert {r.id for r in two_hop} == {b, c}
    inbound = seeded_store.traverse(c, depth=2, direction="in")
    assert {r.id for r in inbound} == {a, b}
    filtered = seeded_store.traverse(a, depth=2, relation="supports")
    assert {r.id for r in filtered} == {b, c}
    assert seeded_store.traverse(a, depth=2, relation="contradicts") == []

    seeded_store.remove_edge(a, b, relation="supports")
    assert [r.id for r in seeded_store.traverse(a, depth=1)] == []
    # b->c still intact
    assert [r.id for r in seeded_store.traverse(b, depth=1)] == [c]


def test_delete(seeded_store):
    memory_id = seeded_store._smoke_ids[19]
    seeded_store.delete(memory_id)
    assert seeded_store.get(memory_id) is None
    remaining = seeded_store.list(top_k=100)[0]
    assert memory_id not in [m.id for m in remaining]


def test_delete_removes_edges(seeded_store):
    a, b = seeded_store._smoke_ids[10], seeded_store._smoke_ids[12]
    seeded_store.add_edge(a, b, relation="links")
    seeded_store.delete(b)
    assert seeded_store.traverse(a, depth=1) == []  # DETACH DELETE cleaned the edge


def test_upsert_on_duplicate_id(seeded_store):
    memory_id = seeded_store._smoke_ids[2]
    seeded_store.insert([_vec(200)], payloads=[{"data": "reinserted", "user_id": "bob"}], ids=[memory_id])
    result = seeded_store.get(memory_id)
    assert result.payload["data"] == "reinserted"
    total = seeded_store.list(top_k=100)[0]
    assert len([m for m in total if m.id == memory_id]) == 1  # no duplicate rows


def test_second_store_on_same_db_path(tmp_path_factory):
    """mem0's telemetry/entity stores open the same db_path; handles are shared."""
    db_path = str(tmp_path_factory.mktemp("neug_shared"))
    main = NeuG(collection_name="main_col", embedding_model_dims=DIM, db_path=db_path)
    telemetry = NeuG(collection_name="mem0migrations", embedding_model_dims=DIM, db_path=db_path)
    assert main._db is telemetry._db
    main.insert([_vec(1)], payloads=[{"data": "hello shared db", "user_id": "u"}], ids=["m1"])
    assert telemetry.get("m1") is None  # separate tables
    assert main.get("m1") is not None
    telemetry.close()
    assert main.get("m1") is not None  # shared database still alive
    main.close()


def test_reset_recreates_empty_collection(seeded_store):
    seeded_store.reset()
    assert seeded_store.list(top_k=100) == [[]]
    assert seeded_store.table_name in seeded_store.list_cols()
