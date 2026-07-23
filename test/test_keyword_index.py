from app.retrieval.keyword_index import BM25Index, _tokenize


def test_tokenize_lowercases_and_splits():
    assert _tokenize("Python, RAG! CUDA-11") == ["python", "rag", "cuda", "11"]


def _build():
    ids = ["A::0", "A::1", "B::0"]
    docs = [
        "python machine learning engineer",
        "sales and customer service",
        "python rag embeddings vector search",
    ]
    metas = [
        {"candidate_id": "A"},
        {"candidate_id": "A"},
        {"candidate_id": "B"},
    ]
    return BM25Index(ids, docs, metas)


def test_query_filters_by_candidate():
    idx = _build()
    hits = idx.query("python engineer", candidate_id="A", k=5)
    assert all(h.startswith("A::") for h in hits)


def test_query_ranks_relevant_chunk_first():
    idx = _build()
    hits = idx.query("machine learning", candidate_id="A", k=2)
    assert hits[0] == "A::0"  # the ML chunk beats the sales chunk


def test_empty_index_returns_empty():
    idx = BM25Index([], [], [])
    assert idx.query("anything", candidate_id="A", k=3) == []
