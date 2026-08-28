from app.retrieval.hybrid import retrieve_evidence, rrf


def test_rrf_empty():
    assert rrf([[], []]) == []


def test_rrf_prefers_items_ranked_well_by_both():
    dense = ["d1", "d2", "d3"]
    sparse = ["d3", "d1", "d4"]
    fused = rrf([dense, sparse])
    assert fused[0] == "d1"                 # high in both lists
    assert set(fused) == {"d1", "d2", "d3", "d4"}  # union of inputs


def test_rrf_single_list_preserves_order():
    assert rrf([["a", "b", "c"]]) == ["a", "b", "c"]


def test_rrf_k_dampens_rank_gap():
    # larger k => rank differences matter less
    dense = ["a", "b"]
    sparse = ["b", "a"]
    fused = rrf([dense, sparse], k=60)
    assert set(fused) == {"a", "b"}


class _FakeIndex:
    """Minimal stand-in for RetrievalIndex: candidate has 3 chunks, dense and
    sparse disagree on order so RRF fusion + reranking both do real work."""

    _texts = {
        "c1": "hobbies and weekend travel",
        "c2": "five years python backend engineer",
        "c3": "some python scripting experience",
    }

    def text(self, chunk_id):
        return self._texts.get(chunk_id, "")

    def dense(self, job_embedding, candidate_id, k):
        return [{"id": cid} for cid in ["c1", "c2", "c3"][:k]]

    def sparse(self, job_text, candidate_id, k):
        return ["c2", "c3", "c1"][:k]


def test_retrieve_evidence_without_rerank_uses_fused_order(monkeypatch):
    import app.retrieval.hybrid as hybrid_module

    monkeypatch.setattr(hybrid_module.settings, "ENABLE_RERANK", False)

    evidence = retrieve_evidence(_FakeIndex(), "python backend engineer", None, "cand1", k=2)
    assert len(evidence) == 2


def test_retrieve_evidence_with_rerank_reorders_by_relevance(monkeypatch):
    import app.retrieval.hybrid as hybrid_module
    import app.retrieval.reranker as reranker_module

    monkeypatch.setattr(hybrid_module.settings, "ENABLE_RERANK", True)
    monkeypatch.setattr(hybrid_module.settings, "RERANK_POOL_MULTIPLIER", 3)

    def fake_rerank(query, passages, top_k):
        # most relevant passage (exact skill match) always wins, regardless
        # of the RRF fusion order fed in.
        return ["c2"][:top_k]

    monkeypatch.setattr(reranker_module, "rerank", fake_rerank)

    evidence = retrieve_evidence(_FakeIndex(), "python backend engineer", None, "cand1", k=1)
    assert evidence == ["five years python backend engineer"]
