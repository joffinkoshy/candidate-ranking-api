from app.retrieval.reranker import rerank


class _FakeCrossEncoder:
    """Scores a passage higher the more it overlaps with the query's words."""

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            q_words = set(query.lower().split())
            t_words = set(text.lower().split())
            scores.append(len(q_words & t_words))
        return scores


def test_rerank_empty_passages_returns_empty():
    assert rerank("python engineer", [], top_k=3) == []


def test_rerank_orders_by_relevance_to_query(monkeypatch):
    import app.retrieval.reranker as reranker_module

    monkeypatch.setattr(reranker_module, "_get_model", lambda: _FakeCrossEncoder())

    passages = [
        ("c1", "enjoys hiking and painting on weekends"),
        ("c2", "five years of python backend engineer experience"),
        ("c3", "some python scripting for data cleanup"),
    ]

    ranked_ids = rerank("python backend engineer", passages, top_k=2)

    assert ranked_ids == ["c2", "c3"]  # most word-overlap first, low-overlap passage dropped


def test_rerank_respects_top_k(monkeypatch):
    import app.retrieval.reranker as reranker_module

    monkeypatch.setattr(reranker_module, "_get_model", lambda: _FakeCrossEncoder())

    passages = [(f"c{i}", "python engineer") for i in range(5)]
    assert len(rerank("python engineer", passages, top_k=2)) == 2
