"""Cross-encoder reranking over the RRF-fused candidate pool.

Dense and BM25 retrieval each score (query, passage) independently, and RRF
fuses them by rank only. A cross-encoder instead feeds the query and passage
into one transformer together, so it can model their interaction directly —
a deeper, more accurate signal than either retriever alone. That cost is too
high to run over a whole corpus, so it only reruns on the small pool RRF
already narrowed down, right before the top-k is handed to the LLM judge.
"""

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.RERANK_MODEL)


def rerank(query: str, passages: list[tuple[str, str]], top_k: int) -> list[str]:
    """passages: (chunk_id, text) pairs. Returns chunk_ids sorted by relevance to query."""
    if not passages:
        return []
    model = _get_model()
    pairs = [(query, text) for _, text in passages]
    scores = model.predict(pairs)
    ranked = sorted(zip((cid for cid, _ in passages), scores), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked[:top_k]]
