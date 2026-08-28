"""Hybrid retrieval via Reciprocal Rank Fusion (RRF), with an optional
cross-encoder reranking pass over the fused pool (see reranker.py).

RRF merges two ranked lists using only their *ranks*, so we don't have to
reconcile incompatible score scales (cosine similarity vs. BM25 scores). Each
list contributes 1 / (k + rank) to every item; items ranked highly by either
retriever bubble to the top.
"""

from __future__ import annotations

from app.config import settings


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def retrieve_evidence(index, job_text: str, job_embedding, candidate_id: str,
                      k: int) -> list[str]:
    """Top-k evidence passages for a candidate, fusing dense + keyword search,
    then (optionally) reranked by a cross-encoder before returning."""
    pool_k = max(k * settings.RERANK_POOL_MULTIPLIER, k) if settings.ENABLE_RERANK else k

    dense_ids = [h["id"] for h in index.dense(job_embedding, candidate_id, pool_k)]
    sparse_ids = index.sparse(job_text, candidate_id, pool_k)
    fused = rrf([dense_ids, sparse_ids])[:pool_k]
    passages = [(cid, index.text(cid)) for cid in fused if index.text(cid)]

    if settings.ENABLE_RERANK and passages:
        from app.retrieval.reranker import rerank

        top_ids = rerank(job_text, passages, k)
        text_by_id = dict(passages)
        return [text_by_id[cid] for cid in top_ids]

    return [text for _, text in passages[:k]]
