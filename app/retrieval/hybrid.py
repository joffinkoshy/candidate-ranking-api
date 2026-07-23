"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

RRF merges two ranked lists using only their *ranks*, so we don't have to
reconcile incompatible score scales (cosine similarity vs. BM25 scores). Each
list contributes 1 / (k + rank) to every item; items ranked highly by either
retriever bubble to the top.
"""

from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def retrieve_evidence(index, job_text: str, job_embedding, candidate_id: str,
                      k: int) -> list[str]:
    """Top-k evidence passages for a candidate, fusing dense + keyword search."""
    dense_ids = [h["id"] for h in index.dense(job_embedding, candidate_id, k)]
    sparse_ids = index.sparse(job_text, candidate_id, k)
    fused = rrf([dense_ids, sparse_ids])[:k]
    return [index.text(cid) for cid in fused if index.text(cid)]
