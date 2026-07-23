"""Sparse keyword retrieval (BM25).

Embeddings capture meaning but blur exact tokens: a job needing "CUDA" or "BM25"
should reward a resume that literally says it. BM25 covers that failure mode of
dense search — which is exactly why we combine the two (see hybrid.py).
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        from rank_bm25 import BM25Okapi

        self.ids = ids
        self.metadatas = metadatas
        self._bm25 = BM25Okapi([_tokenize(d) for d in documents]) if documents else None

        # position lists per candidate, so we can score globally then filter.
        self._positions_by_candidate: dict[str, list[int]] = {}
        for pos, meta in enumerate(metadatas):
            self._positions_by_candidate.setdefault(
                meta["candidate_id"], []).append(pos)

    def query(self, text: str, candidate_id: str, k: int) -> list[str]:
        """Return the candidate's top-k chunk ids by BM25 relevance to `text`."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        positions = self._positions_by_candidate.get(candidate_id, [])
        ranked = sorted(positions, key=lambda p: scores[p], reverse=True)[:k]
        return [self.ids[p] for p in ranked]
