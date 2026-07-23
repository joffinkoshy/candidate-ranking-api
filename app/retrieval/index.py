"""RetrievalIndex — bundles the dense (Chroma) and sparse (BM25) indexes over the
same set of resume chunks, so hybrid retrieval can query both from one object.
"""

from __future__ import annotations

import numpy as np

from app.retrieval.keyword_index import BM25Index
from app.retrieval.vector_store import VectorStore


class RetrievalIndex:
    def __init__(self):
        self.store = VectorStore()
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._metas: list[dict] = []
        self._text_by_id: dict[str, str] = {}
        self._bm25: BM25Index | None = None

    def add(self, ids: list[str], embeddings: np.ndarray,
            documents: list[str], metadatas: list[dict]) -> None:
        self.store.add(ids, embeddings, documents, metadatas)
        self._ids += list(ids)
        self._docs += list(documents)
        self._metas += list(metadatas)
        for cid, doc in zip(ids, documents):
            self._text_by_id[cid] = doc

    def finalize(self) -> None:
        """Build the BM25 index once all chunks are added."""
        self._bm25 = BM25Index(self._ids, self._docs, self._metas)

    def text(self, chunk_id: str) -> str:
        return self._text_by_id.get(chunk_id, "")

    def dense(self, job_embedding: np.ndarray, candidate_id: str,
              k: int) -> list[dict]:
        return self.store.query(job_embedding, k, where={"candidate_id": candidate_id})

    def sparse(self, job_text: str, candidate_id: str, k: int) -> list[str]:
        return self._bm25.query(job_text, candidate_id, k) if self._bm25 else []
