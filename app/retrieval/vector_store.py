"""Dense (semantic) retrieval over resume chunks, backed by ChromaDB.

We use an in-memory (ephemeral) index built per ranking request: ranking happens
"within a single job", so the candidate pool for one request is self-contained.
Embeddings are computed by us and passed in explicitly (cosine space), so Chroma
never needs to load its own embedding model.
"""

from __future__ import annotations

import numpy as np


class VectorStore:
    def __init__(self, name: str = "resume_chunks"):
        import chromadb

        self._client = chromadb.EphemeralClient()
        self.collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], embeddings: np.ndarray,
            documents: list[str], metadatas: list[dict]) -> None:
        self.collection.add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, embedding: np.ndarray, k: int,
              where: dict | None = None) -> list[dict]:
        """Return up to k hits: {id, document, metadata, similarity} (cosine, 0..1)."""
        res = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            {"id": i, "document": d, "metadata": m, "similarity": 1.0 - float(dist)}
            for i, d, m, dist in zip(ids, docs, metas, dists)
        ]
