"""Ingestion: turn raw candidate resumes into a searchable hybrid index.

Pipeline per candidate: resume_text -> chunk -> embed -> store in dense (Chroma)
and sparse (BM25) indexes, each chunk tagged with its candidate_id so retrieval
can be filtered per candidate.
"""

from __future__ import annotations

from typing import List

from app.nlp.chunking import chunk_text
from app.nlp.embeddings import embed_texts
from app.retrieval.index import RetrievalIndex
from app.schemas import Candidate


def build_index(candidates: List[Candidate]) -> RetrievalIndex:
    index = RetrievalIndex()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for cand in candidates:
        for i, chunk in enumerate(chunk_text(cand.resume_text)):
            ids.append(f"{cand.candidate_id}::chunk::{i}")
            documents.append(chunk)
            metadatas.append({"candidate_id": cand.candidate_id, "chunk_index": i})

    if documents:
        index.add(ids, embed_texts(documents), documents, metadatas)

    index.finalize()
    return index
