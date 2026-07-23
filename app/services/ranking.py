"""Core ranking orchestrator.

Full pipeline (when an LLM key is configured):
  1. Normalize structured features per request (fair comparison within one job).
  2. Structured score  <- sklearn regression model.
  3. Build a hybrid index (dense Chroma + sparse BM25) over resume chunks.
  4. Per candidate: extract a structured profile (small LLM), retrieve top-k
     evidence via hybrid search + RRF, and have a strong LLM judge score the fit
     with grounded citations.
  5. Blend structured score + judge score, rank deterministically.

Without a key (settings.use_llm == False) it degrades gracefully to Phase 1:
dense-retrieval semantic score blended with the structured score.
"""

from typing import List

import joblib
import numpy as np

from app.config import settings
from app.nlp.embeddings import embed_text
from app.retrieval.hybrid import retrieve_evidence
from app.retrieval.index import RetrievalIndex
from app.schemas import Candidate, RankedCandidate
from app.services.ingestion import build_index

_MODEL = None


def _get_ml_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(settings.ml_model_path())
    return _MODEL


def min_max_normalize(values: List[float]) -> List[float]:
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return [0.5 for _ in values]
    return [(v - min_val) / (max_val - min_val) for v in values]


def _dense_semantic_score(index: RetrievalIndex, job_embedding: np.ndarray,
                          candidate_id: str) -> float:
    """Mean similarity of a candidate's top-k most job-relevant chunks (0..1)."""
    hits = index.dense(job_embedding, candidate_id, settings.TOP_K)
    return float(np.mean([h["similarity"] for h in hits])) if hits else 0.0


def rank_candidates_logic(candidates: List[Candidate],
                          job_description: str) -> List[RankedCandidate]:
    if not candidates:
        return []

    norm_exp = min_max_normalize([c.years_experience for c in candidates])
    norm_salary = min_max_normalize([c.salary_expectation for c in candidates])

    index = build_index(candidates)
    job_embedding = embed_text(job_description)
    model = _get_ml_model()
    use_llm = settings.use_llm

    # Import LLM helpers lazily so dense-only mode never needs the openai client.
    if use_llm:
        from app.llm.extract import extract_profile
        from app.llm.judge import judge_candidate

    scored = []
    for idx, cand in enumerate(candidates):
        features = np.array([
            norm_exp[idx],
            cand.skill_match_score,
            cand.interview_score,
            norm_salary[idx],
        ]).reshape(1, -1)
        structured_score = float(model.predict(features)[0])

        row = {
            "candidate_id": cand.candidate_id,
            "structured_score": structured_score,
            "semantic_score": None,
            "judge_score": None,
            "reasoning": None,
            "citations": None,
        }

        if use_llm:
            profile = extract_profile(cand.resume_text)
            evidence = retrieve_evidence(
                index, job_description, job_embedding,
                cand.candidate_id, settings.TOP_K,
            )
            verdict = judge_candidate(job_description, profile, evidence)
            judge_norm = verdict.score / 100.0
            row["judge_score"] = verdict.score
            row["reasoning"] = verdict.reasoning
            row["citations"] = verdict.citations
            row["final_score"] = (
                (1.0 - settings.JUDGE_WEIGHT) * structured_score
                + settings.JUDGE_WEIGHT * judge_norm
            )
        else:
            semantic_score = _dense_semantic_score(
                index, job_embedding, cand.candidate_id)
            row["semantic_score"] = semantic_score
            row["final_score"] = (
                settings.STRUCTURED_WEIGHT * structured_score
                + settings.SEMANTIC_WEIGHT * semantic_score
            )

        scored.append(row)

    # Deterministic: descending score, ties broken by candidate_id.
    scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    return [
        RankedCandidate(
            candidate_id=item["candidate_id"],
            rank=rank,
            final_score=item["final_score"],
            structured_score=item["structured_score"],
            semantic_score=item["semantic_score"],
            judge_score=item["judge_score"],
            reasoning=item["reasoning"],
            citations=item["citations"],
        )
        for rank, item in enumerate(scored, start=1)
    ]
