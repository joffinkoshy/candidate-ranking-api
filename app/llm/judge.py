"""Phase 4 — LLM-as-a-judge.

A stronger LLM scores each candidate against the job, using ONLY the retrieved
evidence passages, and must cite exact phrases from that evidence. Grounding the
judgment in retrieved text (rather than the model's own priors) is what makes the
output explainable and reduces hallucination.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from app.config import settings
from app.llm.client import chat
from app.schemas import ExtractedProfile


class JudgeResult(BaseModel):
    # Not bounded here on purpose: an out-of-range score is clamped below rather
    # than rejected by validation (which would waste a retry / hit the fallback).
    score: float
    reasoning: str = ""
    citations: List[str] = Field(default_factory=list)


_SYSTEM = (
    "You are a hiring evaluator. Judge how well a candidate fits a job using ONLY "
    "the evidence provided. Cite exact phrases copied from the evidence. "
    "Respond ONLY with a single JSON object, no prose."
)

_SCHEMA_HINT = (
    '{"score": <number 0-100>, "reasoning": <string>, '
    '"citations": [<exact phrase from evidence>, ...]}'
)


def judge_candidate(job_description: str, profile: ExtractedProfile,
                    evidence: List[str], max_retries: int = 2) -> JudgeResult:
    evidence_block = "\n".join(f"- {e}" for e in evidence) or "(no evidence retrieved)"
    prompt = (
        f"Job description:\n{job_description}\n\n"
        f"Extracted candidate profile:\n{profile.model_dump_json()}\n\n"
        f"Evidence passages from the resume:\n{evidence_block}\n\n"
        f"Score the candidate's fit 0-100. Return JSON:\n{_SCHEMA_HINT}\n"
        "Every citation MUST be an exact phrase from the evidence above."
    )

    for _ in range(max_retries + 1):
        raw = chat(
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": prompt}],
            model=settings.JUDGE_MODEL,
            json_mode=True,
        )
        try:
            result = JudgeResult.model_validate_json(raw)
            result.score = max(0.0, min(100.0, result.score))
            return result
        except Exception as err:
            prompt = (
                f"{prompt}\n\nYour previous response was invalid: {err}. "
                "Return ONLY a valid JSON object matching the schema."
            )

    return JudgeResult(
        score=0.0,
        reasoning="Judge failed to produce a valid response.",
        citations=[],
    )
