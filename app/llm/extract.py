"""Phase 3 — structured extraction.

Turn messy resume text into a validated ExtractedProfile using a small, cheap
LLM. This is the "structured output / function-calling" concept: we constrain the
model to JSON and validate against a Pydantic schema, retrying with the error fed
back if validation fails (a guardrail against malformed model output).
"""

from __future__ import annotations

from app.config import settings
from app.llm.client import chat
from app.schemas import ExtractedProfile

_SYSTEM = (
    "You extract structured data from candidate resumes. "
    "Respond ONLY with a single JSON object, no prose."
)

_SCHEMA_HINT = (
    '{"years_experience": <number>, "skills": [<string>, ...], '
    '"education": <string>, "seniority": "junior" | "mid" | "senior"}'
)


def extract_profile(resume_text: str, max_retries: int = 2) -> ExtractedProfile:
    prompt = (
        f"Resume:\n{resume_text}\n\n"
        f"Return a JSON object matching this schema:\n{_SCHEMA_HINT}"
    )

    for _ in range(max_retries + 1):
        raw = chat(
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": prompt}],
            model=settings.EXTRACT_MODEL,
            json_mode=True,
        )
        try:
            return ExtractedProfile.model_validate_json(raw)
        except Exception as err:  # feed the error back and retry
            prompt = (
                f"{prompt}\n\nYour previous response was invalid: {err}. "
                "Return ONLY a valid JSON object matching the schema."
            )

    # Graceful fallback so one bad extraction never breaks the whole ranking.
    return ExtractedProfile(years_experience=0)
