"""Provider-agnostic LLM client.

Targets Groq's OpenAI-compatible endpoint by default (free, hosted, open-source
models). Because the API is OpenAI-compatible, switching providers (OpenRouter,
etc.) is a base-URL change in config — no code change here.

Used from Phase 3 (extraction) and Phase 4 (LLM-as-judge). Requires GROQ_API_KEY.
"""

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _get_client():
    from openai import OpenAI

    if not settings.LLM_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and export GROQ_API_KEY=... before using LLM features."
        )
    return OpenAI(base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY)


def chat(messages: list[dict], model: str, temperature: float = 0.0,
         json_mode: bool = False) -> str:
    """Single-turn chat completion; returns the assistant message content."""
    client = _get_client()
    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content
