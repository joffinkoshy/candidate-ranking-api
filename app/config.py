"""Central configuration.

All settings are overridable via environment variables (12-factor style), so no
business logic needs editing to change models, weights, or paths. A local `.env`
file (gitignored) is loaded automatically if python-dotenv is installed.
"""

import os

try:  # best-effort .env loading; app still works if python-dotenv is absent
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except Exception:
    pass

# Absolute paths anchored to this file, so the app runs from any working dir.
APP_DIR = os.path.dirname(os.path.abspath(__file__))          # .../app
ML_MODELS_DIR = os.path.join(APP_DIR, "ml", "models")


def _flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # --- Embeddings (local, open-source, free) ---
    EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # --- Chunking (character budget per passage) ---
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

    # --- Retrieval ---
    TOP_K = int(os.getenv("TOP_K", "4"))

    # --- Reranking (cross-encoder, second-stage over the RRF-fused pool) ---
    ENABLE_RERANK = _flag("ENABLE_RERANK", True)
    RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # How much bigger the fused candidate pool is than the final k, so the
    # reranker has real choices to make instead of just reordering k items.
    RERANK_POOL_MULTIPLIER = int(os.getenv("RERANK_POOL_MULTIPLIER", "3"))

    # --- Structured ML scorer ---
    ML_MODEL_TYPE = os.getenv("ML_MODEL_TYPE", "linear")  # "linear" | "gboost"

    # --- Final score blend ---
    # Dense-only mode (no LLM key): structured vs. semantic retrieval.
    STRUCTURED_WEIGHT = float(os.getenv("STRUCTURED_WEIGHT", "0.5"))
    SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.5"))
    # LLM mode: how much the LLM judge counts vs. the structured ML score.
    JUDGE_WEIGHT = float(os.getenv("JUDGE_WEIGHT", "0.6"))

    # --- LLM (Groq, OpenAI-compatible API) ---
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    LLM_API_KEY = os.getenv("GROQ_API_KEY", "")
    EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "llama-3.1-8b-instant")     # cheap tier
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")     # strong tier
    # Force-disable LLM even if a key is present (e.g. offline dev): ENABLE_LLM=false
    _ENABLE_LLM = _flag("ENABLE_LLM", True)

    @property
    def use_llm(self) -> bool:
        """Use LLM extraction + judge only when enabled and a key is present."""
        return self._ENABLE_LLM and bool(self.LLM_API_KEY)

    def ml_model_path(self) -> str:
        fname = "linear.pkl" if self.ML_MODEL_TYPE == "linear" else "gboost.pkl"
        return os.path.join(ML_MODELS_DIR, fname)


settings = Settings()
