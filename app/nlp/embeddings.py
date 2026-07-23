"""Text embeddings via a local sentence-transformer (open-source, free).

The model is loaded lazily and cached, so importing this module is cheap and the
heavy dependency is only required when embeddings are actually computed.
"""

from functools import lru_cache

import numpy as np

from app.config import settings


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.EMBED_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings so cosine similarity == dot product."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
