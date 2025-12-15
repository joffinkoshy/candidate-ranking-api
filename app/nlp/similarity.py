from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def calibrate_similarity(similarity: float,
                         min_threshold: float = 0.55,
                         max_threshold: float = 0.90) -> float:
    """
    Calibrate raw similarity into a bounded, meaningful signal.
    """

    if similarity < min_threshold:
        return 0.0

    if similarity >= max_threshold:
        return 1.0

    return (similarity - min_threshold) / (max_threshold - min_threshold)

def normalize_score(score: float) -> float:
    """
    Convert cosine similarity (-1 to 1) → (0 to 1)
    """
    return (score + 1) / 2

def compute_similarity(job_text: str, resume_text: str) -> float:
    """
    Compute semantic similarity between job description and resume.
    Returns a value between 0 and 1.
    """

    if not job_text.strip() or not resume_text.strip():
        raise ValueError("Input texts must be non-empty")

    embeddings = model.encode(
        [job_text, resume_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    job_embedding = embeddings[0].reshape(1, -1)
    resume_embedding = embeddings[1].reshape(1, -1)

    similarity = cosine_similarity(job_embedding, resume_embedding)[0][0]

    raw = normalize_score(similarity)
    return calibrate_similarity(raw)

