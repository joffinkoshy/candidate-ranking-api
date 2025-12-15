from typing import List
import joblib
import numpy as np

from app.schemas import Candidate, RankedCandidate
from app.nlp.similarity import compute_similarity


# -------------------------------
# Model selection
# -------------------------------
MODEL_TYPE = "linear"  # options: "linear", "gboost"

if MODEL_TYPE == "linear":
    MODEL_PATH = "app/ml/models/linear.pkl"
elif MODEL_TYPE == "gboost":
    MODEL_PATH = "app/ml/models/gboost.pkl"
else:
    raise ValueError("Invalid MODEL_TYPE")

MODEL = joblib.load(MODEL_PATH)


# -------------------------------
# Utility: Min–Max Normalization
# -------------------------------
def min_max_normalize(values: List[float]) -> List[float]:
    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return [0.5 for _ in values]

    return [(v - min_val) / (max_val - min_val) for v in values]


# -------------------------------
# Core Ranking Logic
# -------------------------------
def rank_candidates_logic(
    candidates: List[Candidate],
    job_description: str
) -> List[RankedCandidate]:
    """
    Rank candidates using structured ML score + semantic similarity.
    """

    if not candidates:
        return []

    # Collect values for normalization
    exp_values = [c.years_experience for c in candidates]
    salary_values = [c.salary_expectation for c in candidates]

    norm_exp = min_max_normalize(exp_values)
    norm_salary = min_max_normalize(salary_values)

    scored_candidates = []

    for idx, candidate in enumerate(candidates):

        # -------------------------------
        # Structured ML features
        # -------------------------------
        features = np.array([
            norm_exp[idx],
            candidate.skill_match_score,
            candidate.interview_score,
            norm_salary[idx]
        ]).reshape(1, -1)

        structured_score = float(MODEL.predict(features)[0])

        # -------------------------------
        # Semantic similarity (NLP)
        # -------------------------------
        semantic_score = compute_similarity(
            job_text=job_description,
            resume_text=candidate.resume_text
        )

        # -------------------------------
        # Final combined score
        # -------------------------------
        FINAL_SCORE = (
            0.6 * structured_score +
            0.4 * semantic_score
        )

        scored_candidates.append({
            "candidate_id": candidate.candidate_id,
            "final_score": FINAL_SCORE
        })

    # -------------------------------
    # Sorting & ranking
    # -------------------------------
    scored_candidates.sort(
        key=lambda x: (-x["final_score"], x["candidate_id"])
    )

    ranked = [
        RankedCandidate(
            candidate_id=item["candidate_id"],
            rank=rank,
            final_score=item["final_score"]
        )
        for rank, item in enumerate(scored_candidates, start=1)
    ]

    return ranked