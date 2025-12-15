from typing import List
from app.schemas import Candidate, RankedCandidate
import joblib
import numpy as np

MODEL_TYPE = "linear"  # options: "linear", "gboost"
MODEL_PATH = "app/ml/models/linear.pkl"
MODEL = joblib.load(MODEL_PATH)

def min_max_normalize(values):
    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return [0.5 for _ in values]

    return [(v - min_val) / (max_val - min_val) for v in values]


def rank_candidates_logic(candidates: List[Candidate]) -> List[RankedCandidate]:
    if not candidates:
        return []

    scored = []
    exp_values = [c.years_experience for c in candidates]
    salary_values = [c.salary_expectation for c in candidates]

    norm_exp = min_max_normalize(exp_values)
    norm_salary = min_max_normalize(salary_values)

    for idx,candidate in enumerate(candidates):
        features = np.array([
            norm_exp[idx],
            candidate.skill_match_score,
            candidate.interview_score,
            norm_salary[idx]
        ]).reshape(1, -1)

        score = MODEL.predict(features)[0]

        scored.append({
            "candidate_id": candidate.candidate_id,
            "final_score": score
        })

    scored.sort(
        key=lambda x: (-x["final_score"], x["candidate_id"])
    )

    ranked = []

    for index, item in enumerate(scored, start=1):
        ranked.append(
            RankedCandidate(
                candidate_id=item["candidate_id"],
                rank=index,
                final_score=item["final_score"]
            )
        )

    return ranked