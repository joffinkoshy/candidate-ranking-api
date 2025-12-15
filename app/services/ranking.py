from typing import List
from app.schemas import Candidate, RankedCandidate

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

    WEIGHTS = {
        "experience": 0.4,
        "skill": 0.4,
        "interview": 0.3,
        "salary": 0.1
    }

    for idx,candidate in enumerate(candidates):
        score = (
                WEIGHTS["experience"] * norm_exp[idx]
                + WEIGHTS["skill"] * candidate.skill_match_score
                + WEIGHTS["interview"] * candidate.interview_score
                - WEIGHTS["salary"] * norm_salary[idx]
        )

        scored.append({
            "candidate_id": candidate.candidate_id,
            "final_score": score
        })

    scored.sort(
        key=lambda x: (-x["final_score"], -x["years_experience"])
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