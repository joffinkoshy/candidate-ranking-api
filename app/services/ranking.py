from typing import List
from app.schemas import Candidate, RankedCandidate

def rank_candidates_logic(candidates: List[Candidate]) -> List[RankedCandidate]:
    if not candidates:
        return []

    scored = []

    WEIGHTS = {
        "experience": 0.4,
        "skill": 0.4,
        "interview": 0.3,
        "salary": 0.1
    }

    for candidate in candidates:
        score = (
                WEIGHTS["experience"] * candidate.years_experience
                + WEIGHTS["skill"] * candidate.skill_match_score
                + WEIGHTS["interview"] * candidate.interview_score
                - WEIGHTS["salary"] * candidate.salary_expectation
        )

        scored.append({
            "candidate_id": candidate.candidate_id,
            "final_score": score
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)

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