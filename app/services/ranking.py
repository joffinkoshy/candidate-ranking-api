from typing import List
from app.schemas import Candidate, RankedCandidate

def rank_candidates_logic(candidates: List[Candidate]) -> List[RankedCandidate]:
    if not candidates:
        return []

    scored = []

    for candidate in candidates:
        score = (
            0.3 * candidate.years_experience
            + 0.4 * candidate.skill_match_score
            + 0.3 * candidate.interview_score
            - 0.2 * candidate.salary_expectation
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