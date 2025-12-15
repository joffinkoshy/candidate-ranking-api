from app.services.ranking import rank_candidates_logic
from app.schemas import Candidate

def test_candidates_are_ranked_by_score():
    candidates = [
        Candidate(
            candidate_id="A",
            years_experience=2,
            skill_match_score=0.6,
            interview_score=0.6,
            salary_expectation=10
        ),
        Candidate(
            candidate_id="B",
            years_experience=5,
            skill_match_score=0.9,
            interview_score=0.8,
            salary_expectation=12
        ),
    ]

    ranked = rank_candidates_logic(candidates)

    assert ranked[0].candidate_id == "B"
    assert ranked[1].candidate_id == "A"