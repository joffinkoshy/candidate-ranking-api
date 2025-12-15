from app.services.ranking import rank_candidates_logic
from app.schemas import Candidate

def test_ranking_orders_candidates_correctly():
    candidates = [
        Candidate(
            candidate_id="A",
            years_experience=1,
            skill_match_score=0.2,
            interview_score=0.3,
            salary_expectation=10,
            resume_text="bad resume"
        ),
        Candidate(
            candidate_id="B",
            years_experience=5,
            skill_match_score=0.9,
            interview_score=0.8,
            salary_expectation=5,
            resume_text="junior engineer with basic python"
        )
    ]

    ranked = rank_candidates_logic(
        candidates=candidates,
        job_description="Machine learning engineer"
    )

    assert ranked[0].candidate_id == "B"
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2