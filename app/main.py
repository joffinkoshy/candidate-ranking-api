from fastapi import FastAPI
from app.schemas import RankCandidatesRequest,RankCandidatesResponse,RankedCandidate
app = FastAPI(title="Candidate Ranking API")

#HEALTH CHECK
@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/rank_candidates",response_model=RankCandidatesResponse) # FastAPI checking
def rank_candidates(request: RankCandidatesRequest):
    scored_candidates=[]

    for candidate in request.candidates:
        score = (
                0.3 * candidate.years_experience
                + 0.4 * candidate.skill_match_score
                + 0.3 * candidate.interview_score
                - 0.2 * candidate.salary_expectation
        )

        scored_candidates.append(
            {
                "candidate_id": candidate.candidate_id,
                "final_score": score,
            }
        )

    scored_candidates.sort(key=lambda x:x['final_score'],reverse=True)

    ranked_candidates = []

    for idx, candidate in enumerate(scored_candidates):
        ranked_candidates.append(
            RankedCandidate(
                candidate_id=candidate["candidate_id"],
                rank=idx + 1,
                final_score=candidate["final_score"]
            )
        )

    return RankCandidatesResponse(ranked_candidates)





