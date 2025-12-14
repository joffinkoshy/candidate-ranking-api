from fastapi import FastAPI
from app.schemas import RankCandidatesRequest,RankCandidatesResponse,RankedCandidate
from app.services.ranking import rank_candidates_logic
app = FastAPI(title="Candidate Ranking API")

#HEALTH CHECK
@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/rank_candidates", response_model=RankCandidatesResponse)
def rank_candidates(request: RankCandidatesRequest):
    ranked_candidates = rank_candidates_logic(request.candidates)
    return RankCandidatesResponse(ranked_candidates=ranked_candidates)





