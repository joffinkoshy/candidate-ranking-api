from fastapi import FastAPI

from app.config import settings
from app.schemas import RankCandidatesRequest, RankCandidatesResponse
from app.services.ranking import rank_candidates_logic

app = FastAPI(title="Candidate Ranking API")


@app.get("/")
def root():
    return {"status": "API is running"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "ml_model_type": settings.ML_MODEL_TYPE,
        "embed_model": settings.EMBED_MODEL,
        "blend": {
            "structured": settings.STRUCTURED_WEIGHT,
            "semantic": settings.SEMANTIC_WEIGHT,
        },
    }


@app.post("/rank_candidates", response_model=RankCandidatesResponse)
def rank_candidates(request: RankCandidatesRequest):
    ranked = rank_candidates_logic(request.candidates, request.job_description)
    return RankCandidatesResponse(ranked_candidates=ranked)
