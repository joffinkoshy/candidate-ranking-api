from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import RankCandidatesRequest,RankCandidatesResponse,RankedCandidate
from app.services.ranking import rank_candidates_logic, MODEL_TYPE
from app.config import config
import logging

# Get logger from config
logger = logging.getLogger(__name__)

# Create FastAPI app with config
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
    debug=config.DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"Starting {config.API_TITLE} v{config.API_VERSION} in {config.ENVIRONMENT} mode")

#HEALTH CHECK
@app.get("/")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "API is running"}

@app.get("/health")
def health_check():
    logger.info("Detailed health check endpoint called")
    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "environment": config.ENVIRONMENT,
        "api_version": config.API_VERSION,
        "debug_mode": config.DEBUG
    }

@app.post("/rank_candidates", response_model=RankCandidatesResponse)
def rank_candidates(request: RankCandidatesRequest):
    """
    Rank candidates for a job opening using ML-powered scoring.
    
    Args:
        request: RankCandidatesRequest containing job details and candidate list
        
    Returns:
        RankCandidatesResponse with ranked candidates and scores
        
    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    logger.info(f"Received ranking request for job {request.job_id} with {len(request.candidates)} candidates")
    
    try:
        # Validate input
        if not request.candidates:
            logger.warning("Empty candidate list received")
            raise HTTPException(status_code=400, detail="Candidate list cannot be empty")
            
        if not request.job_description or not request.job_description.strip():
            logger.warning("Empty job description received")
            raise HTTPException(status_code=400, detail="Job description cannot be empty")
        
        # Process ranking
        ranked_candidates = rank_candidates_logic(request.candidates, request.job_description)
        logger.info(f"Successfully ranked {len(ranked_candidates)} candidates")
        
        return RankCandidatesResponse(ranked_candidates=ranked_candidates)
        
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
        
    except FileNotFoundError as fnf:
        logger.error(f"Model file error: {str(fnf)}")
        raise HTTPException(status_code=500, detail="Model loading failed. Please check server configuration.")
        
    except Exception as e:
        logger.error(f"Unexpected error during ranking: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing your request.")





