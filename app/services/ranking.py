from typing import List
import joblib
import numpy as np
import logging
from pathlib import Path

from app.schemas import Candidate, RankedCandidate
from app.nlp.similarity import compute_similarity
from app.config import config

# Configure logging
logger = logging.getLogger(__name__)

# -------------------------------
# Configuration from config module
# -------------------------------
MODEL_TYPE = config.MODEL_TYPE
MODEL_PATH = config.get_model_path()

# Load model with error handling
try:
    MODEL = joblib.load(MODEL_PATH)
    logger.info(f"Loaded {MODEL_TYPE} model from {MODEL_PATH}")
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and available.")
except Exception as e:
    raise Exception(f"Failed to load model: {str(e)}")


# -------------------------------
# Utility: Min–Max Normalization
# -------------------------------
def min_max_normalize(values: List[float]) -> List[float]:
    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return [0.5 for _ in values]

    return [(v - min_val) / (max_val - min_val) for v in values]


# -------------------------------
# Core Ranking Logic
# -------------------------------
def rank_candidates_logic(
    candidates: List[Candidate],
    job_description: str
) -> List[RankedCandidate]:
    """
    Rank candidates using structured ML score + semantic similarity.
    
    This function accepts raw feature values and normalizes them internally:
    - years_experience: Raw years (e.g., 5, 10, 15) → normalized 0-1 based on candidate pool
    - skill_match_score: Raw score 0-100 → normalized 0-1
    - interview_score: Raw score 0-100 → normalized 0-1
    - salary_expectation: Raw salary → normalized 0-1 based on candidate pool (treated as negative signal)
    
    Args:
        candidates: List of Candidate objects with raw feature values
        job_description: Job description text for semantic matching
        
    Returns:
        List of RankedCandidate objects with ranks and final scores
        
    Raises:
        ValueError: If input validation fails (e.g., scores outside 0-100 range)
        Exception: For processing errors
    """
    logger.info(f"Starting ranking process for {len(candidates)} candidates")

    # Input validation
    if not candidates:
        logger.warning("Empty candidate list provided")
        return []
    
    if not job_description or not job_description.strip():
        logger.error("Job description is empty or invalid")
        raise ValueError("Job description cannot be empty")
    
    # Validate candidate data
    for candidate in candidates:
        if not candidate.candidate_id or not candidate.candidate_id.strip():
            raise ValueError(f"Candidate missing valid ID: {candidate}")
        
        if not candidate.resume_text or not candidate.resume_text.strip():
            logger.warning(f"Candidate {candidate.candidate_id} has empty resume text")
        
        # Validate numeric ranges for raw values
        if not (0 <= candidate.skill_match_score <= 100):
            raise ValueError(f"Skill match score must be between 0 and 100, got {candidate.skill_match_score}")
        
        if not (0 <= candidate.interview_score <= 100):
            raise ValueError(f"Interview score must be between 0 and 100, got {candidate.interview_score}")
        
        if candidate.years_experience < 0:
            raise ValueError(f"Years of experience cannot be negative, got {candidate.years_experience}")
        
        if candidate.salary_expectation < 0:
            raise ValueError(f"Salary expectation cannot be negative, got {candidate.salary_expectation}")

    # Normalize all features
    try:
        # Normalize years of experience (0-1 scale based on candidate pool)
        exp_values = [c.years_experience for c in candidates]
        norm_exp = min_max_normalize(exp_values)
        
        # Normalize skill match score (0-100 → 0-1)
        norm_skill = [c.skill_match_score / 100.0 for c in candidates]
        
        # Normalize interview score (0-100 → 0-1)
        norm_interview = [c.interview_score / 100.0 for c in candidates]
        
        # Normalize salary expectation (0-1 scale based on candidate pool, but treat as negative)
        salary_values = [c.salary_expectation for c in candidates]
        norm_salary = min_max_normalize(salary_values)
        
    except Exception as e:
        logger.error(f"Normalization failed: {str(e)}")
        raise Exception("Failed to normalize candidate features")

    scored_candidates = []

    for idx, candidate in enumerate(candidates):
        try:
            # -------------------------------
            # Structured ML features
            # -------------------------------
            features = np.array([
                norm_exp[idx],
                norm_skill[idx],
                norm_interview[idx],
                norm_salary[idx]
            ]).reshape(1, -1)

            structured_score = float(MODEL.predict(features)[0])
            logger.debug(f"Candidate {candidate.candidate_id}: structured score = {structured_score:.3f}")

            # -------------------------------
            # Semantic similarity (NLP)
            # -------------------------------
            semantic_score = compute_similarity(
                job_text=job_description,
                resume_text=candidate.resume_text
            )
            logger.debug(f"Candidate {candidate.candidate_id}: semantic score = {semantic_score:.3f}")

            # -------------------------------
            # Final combined score
            # -------------------------------
            FINAL_SCORE = (
                0.6 * structured_score +
                0.4 * semantic_score
            )
            
            logger.info(f"Candidate {candidate.candidate_id}: final score = {FINAL_SCORE:.3f}")

            scored_candidates.append({
                "candidate_id": candidate.candidate_id,
                "final_score": FINAL_SCORE
            })
            
        except Exception as e:
            logger.error(f"Failed to process candidate {candidate.candidate_id}: {str(e)}")
            raise Exception(f"Failed to process candidate {candidate.candidate_id}: {str(e)}")

    # -------------------------------
    # Sorting & ranking
    # -------------------------------
    try:
        scored_candidates.sort(
            key=lambda x: (-x["final_score"], x["candidate_id"])
        )

        ranked = [
            RankedCandidate(
                candidate_id=item["candidate_id"],
                rank=rank,
                final_score=item["final_score"]
            )
            for rank, item in enumerate(scored_candidates, start=1)
        ]
        
        logger.info(f"Successfully ranked {len(ranked)} candidates")
        return ranked
        
    except Exception as e:
        logger.error(f"Ranking failed during sorting: {str(e)}")
        raise Exception("Failed to complete ranking process")