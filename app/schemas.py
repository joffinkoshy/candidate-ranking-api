from typing import List, Optional

from pydantic import BaseModel, Field


class Candidate(BaseModel):  # a single candidate for a job
    candidate_id: str
    years_experience: float = Field(ge=0)
    skill_match_score: float = Field(ge=0, le=1)
    interview_score: float = Field(ge=0, le=1)
    salary_expectation: float = Field(ge=0)
    resume_text: str


class RankCandidatesRequest(BaseModel):
    job_id: str
    job_description: str
    candidates: List[Candidate]


class RankedCandidate(BaseModel):
    candidate_id: str
    rank: int
    final_score: float
    # Explainability breakdown of the blended final score.
    structured_score: Optional[float] = None
    semantic_score: Optional[float] = None
    judge_score: Optional[float] = None       # 0..100 from the LLM judge
    reasoning: Optional[str] = None           # grounded justification
    citations: Optional[List[str]] = None     # exact phrases from the resume


class RankCandidatesResponse(BaseModel):
    ranked_candidates: List[RankedCandidate]


# --- Structured extraction (Phase 3): resume text -> validated profile ---
class ExtractedProfile(BaseModel):
    years_experience: float = Field(ge=0)
    skills: List[str] = Field(default_factory=list)
    education: str = ""
    seniority: str = "unknown"  # junior | mid | senior | unknown
