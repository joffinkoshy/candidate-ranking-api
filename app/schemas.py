from pydantic import BaseModel
from typing import List

class Candidate(BaseModel): # for one candidate
    candidate_id: str
    years_experience: int  # Raw years (e.g., 5, 10, 15)
    skill_match_score: float  # Raw score (0-100 scale)
    interview_score: float  # Raw score (0-100 scale)  
    salary_expectation: float  # Raw salary (e.g., 80000, 120000)
    resume_text: str

class RankCandidatesRequest(BaseModel): # list of candidates each list applying for unique job_id
    job_id: str
    job_description:str
    candidates: List[Candidate]

class RankedCandidate(BaseModel):
    candidate_id: str
    rank: int
    final_score: float

class RankCandidatesResponse(BaseModel):
    ranked_candidates: List[RankedCandidate]

class JobText(BaseModel):
    job_id:str
    description:str

class ResumeText(BaseModel):
    candidate_id:str
    resume_text:str


class ResumeMatchScore(BaseModel):
    candidate_id: str
    similarity_score: float

class ResumeJobMatchResponse(BaseModel):
    matches: List[ResumeMatchScore]



