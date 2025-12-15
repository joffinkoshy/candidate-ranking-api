from pydantic import BaseModel
from typing import List

class Candidate(BaseModel): # for one candidate
    candidate_id: str
    years_experience: int
    skill_match_score:float
    interview_score:float
    salary_expectation:float

class RankCandidatesRequest(BaseModel): # list of candidates each list applying for unique job_id
    job_id: str
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

class ResumeJobMatchRequest(BaseModel):
    job:JobText
    resumes:List[ResumeText]

class ResumeMatchScore(BaseModel):
    candidate_id: str
    similarity_score: float

class ResumeJobMatchResponse(BaseModel):
    matches: List[ResumeMatchScore]



