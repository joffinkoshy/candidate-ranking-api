from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_rank_candidates_endpoint():
    payload = {
        "job_id": "job_123",
        "job_description": "Machine learning engineer",
        "candidates": [
            {
                "candidate_id": "A",
                "years_experience": 1,
                "skill_match_score": 20,  # Raw score 0-100
                "interview_score": 30,   # Raw score 0-100
                "salary_expectation": 10000,
                "resume_text": "sales experience"
            },
            {
                "candidate_id": "B",
                "years_experience": 5,
                "skill_match_score": 90,  # Raw score 0-100
                "interview_score": 80,   # Raw score 0-100
                "salary_expectation": 60000,
                "resume_text": "machine learning projects in python"
            }
        ]
    }

    response = client.post("/rank_candidates", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "ranked_candidates" in data
    assert data["ranked_candidates"][0]["candidate_id"] == "B"