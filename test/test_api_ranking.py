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
                "skill_match_score": 0.2,
                "interview_score": 0.3,
                "salary_expectation": 10,
                "resume_text": "sales experience"
            },
            {
                "candidate_id": "B",
                "years_experience": 5,
                "skill_match_score": 0.9,
                "interview_score": 0.8,
                "salary_expectation": 6,
                "resume_text": "machine learning projects in python"
            }
        ]
    }

    response = client.post("/rank_candidates", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "ranked_candidates" in data
    assert data["ranked_candidates"][0]["candidate_id"] == "B"