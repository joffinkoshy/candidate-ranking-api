"""
Integration test suite for the upgraded candidate ranking API.
Tests the complete workflow with all improvements.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.config import config

client = TestClient(app)


class TestCompleteWorkflow:
    """Test the complete API workflow with all improvements."""
    
    def test_complete_ranking_workflow(self):
        """Test a complete ranking request with all features."""
        payload = {
            "job_id": "senior_ml_engineer",
            "job_description": "Senior Machine Learning Engineer with 5+ years experience in Python, TensorFlow, and production ML systems",
            "candidates": [
                {
                    "candidate_id": "candidate_1",
                    "years_experience": 7,
                    "skill_match_score": 95,  # Raw score 0-100
                    "interview_score": 90,   # Raw score 0-100
                    "salary_expectation": 150000,
                    "resume_text": "Senior ML Engineer with 7 years experience in Python, TensorFlow, and production ML systems. Built scalable ML pipelines."
                },
                {
                    "candidate_id": "candidate_2",
                    "years_experience": 3,
                    "skill_match_score": 70,  # Raw score 0-100
                    "interview_score": 60,   # Raw score 0-100
                    "salary_expectation": 90000,
                    "resume_text": "Junior ML Engineer with 3 years experience in Python and scikit-learn."
                },
                {
                    "candidate_id": "candidate_3",
                    "years_experience": 5,
                    "skill_match_score": 85,  # Raw score 0-100
                    "interview_score": 80,   # Raw score 0-100
                    "salary_expectation": 120000,
                    "resume_text": "ML Engineer with 5 years experience in Python and TensorFlow. Some production experience."
                }
            ]
        }
        
        # Test successful ranking
        response = client.post("/rank_candidates", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "ranked_candidates" in data
        assert len(data["ranked_candidates"]) == 3
        
        # Verify ranking makes sense (higher experience and scores should rank higher)
        ranked_candidates = data["ranked_candidates"]
        
        # Candidate 1 should be ranked highest
        assert ranked_candidates[0]["candidate_id"] == "candidate_1"
        assert ranked_candidates[0]["rank"] == 1
        assert ranked_candidates[0]["final_score"] > 0.8
        
        # Candidate 3 should be ranked second
        assert ranked_candidates[1]["candidate_id"] == "candidate_3"
        assert ranked_candidates[1]["rank"] == 2
        
        # Candidate 2 should be ranked last
        assert ranked_candidates[2]["candidate_id"] == "candidate_2"
        assert ranked_candidates[2]["rank"] == 3
        
        # Verify scores are in descending order
        scores = [c["final_score"] for c in ranked_candidates]
        assert scores == sorted(scores, reverse=True)
    
    def test_error_workflow(self):
        """Test complete error handling workflow."""
        # Test empty candidate list
        payload = {
            "job_id": "test_job",
            "job_description": "Test job",
            "candidates": []
        }
        
        response = client.post("/rank_candidates", json=payload)
        assert response.status_code == 400
        assert "Candidate list cannot be empty" in response.text
        
        # Test empty job description
        payload = {
            "job_id": "test_job",
            "job_description": "",
            "candidates": [
                {
                    "candidate_id": "test",
                    "years_experience": 5,
                    "skill_match_score": 0.8,
                    "interview_score": 0.7,
                    "salary_expectation": 100000,
                    "resume_text": "Test resume"
                }
            ]
        }
        
        response = client.post("/rank_candidates", json=payload)
        assert response.status_code == 400
        assert "Job description cannot be empty" in response.text
    
    def test_health_endpoint_comprehensive(self):
        """Test health endpoint returns all expected information."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        expected_keys = ["status", "model_type", "environment", "api_version", "debug_mode"]
        
        for key in expected_keys:
            assert key in data, f"Health endpoint missing expected key: {key}"
        
        assert data["status"] == "ok"
        assert data["model_type"] in ["linear", "gboost"]
        assert data["environment"] == config.ENVIRONMENT
        assert data["api_version"] == config.API_VERSION
        assert isinstance(data["debug_mode"], bool)


class TestConfigurationIntegration:
    """Test configuration integration with the full application."""
    
    def test_config_affects_model_loading(self):
        """Test that configuration affects which model is loaded."""
        # The current model type should match the config
        from app.services.ranking import MODEL_TYPE
        assert MODEL_TYPE == config.MODEL_TYPE
        
        # Test that we can see the model type in the health endpoint
        response = client.get("/health")
        data = response.json()
        assert data["model_type"] == config.MODEL_TYPE
    
    def test_environment_affects_debug_mode(self):
        """Test that environment affects debug mode."""
        response = client.get("/health")
        data = response.json()
        
        assert data["debug_mode"] == config.DEBUG
        assert data["environment"] == config.ENVIRONMENT


class TestLoggingIntegration:
    """Test logging integration in the complete workflow."""
    
    @patch('app.main.logger')
    @patch('app.services.ranking.logger')
    def test_complete_logging_workflow(self, mock_ranking_logger, mock_api_logger):
        """Test that logging works throughout the complete workflow."""
        
        payload = {
            "job_id": "test_job",
            "job_description": "Test job description",
            "candidates": [
                {
                    "candidate_id": "test_candidate",
                    "years_experience": 5,
                    "skill_match_score": 0.8,
                    "interview_score": 0.7,
                    "salary_expectation": 100000,
                    "resume_text": "Test resume text"
                }
            ]
        }
        
        # Make the request
        response = client.post("/rank_candidates", json=payload)
        
        # Verify API logging
        mock_api_logger.info.assert_any_call("Received ranking request for job test_job with 1 candidates")
        mock_api_logger.info.assert_any_call("Successfully ranked 1 candidates")
        
        # Verify ranking service logging
        mock_ranking_logger.info.assert_any_call("Starting ranking process for 1 candidates")
        mock_ranking_logger.info.assert_any_call("Successfully ranked 1 candidates")
        
        # Check that candidate score was logged (contains the candidate ID and score pattern)
        candidate_logs = [
            call for call in mock_ranking_logger.info.call_args_list
            if "Candidate test_candidate: final score = " in str(call)
        ]
        assert len(candidate_logs) > 0, "Candidate score should be logged"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_candidate(self):
        """Test ranking with only one candidate."""
        payload = {
            "job_id": "test_job",
            "job_description": "Test job",
            "candidates": [
                {
                    "candidate_id": "only_candidate",
                    "years_experience": 5,
                    "skill_match_score": 80,  # Raw score 0-100
                    "interview_score": 70,   # Raw score 0-100
                    "salary_expectation": 100000,
                    "resume_text": "Test resume"
                }
            ]
        }
        
        response = client.post("/rank_candidates", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["ranked_candidates"]) == 1
        assert data["ranked_candidates"][0]["rank"] == 1
    
    def test_many_candidates(self):
        """Test ranking with many candidates."""
        candidates = []
        for i in range(10):
            candidates.append({
                "candidate_id": f"candidate_{i}",
                "years_experience": 5 + i,
                "skill_match_score": 70 + (i * 2),  # Raw score 0-100
                "interview_score": 60 + (i * 3),   # Raw score 0-100
                "salary_expectation": 80000 + (i * 5000),
                "resume_text": f"Test resume for candidate {i}"
            })
        
        payload = {
            "job_id": "test_job",
            "job_description": "Test job",
            "candidates": candidates
        }
        
        response = client.post("/rank_candidates", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["ranked_candidates"]) == 10
        
        # Verify all ranks are unique and sequential
        ranks = [c["rank"] for c in data["ranked_candidates"]]
        assert sorted(ranks) == list(range(1, 11))
        
        # Verify scores are in descending order
        scores = [c["final_score"] for c in data["ranked_candidates"]]
        assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])