"""
Test suite for the improved candidate ranking API.
Tests configuration, error handling, and logging functionality.
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import logging
from app.main import app
from app.config import config
from app.services.ranking import rank_candidates_logic, MODEL_TYPE, MODEL_PATH
from app.schemas import Candidate

# Create test client
client = TestClient(app)


class TestConfiguration:
    """Test configuration functionality."""
    
    def test_config_loading(self):
        """Test that configuration loads correctly."""
        assert config is not None
        assert config.ENVIRONMENT == "development"
        assert config.MODEL_TYPE in ["linear", "gboost"]
        assert config.DEBUG is True
    
    def test_model_path_generation(self):
        """Test model path generation."""
        path = config.get_model_path()
        assert path.exists() or path.name in ["linear.pkl", "gboost.pkl"]
    
    def test_environment_variables(self):
        """Test environment variable override."""
        # Test with environment variable override
        with patch.dict(os.environ, {"MODEL_TYPE": "gboost"}):
            # Import config module fresh
            import importlib
            import app.config
            importlib.reload(app.config)
            from app.config import Config
            test_config = Config()
            assert test_config.MODEL_TYPE == "gboost"


class TestErrorHandling:
    """Test improved error handling."""
    
    def test_empty_candidate_list(self):
        """Test error handling for empty candidate list."""
        payload = {
            "job_id": "test_job",
            "job_description": "Test job",
            "candidates": []
        }
        
        response = client.post("/rank_candidates", json=payload)
        assert response.status_code == 400
        assert "Candidate list cannot be empty" in response.text
    
    def test_empty_job_description(self):
        """Test error handling for empty job description."""
        payload = {
            "job_id": "test_job",
            "job_description": "",
            "candidates": [
                {
                    "candidate_id": "test_candidate",
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
    
    def test_invalid_skill_score(self):
        """Test error handling for invalid skill score."""
        candidates = [
            Candidate(
                candidate_id="test",
                years_experience=5,
                skill_match_score=150,  # Invalid - should be <= 100
                interview_score=70,
                salary_expectation=100000,
                resume_text="Test resume"
            )
        ]
        
        with pytest.raises(ValueError) as exc_info:
            rank_candidates_logic(candidates, "Test job")
        
        assert "Skill match score must be between 0 and 100" in str(exc_info.value)
    
    def test_missing_candidate_id(self):
        """Test error handling for missing candidate ID."""
        candidates = [
            Candidate(
                candidate_id="",  # Invalid
                years_experience=5,
                skill_match_score=0.8,
                interview_score=0.7,
                salary_expectation=100000,
                resume_text="Test resume"
            )
        ]
        
        with pytest.raises(ValueError) as exc_info:
            rank_candidates_logic(candidates, "Test job")
        
        assert "missing valid ID" in str(exc_info.value)


class TestLogging:
    """Test logging functionality."""
    
    @patch('app.main.logger')
    def test_api_logging(self, mock_logger):
        """Test that API endpoints log properly."""
        # Test health endpoint logging
        response = client.get("/health")
        assert response.status_code == 200
        mock_logger.info.assert_called_with("Detailed health check endpoint called")
    
    @patch('app.services.ranking.logger')
    def test_ranking_logging(self, mock_logger):
        """Test that ranking service logs properly."""
        candidates = [
            Candidate(
                candidate_id="test",
                years_experience=5,
                skill_match_score=0.8,
                interview_score=0.7,
                salary_expectation=100000,
                resume_text="Test resume"
            )
        ]
        
        # This should log various messages
        result = rank_candidates_logic(candidates, "Test job description")
        
        # Check that logging was called
        mock_logger.info.assert_any_call("Starting ranking process for 1 candidates")
        mock_logger.info.assert_any_call("Successfully ranked 1 candidates")


class TestConfigurationOverride:
    """Test configuration override via environment variables."""
    
    @patch.dict(os.environ, {"MODEL_TYPE": "gboost", "ENVIRONMENT": "test"})
    def test_model_type_override(self):
        """Test that MODEL_TYPE can be overridden."""
        # Reload the module to pick up new env vars
        import importlib
        import app.config
        importlib.reload(app.config)
        
        from app.config import config as new_config
        assert new_config.MODEL_TYPE == "gboost"
        assert new_config.ENVIRONMENT == "test"


class TestHealthEndpoint:
    """Test the improved health endpoint."""
    
    def test_health_endpoint_response(self):
        """Test health endpoint returns expected data."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "model_type" in data
        assert "environment" in data
        assert "api_version" in data
        assert "debug_mode" in data


class TestCORSMiddleware:
    """Test CORS middleware configuration."""
    
    def test_cors_middleware_configured(self):
        """Test that CORS middleware is properly configured in the app."""
        # Check that CORS middleware is present in the app
        middleware_found = False
        for middleware in app.user_middleware:
            if hasattr(middleware, 'cls') and 'CORSMiddleware' in str(middleware.cls):
                middleware_found = True
                break
        
        assert middleware_found is True, "CORS middleware should be configured"
        
        # Test that the app doesn't block requests (basic functionality test)
        response = client.get("/health")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])