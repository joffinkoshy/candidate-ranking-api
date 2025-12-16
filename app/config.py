"""
Configuration module for Candidate Ranking API.
Centralizes all application settings and environment variables.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration settings."""
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # API Settings
    API_TITLE: str = os.getenv("API_TITLE", "Candidate Ranking API")
    API_VERSION: str = os.getenv("API_VERSION", "1.0.0")
    API_DESCRIPTION: str = os.getenv("API_DESCRIPTION", "ML-powered candidate ranking service")
    
    # Model Configuration
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "linear").lower()
    MODEL_DIR: str = os.getenv("MODEL_DIR", "app/ml/models")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # NLP Configuration
    NLP_MODEL_NAME: str = os.getenv("NLP_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Rate Limiting
    RATE_LIMIT: int = int(os.getenv("RATE_LIMIT", "100"))
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() in ["development", "dev"]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() in ["production", "prod"]
    
    def get_model_path(self, model_type: Optional[str] = None) -> Path:
        """Get the full path to a model file."""
        model_type = model_type or self.MODEL_TYPE
        return Path(self.MODEL_DIR) / f"{model_type}.pkl"
    
    def validate(self) -> None:
        """Validate configuration settings."""
        valid_model_types = ["linear", "gboost"]
        if self.MODEL_TYPE not in valid_model_types:
            raise ValueError(f"Invalid MODEL_TYPE: {self.MODEL_TYPE}. Must be one of: {valid_model_types}")
        
        if not Path(self.MODEL_DIR).exists():
            raise FileNotFoundError(f"Model directory not found: {self.MODEL_DIR}")

# Initialize configuration
config = Config()

# Configure logging based on config
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT
)

# Validate configuration on import
try:
    config.validate()
    logging.getLogger(__name__).info("Configuration loaded and validated successfully")
except Exception as e:
    logging.getLogger(__name__).error(f"Configuration validation failed: {str(e)}")
    raise

__all__ = ["config", "Config"]