"""
Configuration settings for ChronoTrail API
"""
import os
from typing import Dict, List, Optional, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ChronoTrail API"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/chronotrail")
    
    # AWS Configuration
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    # S3 Configuration
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "chronotrail-media")
    
    # OpenSearch Configuration
    OPENSEARCH_ENDPOINT: str = os.getenv("OPENSEARCH_ENDPOINT", "")
    OPENSEARCH_USERNAME: Optional[str] = os.getenv("OPENSEARCH_USERNAME")
    OPENSEARCH_PASSWORD: Optional[str] = os.getenv("OPENSEARCH_PASSWORD")
    OPENSEARCH_USE_IAM_AUTH: bool = os.getenv("OPENSEARCH_USE_IAM_AUTH", "").lower() == "true"
    
    # Cognito Configuration
    COGNITO_USER_POOL_ID: str = os.getenv("COGNITO_USER_POOL_ID", "")
    COGNITO_CLIENT_ID: str = os.getenv("COGNITO_CLIENT_ID", "")
    COGNITO_REGION: str = os.getenv("COGNITO_REGION", "us-east-1")
    
    # Application Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # File Upload Limits (in MB)
    MAX_FILE_SIZE_FREE: int = 5
    MAX_FILE_SIZE_PREMIUM: int = 25
    MAX_FILE_SIZE_PRO: int = 100
    
    # Usage Limits
    TIER_LIMITS: Dict[str, Dict[str, Any]] = {
        "free": {
            "daily_content_limit": 3,
            "daily_query_limit": 10,
            "query_history_months": 3,
            "max_file_size_mb": 5,
            "max_storage_mb": 100
        },
        "premium": {
            "daily_content_limit": 10,
            "daily_query_limit": 50,
            "query_history_months": 24,
            "max_file_size_mb": 25,
            "max_storage_mb": 1024
        },
        "pro": {
            "daily_content_limit": -1,  # Unlimited
            "daily_query_limit": -1,    # Unlimited
            "query_history_months": -1, # Unlimited
            "max_file_size_mb": 100,
            "max_storage_mb": 10240
        }
    }
    
    model_config = {"case_sensitive": True, "env_file": ".env"}


settings = Settings()