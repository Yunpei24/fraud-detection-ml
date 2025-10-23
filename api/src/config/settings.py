"""
Application settings using Pydantic BaseSettings.
Reads from environment variables with validation.
"""
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=()
    )
    
    # Azure Configuration
    azure_storage_connection_string: str = Field(
        default="",
        description="Azure Storage connection string"
    )
    azure_storage_container_name: str = Field(
        default="models",
        description="Azure Storage container for models"
    )
    azure_key_vault_url: Optional[str] = Field(
        default=None,
        description="Azure Key Vault URL"
    )
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://localhost:5432/fraud_db",
        description="PostgreSQL connection URL"
    )
    
    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    redis_ssl: bool = Field(default=False, description="Use SSL for Redis")
    redis_ttl: int = Field(default=3600, description="Redis cache TTL in seconds")
    
    # ML Model Configuration
    model_path: str = Field(default="/app/models", description="Path to model files")
    model_version: str = Field(default="v1.0.0", description="Current model version")
    xgboost_model_name: str = Field(default="xgboost_fraud_model.pkl")
    random_forest_model_name: str = Field(default="random_forest_fraud_model.pkl")
    nn_model_name: str = Field(default="nn_fraud_model.pth")
    isolation_forest_model_name: str = Field(default="isolation_forest_model.pkl")
    
    # SHAP Explainers (one per model)
    shap_explainer_xgb_name: str = Field(default="shap_explainer_xgb.pkl")
    shap_explainer_rf_name: str = Field(default="shap_explainer_rf.pkl")
    shap_explainer_nn_name: str = Field(default="shap_explainer_nn.pkl")
    shap_explainer_iforest_name: str = Field(default="shap_explainer_iforest.pkl")
    
    # API Configuration
    api_version: str = Field(default="1.0.0", description="API version")
    api_title: str = Field(default="Fraud Detection API", description="API title")
    api_description: str = Field(
        default="Real-time fraud detection API",
        description="API description"
    )
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, description="Number of Uvicorn workers")
    reload: bool = Field(default=False, description="Enable auto-reload")
    
    # Security Configuration
    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for JWT"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration"
    )
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    
    # Prediction Configuration
    fraud_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraud classification threshold"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for alerts"
    )
    max_request_size: int = Field(
        default=1048576,
        description="Max request size in bytes"
    )
    timeout_seconds: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    
    # Monitoring & Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")
    enable_audit_log: bool = Field(default=True, description="Enable audit logging")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(
        default=100,
        description="Max requests per period"
    )
    rate_limit_period: int = Field(
        default=60,
        description="Rate limit period in seconds"
    )
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS"
    )
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods"
    )
    cors_allow_headers: List[str] = Field(
        default=["*"],
        description="Allowed headers"
    )
    
    # Feature Flags
    enable_batch_prediction: bool = Field(
        default=True,
        description="Enable batch prediction endpoint"
    )
    enable_shap_explanation: bool = Field(
        default=True,
        description="Enable SHAP explanations"
    )
    enable_model_reload: bool = Field(
        default=True,
        description="Enable model hot-reload"
    )
    enable_cache: bool = Field(default=True, description="Enable Redis cache")
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("cors_allow_methods", mode="before")
    @classmethod
    def parse_cors_methods(cls, v):
        """Parse CORS methods from comma-separated string."""
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v
    
    @field_validator("cors_allow_headers", mode="before")
    @classmethod
    def parse_cors_headers(cls, v):
        """Parse CORS headers from comma-separated string."""
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()


# Global settings instance
settings = Settings()
