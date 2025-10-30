"""
Centralized Configuration for Fraud Detection MLOps Project
================================================================

This module provides a unified configuration system for all modules in the
fraud detection project. It centralizes environment variables and provides
consistent settings across API, Data, Drift, Training, and Airflow modules.

Key Features:
- Single source of truth for environment variables
- Hierarchical configuration with module-specific overrides
- Pydantic validation for type safety
- Environment-specific configurations
- Shared common settings (database, Azure, monitoring, etc.)

Usage:
    from config.settings import get_settings

    # Get global settings
    settings = get_settings()

    # Access common config
    db_url = settings.database.url

    # Access module-specific config
    api_port = settings.api.port
    drift_threshold = settings.drift.data_drift_threshold
"""

from typing import List, Optional, Any
from pathlib import Path
from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str):
    """Environment enumeration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class DatabaseSettings(BaseSettings):
    """Database configuration shared across all modules"""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="fraud_db", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="postgres", description="Database password")
    pool_size: int = Field(default=20, description="Connection pool size")
    max_overflow: int = Field(default=40, description="Max overflow connections")
    ssl_mode: str = Field(default="prefer", description="SSL mode")

    @computed_field
    @property
    def url(self) -> str:
        """Construct database URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @computed_field
    @property
    def sqlalchemy_url(self) -> str:
        """SQLAlchemy-compatible URL with connection parameters"""
        return f"{self.url}?sslmode={self.ssl_mode}&pool_size={self.pool_size}&max_overflow={self.max_overflow}"

    class Config:
        env_prefix = "DB_"


class RedisSettings(BaseSettings):
    """Redis/Cache configuration"""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    ssl: bool = Field(default=False, description="Use SSL for Redis")
    ttl_seconds: int = Field(default=3600, description="Default cache TTL")

    @computed_field
    @property
    def url(self) -> str:
        """Construct Redis URL"""
        auth = f":{self.password}@" if self.password else ""
        protocol = "rediss" if self.ssl else "redis"
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"

    class Config:
        env_prefix = "REDIS_"


class AzureSettings(BaseSettings):
    """Azure cloud services configuration"""

    storage_connection_string: str = Field(
        default="DefaultEndpointsProtocol=https;AccountName=devaccount;AccountKey=devkey;EndpointSuffix=core.windows.net",
        description="Azure Storage connection string"
    )
    storage_account_name: str = Field(default="frauddetection", description="Storage account name")
    storage_account_key: str = Field(default="devkey", description="Storage account key")
    container_models: str = Field(default="models", description="Models container")
    container_reports: str = Field(default="reports", description="Reports container")
    container_data: str = Field(default="data", description="Data container")

    data_lake_path: str = Field(default="abfss://data@frauddetection.dfs.core.windows.net", description="Data lake path")

    key_vault_url: Optional[str] = Field(default=None, description="Key Vault URL")

    event_hub_connection_string: Optional[str] = Field(default=None, description="Event Hub connection string")
    event_hub_name: str = Field(default="fraud-transactions", description="Event Hub name")

    class Config:
        env_prefix = "AZURE_"


class KafkaSettings(BaseSettings):
    """Kafka message queue configuration"""

    bootstrap_servers: str = Field(default="localhost:9092", description="Bootstrap servers")
    topic_transactions: str = Field(default="fraud-transactions", description="Transactions topic")
    consumer_group: str = Field(default="fraud-detection-pipeline", description="Consumer group")
    auto_offset_reset: str = Field(default="earliest", description="Auto offset reset")
    batch_size: int = Field(default=1000, description="Batch size")
    timeout_ms: int = Field(default=60000, description="Timeout milliseconds")

    @computed_field
    @property
    def bootstrap_servers_list(self) -> List[str]:
        """Convert comma-separated servers to list"""
        return [s.strip() for s in self.bootstrap_servers.split(",")]

    class Config:
        env_prefix = "KAFKA_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration"""

    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus")
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format")
    enable_audit_log: bool = Field(default=True, description="Enable audit logging")

    class Config:
        env_prefix = "MONITORING_"


class SecuritySettings(BaseSettings):
    """Security configuration"""

    secret_key: str = Field(default="change-me-in-production", description="Secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Token expiration")
    api_key_header: str = Field(default="X-API-Key", description="API key header")

    class Config:
        env_prefix = "SECURITY_"


class AlertSettings(BaseSettings):
    """Alerting configuration"""

    email_enabled: bool = Field(default=True, description="Enable email alerts")
    email_smtp_host: str = Field(default="smtp.gmail.com", description="SMTP host")
    email_smtp_port: int = Field(default=587, description="SMTP port")
    email_username: str = Field(default="", description="SMTP username")
    email_password: str = Field(default="", description="SMTP password")
    email_from: str = Field(default="fraud-detection@example.com", description="From email")
    email_recipients: List[str] = Field(default=["ml-team@example.com"], description="Email recipients")

    slack_enabled: bool = Field(default=False, description="Enable Slack alerts")
    slack_webhook: str = Field(default="", description="Slack webhook URL")
    slack_channel: str = Field(default="#fraud-alerts", description="Slack channel")

    max_alerts_per_hour: int = Field(default=5, description="Max alerts per hour")
    alert_debounce_minutes: int = Field(default=30, description="Alert debounce minutes")

    class Config:
        env_prefix = "ALERT_"

    @field_validator("email_recipients", mode="before")
    @classmethod
    def parse_email_recipients(cls, v):
        """Parse email recipients from comma-separated string."""
        if isinstance(v, str):
            return [email.strip() for email in v.split(",") if email.strip()]
        return v


class MLflowSettings(BaseSettings):
    """MLflow configuration"""

    tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking URI")
    experiment_name: str = Field(default="fraud-detection", description="Experiment name")
    model_name: str = Field(default="fraud-detection-ensemble", description="Model name")

    class Config:
        env_prefix = "MLFLOW_"


# Module-specific settings
class APISettings(BaseSettings):
    """API module specific settings"""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, description="Uvicorn workers")
    reload: bool = Field(default=False, description="Auto reload")

    version: str = Field(default="1.0.0", description="API version")
    title: str = Field(default="Fraud Detection API", description="API title")
    description: str = Field(default="Real-time fraud detection API", description="API description")

    fraud_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Fraud threshold")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold")

    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Requests per period")
    rate_limit_period: int = Field(default=60, description="Rate limit period")

    cors_origins: List[str] = Field(default=["http://localhost:3000"], description="CORS origins")
    cors_allow_credentials: bool = Field(default=True, description="Allow credentials")
    cors_allow_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS"], description="Allowed methods")
    cors_allow_headers: List[str] = Field(default=["*"], description="Allowed headers")

    enable_batch_prediction: bool = Field(default=True, description="Enable batch prediction")
    enable_shap_explanation: bool = Field(default=True, description="Enable SHAP explanations")
    enable_model_reload: bool = Field(default=True, description="Enable model reload")
    enable_cache: bool = Field(default=True, description="Enable caching")

    model_path: str = Field(default="/app/models", description="Model path")
    model_version: str = Field(default="v1.0.0", description="Model version")

    # Model names
    xgboost_model_name: str = Field(default="xgboost_fraud_model.pkl", description="XGBoost model filename")
    random_forest_model_name: str = Field(default="random_forest_fraud_model.pkl", description="Random Forest model filename")
    nn_model_name: str = Field(default="nn_fraud_model.pth", description="Neural Network model filename")
    isolation_forest_model_name: str = Field(default="isolation_forest_model.pkl", description="Isolation Forest model filename")

    # SHAP explainer names
    shap_explainer_xgb_name: str = Field(default="shap_explainer_xgb.pkl", description="XGBoost SHAP explainer")
    shap_explainer_rf_name: str = Field(default="shap_explainer_rf.pkl", description="Random Forest SHAP explainer")
    shap_explainer_nn_name: str = Field(default="shap_explainer_nn.pkl", description="Neural Network SHAP explainer")
    shap_explainer_iforest_name: str = Field(default="shap_explainer_iforest.pkl", description="Isolation Forest SHAP explainer")

    prometheus_port: int = Field(default=9090, description="Prometheus port")

    class Config:
        env_prefix = "API_"

    @field_validator("cors_origins", "cors_allow_methods", "cors_allow_headers", mode="before")
    @classmethod
    def parse_comma_separated(cls, v):
        """Parse comma-separated values to list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v


class DataSettings(BaseSettings):
    """Data module specific settings"""

    api_url: str = Field(default="http://api:8000", description="API URL")
    api_timeout_seconds: int = Field(default=60, description="API timeout")

    webapp_url: str = Field(default="", description="Web app URL")
    webapp_timeout_seconds: int = Field(default=30, description="Web app timeout")

    prometheus_port: int = Field(default=9092, description="Prometheus port")
    enable_profiling: bool = Field(default=False, description="Enable profiling")
    enable_data_validation: bool = Field(default=True, description="Enable data validation")

    class Config:
        env_prefix = "DATA_"


class DriftSettings(BaseSettings):
    """Drift detection module specific settings"""

    # Drift thresholds
    data_drift_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Data drift threshold")
    target_drift_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Target drift threshold")
    concept_drift_threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Concept drift threshold")

    # Performance baselines
    baseline_recall: float = Field(default=0.98, ge=0.0, le=1.0, description="Baseline recall")
    baseline_precision: float = Field(default=0.95, ge=0.0, le=1.0, description="Baseline precision")
    baseline_fpr: float = Field(default=0.015, ge=0.0, le=1.0, description="Baseline FPR")
    baseline_fraud_rate: float = Field(default=0.002, ge=0.0, le=1.0, description="Baseline fraud rate")

    # Monitoring windows
    hourly_window_size: int = Field(default=3600, description="Hourly window size")
    daily_window_size: int = Field(default=86400, description="Daily window size")
    weekly_window_size: int = Field(default=604800, description="Weekly window size")

    # Retraining
    min_samples_for_drift: int = Field(default=1000, description="Min samples for drift")
    retraining_cooldown_hours: int = Field(default=48, description="Retraining cooldown")
    consecutive_drift_detections: int = Field(default=3, description="Consecutive detections needed")

    # Reports
    report_enabled: bool = Field(default=True, description="Enable reports")
    report_retention_days: int = Field(default=90, description="Report retention")
    report_format: str = Field(default="html", description="Report format")

    prometheus_port: int = Field(default=9091, description="Prometheus port")

    class Config:
        env_prefix = "DRIFT_"


class TrainingSettings(BaseSettings):
    """Training module specific settings"""

    # Data paths
    train_data_path: str = Field(default="/data/train.csv", description="Training data path")
    test_data_path: str = Field(default="/data/test.csv", description="Test data path")
    model_output_dir: str = Field(default="/models", description="Model output directory")
    checkpoint_dir: str = Field(default="/checkpoints", description="Checkpoint directory")

    # Training parameters
    batch_size: int = Field(default=1024, description="Batch size")
    epochs: int = Field(default=10, description="Training epochs")
    validation_split: float = Field(default=0.2, ge=0.0, le=1.0, description="Validation split")
    early_stopping_patience: int = Field(default=5, description="Early stopping patience")

    # Cross-validation
    cv_folds: int = Field(default=5, description="CV folds")
    cv_strategy: str = Field(default="stratified", description="CV strategy")

    # Feature engineering
    enable_feature_selection: bool = Field(default=True, description="Enable feature selection")
    feature_importance_threshold: float = Field(default=0.01, ge=0.0, le=1.0, description="Feature importance threshold")
    enable_scaling: bool = Field(default=True, description="Enable scaling")
    scaler_type: str = Field(default="standard", description="Scaler type")

    # Evaluation
    eval_metrics: List[str] = Field(default=["precision", "recall", "f1", "auc", "accuracy"], description="Evaluation metrics")
    eval_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Evaluation threshold")

    prometheus_port: int = Field(default=9093, description="Prometheus port")

    class Config:
        env_prefix = "TRAINING_"

    @field_validator("eval_metrics", mode="before")
    @classmethod
    def parse_eval_metrics(cls, v):
        """Parse evaluation metrics from comma-separated string."""
        if isinstance(v, str):
            return [metric.strip() for metric in v.split(",")]
        return v


class AirflowSettings(BaseSettings):
    """Airflow module specific settings"""

    home: str = Field(default="/opt/airflow", description="Airflow home")
    database_url: str = Field(default="postgresql://airflow:airflow@postgres:5432/airflow_db", description="Airflow database")
    executor: str = Field(default="LocalExecutor", description="Executor")
    parallelism: int = Field(default=32, description="Parallelism")
    max_active_runs_per_dag: int = Field(default=3, description="Max active runs per DAG")

    # Service URLs
    api_base_url: str = Field(default="http://fraud-api:8000", description="API base URL")
    data_base_url: str = Field(default="http://fraud-data:8001", description="Data base URL")
    drift_base_url: str = Field(default="http://fraud-drift:8002", description="Drift base URL")

    # Training config
    min_training_samples: int = Field(default=10000, description="Min training samples")
    training_cooldown_hours: int = Field(default=48, description="Training cooldown")

    # Databricks
    databricks_host: str = Field(default="", description="Databricks host")
    databricks_token: str = Field(default="", description="Databricks token")
    databricks_training_job_id: int = Field(default=0, description="Databricks job ID")

    # Azure ACR
    azure_acr_login_server: str = Field(default="frauddetection.azurecr.io", description="ACR login server")

    training_dag_id: str = Field(default="02_model_training", description="Training DAG ID")

    class Config:
        env_prefix = "AIRFLOW_"


class GlobalSettings(BaseSettings):
    """Global settings for the entire fraud detection project"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    # Environment
    environment: str = Field(default=Environment.DEVELOPMENT, description="Environment")
    debug: bool = Field(default=False, description="Debug mode")

    # Shared services
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    azure: AzureSettings = Field(default_factory=AzureSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)

    # Module-specific settings
    api: APISettings = Field(default_factory=APISettings)
    data: DataSettings = Field(default_factory=DataSettings)
    drift: DriftSettings = Field(default_factory=DriftSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    airflow: AirflowSettings = Field(default_factory=AirflowSettings)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment value."""
        valid_envs = [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION, Environment.TEST]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()


# Global settings instance
_settings: Optional[GlobalSettings] = None


def get_settings() -> GlobalSettings:
    """Get global settings instance (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = GlobalSettings()
    return _settings


def get_module_settings(module_name: str) -> BaseSettings:
    """Get settings for a specific module"""
    settings = get_settings()
    module_map = {
        "api": settings.api,
        "data": settings.data,
        "drift": settings.drift,
        "training": settings.training,
        "airflow": settings.airflow,
    }
    if module_name not in module_map:
        raise ValueError(f"Unknown module: {module_name}")
    return module_map[module_name]


# Convenience functions for backward compatibility
def get_database_url() -> str:
    """Get database URL"""
    return get_settings().database.url


def get_redis_url() -> str:
    """Get Redis URL"""
    return get_settings().redis.url


def get_azure_storage_connection_string() -> str:
    """Get Azure storage connection string"""
    return get_settings().azure.storage_connection_string


# Export commonly used settings
settings = get_settings()