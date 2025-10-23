"""
Configuration settings for Airflow module
"""
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class AirflowSettings(BaseSettings):
    """Airflow module settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=()
    )
    
    # ========== Airflow Core ==========
    airflow_home: str = Field(
        default="/opt/airflow",
        description="Airflow home directory"
    )
    
    airflow_database_url: str = Field(
        default="postgresql://airflow:airflow@postgres:5432/airflow_db",
        description="Airflow metadata database (SEPARATE from fraud_db)"
    )
    
    executor: str = Field(
        default="LocalExecutor",
        description="Airflow executor (LocalExecutor, CeleryExecutor)"
    )
    
    parallelism: int = Field(
        default=32,
        description="Maximum parallel tasks across all DAGs"
    )
    
    max_active_runs_per_dag: int = Field(
        default=3,
        description="Max concurrent runs per DAG"
    )
    
    # ========== Fraud Detection Database (READ-ONLY pour Airflow) ==========
    fraud_database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/fraud_db",
        description="Fraud detection application database"
    )
    
    # ========== Modules Endpoints ==========
    api_base_url: str = Field(
        default="http://fraud-api:8000",
        description="Fraud Detection API base URL"
    )
    
    data_base_url: str = Field(
        default="http://fraud-data:8001",
        description="Data pipeline service base URL"
    )
    
    drift_base_url: str = Field(
        default="http://fraud-drift:8002",
        description="Drift detection service base URL"
    )
    
    # ========== MLflow ==========
    mlflow_tracking_uri: str = Field(
        default="http://mlflow:5000",
        description="MLflow tracking server URI"
    )
    
    mlflow_model_name: str = Field(
        default="fraud_detection_ensemble",
        description="MLflow registered model name"
    )
    
    mlflow_experiment_name: str = Field(
        default="/fraud-detection/experiments",
        description="MLflow experiment name"
    )
    
    # ========== Databricks ==========
    databricks_host: str = Field(
        default="https://adb-xxx.azuredatabricks.net",
        description="Databricks workspace URL"
    )
    
    databricks_token: str = Field(
        default="",
        description="Databricks access token"
    )
    
    databricks_training_job_id: int = Field(
        default=12345,
        description="Databricks job ID for model training"
    )
    
    # ========== Azure ==========
    azure_storage_connection_string: str = Field(
        default="",
        description="Azure Storage connection string"
    )
    
    azure_acr_login_server: str = Field(
        default="frauddetection.azurecr.io",
        description="Azure Container Registry server"
    )
    
    # ========== Training Configuration ==========
    min_training_samples: int = Field(
        default=10000,
        description="Minimum new samples before retraining"
    )
    
    training_cooldown_hours: int = Field(
        default=48,
        description="Minimum hours between trainings"
    )
    
    # ========== Drift Thresholds ==========
    data_drift_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="PSI threshold for data drift"
    )
    
    concept_drift_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Performance drop threshold"
    )
    
    # ========== Alerting ==========
    alert_email_enabled: bool = Field(
        default=True,
        description="Enable email alerts"
    )
    
    alert_email_recipients: List[str] = Field(
        default=["ml-team@example.com"],
        description="Alert email recipients"
    )
    
    # ========== Logging ==========
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # ========== Environment ==========
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)"
    )


# Global settings instance
settings = AirflowSettings()
