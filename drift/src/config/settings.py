"""
Configuration settings for drift detection component.
Supports environment variables for cloud deployment.
"""
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Drift detection settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=(),
        extra="allow"  # Allow extra fields
    )
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://localhost:5432/fraud_db",
        description="PostgreSQL connection URL"
    )
    
    # Drift Thresholds
    data_drift_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="PSI threshold for data drift detection"
    )
    target_drift_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative change threshold for target drift"
    )
    concept_drift_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Performance drop threshold for concept drift"
    )
    drift_recall_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable recall before concept drift alert"
    )
    drift_fpr_threshold: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable false positive rate before concept drift alert"
    )
    
    # Performance Baselines
    baseline_recall: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description="Baseline recall from validation set"
    )
    baseline_precision: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Baseline precision from validation set"
    )
    baseline_fpr: float = Field(
        default=0.015,
        ge=0.0,
        le=1.0,
        description="Baseline false positive rate"
    )
    baseline_fraud_rate: float = Field(
        default=0.002,
        ge=0.0,
        le=1.0,
        description="Baseline fraud rate (0.2%)"
    )
    
    # Monitoring Windows
    hourly_window_size: int = Field(
        default=3600,
        description="Hourly monitoring window in seconds"
    )
    daily_window_size: int = Field(
        default=86400,
        description="Daily monitoring window in seconds"
    )
    weekly_window_size: int = Field(
        default=604800,
        description="Weekly monitoring window in seconds"
    )
    
    # Retraining Configuration
    min_samples_for_drift: int = Field(
        default=1000,
        description="Minimum samples before drift detection"
    )
    retraining_cooldown_hours: int = Field(
        default=48,
        description="Hours to wait between retraining"
    )
    consecutive_drift_detections: int = Field(
        default=3,
        description="Required consecutive detections before retraining"
    )
    
    # Alerting Configuration
    alert_email_enabled: bool = Field(
        default=True,
        description="Enable email alerts"
    )
    alert_email_recipients: List[str] = Field(
        default=["ml-team@example.com"],
        description="Email recipients for alerts"
    )
    alert_email_smtp_host: str = Field(
        default="smtp.gmail.com",
        description="SMTP server host"
    )
    alert_email_smtp_port: int = Field(
        default=587,
        description="SMTP server port"
    )
    alert_email_username: str = Field(
        default="",
        description="SMTP username"
    )
    alert_email_password: str = Field(
        default="",
        description="SMTP password"
    )
    alert_email_from: str = Field(
        default="fraud-detection@example.com",
        description="Sender email address"
    )
    
    alert_slack_enabled: bool = Field(
        default=False,
        description="Enable Slack alerts"
    )
    alert_slack_webhook: str = Field(
        default="",
        description="Slack webhook URL"
    )
    alert_slack_channel: str = Field(
        default="#fraud-alerts",
        description="Slack channel for alerts"
    )
    
    alert_max_per_hour: int = Field(
        default=5,
        description="Maximum alerts per hour (rate limit)"
    )
    alert_debounce_minutes: int = Field(
        default=30,
        description="Minutes to debounce duplicate alerts"
    )
    
    # Prometheus Configuration
    prometheus_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    prometheus_port: int = Field(
        default=9091,
        description="Prometheus metrics port"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)"
    )
    
    # Azure Blob Storage (for reports)
    azure_storage_connection_string: str = Field(
        default="",
        description="Azure Storage connection string"
    )
    azure_storage_container_reports: str = Field(
        default="drift-reports",
        description="Container for drift reports"
    )
    
    # Report Configuration
    report_enabled: bool = Field(
        default=True,
        description="Enable daily drift reports"
    )
    report_retention_days: int = Field(
        default=90,
        description="Days to retain reports"
    )
    report_format: str = Field(
        default="html",
        description="Report format (html or pdf)"
    )
    
    # Airflow Integration
    airflow_api_url: str = Field(
        default="http://localhost:8080/api/v1",
        description="Airflow REST API URL"
    )
    airflow_username: str = Field(
        default="admin",
        description="Airflow API username"
    )
    airflow_password: str = Field(
        default="admin",
        description="Airflow API password"
    )
    training_dag_id: str = Field(
        default="02_model_training",
        description="DAG ID for model retraining"
    )
    
    # Environment
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    @field_validator("alert_email_recipients", mode="before")
    @classmethod
    def parse_email_recipients(cls, v):
        """Parse email recipients from comma-separated string."""
        if isinstance(v, str):
            return [email.strip() for email in v.split(",") if email.strip()]
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
        valid_envs = ["development", "staging", "production", "test"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()
    
    @field_validator("report_format")
    @classmethod
    def validate_report_format(cls, v):
        """Validate report format."""
        valid_formats = ["html", "pdf"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Report format must be one of {valid_formats}")
        return v.lower()


# Global settings instance
settings = Settings()
