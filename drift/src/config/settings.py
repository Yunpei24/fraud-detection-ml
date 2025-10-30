"""
Configuration settings for drift detection component.
Standalone configuration for the simplified drift component.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Settings:
    """Drift detection settings for the simplified component."""

    # Database Configuration
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://fraud_user:password@localhost:5432/fraud_db"
    )

    # API Configuration (for calling drift service)
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    api_timeout: int = int(os.getenv("API_TIMEOUT", "30"))
    api_auth_token: Optional[str] = os.getenv("API_AUTH_TOKEN", None)
    reference_window_days: int = int(os.getenv("REFERENCE_WINDOW_DAYS", "30"))

    # Core Drift Thresholds (essential)
    data_drift_threshold: float = float(os.getenv("DATA_DRIFT_THRESHOLD", "0.3"))
    target_drift_threshold: float = float(os.getenv("TARGET_DRIFT_THRESHOLD", "0.5"))
    concept_drift_threshold: float = float(os.getenv("CONCEPT_DRIFT_THRESHOLD", "0.05"))

    # Retraining Configuration (essential)
    retraining_cooldown_hours: int = int(os.getenv("RETRAINING_COOLDOWN_HOURS", "48"))
    consecutive_drift_detections: int = int(
        os.getenv("CONSECUTIVE_DRIFT_DETECTIONS", "2")
    )

    # Alerting Configuration (essential)
    alert_email_enabled: bool = (
        os.getenv("ALERT_EMAIL_ENABLED", "true").lower() == "true"
    )
    alert_email_recipients: List[str] = field(
        default_factory=lambda: os.getenv(
            "ALERT_EMAIL_RECIPIENTS", "ml-team@example.com"
        ).split(",")
    )
    alert_email_smtp_host: str = os.getenv("ALERT_EMAIL_SMTP_HOST", "smtp.gmail.com")
    alert_email_smtp_port: int = int(os.getenv("ALERT_EMAIL_SMTP_PORT", "587"))
    alert_email_username: str = os.getenv("ALERT_EMAIL_USERNAME", "")
    alert_email_password: str = os.getenv("ALERT_EMAIL_PASSWORD", "")
    alert_email_from: str = os.getenv("ALERT_EMAIL_FROM", "drift-monitor@example.com")
    alert_max_per_hour: int = int(os.getenv("ALERT_MAX_PER_HOUR", "10"))
    alert_debounce_minutes: int = int(os.getenv("ALERT_DEBOUNCE_MINUTES", "5"))

    # Prometheus Configuration (essential)
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "9091"))

    # Logging Configuration (essential)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")

    # Airflow Integration (essential)
    airflow_api_url: str = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v1")
    airflow_username: str = os.getenv("AIRFLOW_USERNAME", "admin")
    airflow_password: str = os.getenv("AIRFLOW_PASSWORD", "admin")
    training_dag_id: str = os.getenv("TRAINING_DAG_ID", "01_training_pipeline")

    # Report Configuration
    report_output_dir: Optional[str] = os.getenv("REPORT_OUTPUT_DIR", None)

    # Report Configuration
    report_output_dir: Optional[str] = os.getenv("REPORT_OUTPUT_DIR", None)


# Global settings instance
settings = Settings()
