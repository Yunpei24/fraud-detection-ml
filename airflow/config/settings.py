"""
Configuration settings for Airflow module
Migrated to use centralized configuration for consistency.
"""

from typing import List

from config import get_settings

# Add project root to path to import centralized config
# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))


# Get centralized settings
settings = get_settings()


class AirflowSettings:
    """Airflow module settings migrated to centralized config."""

    # ========== Airflow Core ==========
    airflow_home: str = settings.airflow.home

    airflow_database_url: str = settings.airflow.database_url

    executor: str = settings.airflow.executor

    parallelism: int = settings.airflow.parallelism

    max_active_runs_per_dag: int = settings.airflow.max_active_runs_per_dag

    # ========== Fraud Detection Database (READ-ONLY pour Airflow) ==========
    fraud_database_url: str = settings.database.url

    # ========== Modules Endpoints ==========
    api_base_url: str = settings.airflow.api_base_url

    data_base_url: str = settings.airflow.data_base_url

    drift_base_url: str = settings.airflow.drift_base_url

    # ========== MLflow ==========
    mlflow_tracking_uri: str = settings.mlflow.tracking_uri

    mlflow_model_name: str = settings.mlflow.model_name

    mlflow_experiment_name: str = settings.mlflow.experiment_name

    # ========== Databricks ==========
    databricks_host: str = settings.airflow.databricks_host

    databricks_token: str = settings.airflow.databricks_token

    databricks_training_job_id: int = settings.airflow.databricks_training_job_id

    # ========== Azure ==========
    azure_storage_connection_string: str = settings.azure.storage_connection_string

    azure_acr_login_server: str = settings.airflow.azure_acr_login_server

    # ========== Training Configuration ==========
    min_training_samples: int = settings.airflow.min_training_samples

    training_cooldown_hours: int = settings.airflow.training_cooldown_hours

    # ========== Drift Thresholds ==========
    data_drift_threshold: float = settings.drift.data_drift_threshold

    concept_drift_threshold: float = settings.drift.concept_drift_threshold

    # ========== Alerting ==========
    alert_email_enabled: bool = settings.alerts.email_enabled

    alert_email_recipients: List[str] = settings.alerts.email_recipients

    # ========== Logging ==========
    log_level: str = settings.monitoring.log_level

    # ========== Environment ==========
    environment: str = settings.environment


# Global settings instance
settings_instance = AirflowSettings()
