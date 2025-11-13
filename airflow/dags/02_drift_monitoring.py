"""
DAG 02: Drift Monitoring Pipeline
Runs hourly to detect data, target, and concept drift in Docker containers

Priority: #1 (MOST CRITICAL)
Refactored to use DockerOperator for production deployment
"""

import os
from datetime import datetime, timedelta

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from airflow import DAG

# Import centralized configuration
from config.constants import (
    DOCKER_IMAGE_DRIFT,
    DOCKER_NETWORK,
    ENV_VARS,
    SCHEDULES,
    THRESHOLDS,
)

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["ml-alerts@frauddetection.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Define DAG
with DAG(
    "02_drift_monitoring",
    default_args=default_args,
    description="Hourly drift detection via Docker container",
    schedule_interval=SCHEDULES["DRIFT_MONITORING"],  # Every hour
    start_date=days_ago(1),
    catchup=False,
    tags=["monitoring", "drift", "critical", "docker"],
) as dag:
    # Task: Run drift detection in Docker container
    # The drift module container will:
    # 1. Load reference data from PostgreSQL
    # 2. Load current production data
    # 3. Apply preprocessing via fraud_detection_common
    # 4. Calculate PSI, KS stats
    # 5. Save drift metrics to PostgreSQL
    # 6. Expose metrics via Prometheus endpoint
    run_drift_detection = DockerOperator(
        task_id="run_drift_detection",
        image=DOCKER_IMAGE_DRIFT,
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        command="python -m src.pipelines.hourly_monitoring",
        environment={
            "POSTGRES_HOST": ENV_VARS["POSTGRES_HOST"],
            "POSTGRES_PORT": ENV_VARS["POSTGRES_PORT"],
            "POSTGRES_DB": ENV_VARS["POSTGRES_DB"],
            "POSTGRES_USER": ENV_VARS["POSTGRES_USER"],
            "POSTGRES_PASSWORD": ENV_VARS["POSTGRES_PASSWORD"],
            "MLFLOW_TRACKING_URI": ENV_VARS["MLFLOW_TRACKING_URI"],
            "DATA_DRIFT_THRESHOLD": str(THRESHOLDS["PSI_THRESHOLD"]),
            "CONCEPT_DRIFT_THRESHOLD": str(THRESHOLDS["CONCEPT_DRIFT_THRESHOLD"]),
            "MONITORING_WINDOW_HOURS": "24",
            "LOG_LEVEL": "INFO",
            # API configuration for drift detection
            "API_BASE_URL": os.getenv("API_BASE_URL", "http://api:8000"),
            "API_USERNAME": os.getenv("API_USERNAME_DRIFT", "admin"),
            "API_PASSWORD": os.getenv("API_PASSWORD_DRIFT", "admin123"),
        },
        mount_tmp_dir=False,
    )

    # Dependencies: Single task (no complex branching in Airflow)
    # Note: The drift container itself handles all logic including:
    # - Deciding if retraining is needed (saves decision to DB)
    # - Sending alerts via configured channels
    # - Saving metrics to PostgreSQL
    # Airflow only orchestrates the execution schedule
    run_drift_detection
