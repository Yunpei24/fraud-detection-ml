"""
DAG 00: Batch Prediction Pipeline (Manual Trigger)

Consumes 1000 transactions from Kafka in batch, makes predictions,
and saves to PostgreSQL using Docker container.

Schedule: Manual trigger only
Mode: On-demand
"""

import logging
import os
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.trigger_rule import TriggerRule

# Import centralized configuration
from config.constants import (
    DOCKER_IMAGE_DATA,
    DOCKER_NETWORK,
    ENV_VARS,
)

logger = logging.getLogger(__name__)

# Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["emmanuelyunpei@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}


def validate_batch_config(**context):
    """
    Validate batch configuration before execution
    """
    logger.info(f"Validating batch configuration")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Docker image: {DOCKER_IMAGE_DATA}")
    logger.info(f"  Docker network: {DOCKER_NETWORK}")

    # Push config to XCom
    context["task_instance"].xcom_push(key="batch_size", value=BATCH_SIZE)

    return True


def log_batch_completion(**context):
    """
    Log batch completion metrics
    """
    logger.info("Batch prediction pipeline completed")
    logger.info(f"  Batch size: {BATCH_SIZE} transactions")
    logger.info("  Check Docker logs for detailed metrics")


# Define DAG
with DAG(
    "00_batch_prediction",
    default_args=default_args,
    description="Batch prediction pipeline - manual trigger using Docker",
    schedule_interval=None,  # Manual trigger only
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["batch", "prediction", "kafka", "manual", "docker"],
    doc_md=__doc__,
) as dag:
    # Task 1: Validate configuration
    validate_config = PythonOperator(
        task_id="validate_batch_config",
        python_callable=validate_batch_config,
        provide_context=True,
    )

    # Task 2: Run batch prediction via Docker
    run_batch_prediction = DockerOperator(
        task_id="run_batch_prediction",
        image=DOCKER_IMAGE_DATA,
        command=f"python -m src.pipelines.realtime_pipeline --mode batch --count {BATCH_SIZE}",
        environment=ENV_VARS,
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove=True,
        mount_tmp_dir=False,
        trigger_rule=TriggerRule.NONE_FAILED,
        execution_timeout=timedelta(minutes=10),
        doc_md="""
        Runs batch prediction pipeline in Docker container:
        1. Consumes transactions from Kafka
        2. Cleans and preprocesses data
        3. Makes predictions via API (with JWT auth)
        4. Saves results to PostgreSQL
        5. Sends fraud alerts to web app
        """,
    )

    # Task 3: Log completion
    log_completion = PythonOperator(
        task_id="log_batch_completion",
        python_callable=log_batch_completion,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )
