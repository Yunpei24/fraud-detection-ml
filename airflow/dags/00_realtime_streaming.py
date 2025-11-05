"""
DAG 00: Real-time Streaming Prediction Pipeline

Consumes transactions from Kafka in streaming mode (100 per run),
makes predictions via API, and saves to PostgreSQL.

Schedule: Every 10 seconds (continuous streaming)
Mode: Automatic
Execution: Runs via Docker container (no direct Python imports)
"""

import logging
import pendulum
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.trigger_rule import TriggerRule

# Import Docker configuration
from config import constants

# Configuration
DOCKER_IMAGE_DATA = constants.DOCKER_IMAGE_DATA
DOCKER_NETWORK = constants.DOCKER_NETWORK
ENV_VARS = constants.ENV_VARS
STREAMING_BATCH_SIZE = 100

logger = logging.getLogger(__name__)

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["ml-alerts@frauddetection.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(seconds=30),
}


def validate_streaming_config(**context):
    """
    Validate streaming configuration before execution
    """
    logger.info(f"Validating streaming configuration")
    logger.info(f"  Streaming batch size: {STREAMING_BATCH_SIZE}")
    logger.info(f"  Docker image: {DOCKER_IMAGE_DATA}")
    logger.info(f"  Docker network: {DOCKER_NETWORK}")

    # Push config to XCom
    context["task_instance"].xcom_push(key="batch_size", value=STREAMING_BATCH_SIZE)

    return True


def log_streaming_completion(**context):
    """
    Log streaming execution metrics
    """
    logger.info("Streaming prediction pipeline completed")
    logger.info(f"  Batch size: {STREAMING_BATCH_SIZE} transactions")
    logger.info("  Check Docker logs for detailed metrics")


# Define DAG
with DAG(
    "00_realtime_streaming",
    default_args=default_args,
    description="Real-time streaming prediction pipeline using Docker (every 10 seconds)",
    schedule_interval="*/10 * * * * *",  # Every 10 seconds
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["streaming", "realtime", "prediction", "kafka", "docker"],
    doc_md=__doc__,
) as dag:
    # Task 1: Validate configuration
    validate_config = PythonOperator(
        task_id="validate_streaming_config",
        python_callable=validate_streaming_config,
        provide_context=True,
    )

    # Task 2: Run streaming prediction via Docker
    run_streaming_prediction = DockerOperator(
        task_id="run_streaming_prediction",
        image=DOCKER_IMAGE_DATA,
        command=f"python -m src.pipelines.realtime_pipeline --mode batch --count {STREAMING_BATCH_SIZE}",
        environment=ENV_VARS,
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove=True,
        mount_tmp_dir=False,
        trigger_rule=TriggerRule.NONE_FAILED,
        execution_timeout=timedelta(
            seconds=60
        ),  # Increased from 30s to 60s (Kafka consumer timeout)
        doc_md="""
        Runs streaming prediction pipeline in Docker container:
        1. Consumes up to 100 transactions from Kafka
        2. Cleans and preprocesses data
        3. Makes predictions via API (with JWT auth: admin/admin123)
        4. Saves results to PostgreSQL
        5. Sends fraud alerts to web app (http://webapp:3000)
        
        Environment variables used:
        - KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, KAFKA_CONSUMER_GROUP
        - API_URL, API_USERNAME, API_PASSWORD
        - WEBAPP_URL, WEBAPP_TIMEOUT_SECONDS
        - POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
        """,
    )

    # Task 3: Log completion
    log_completion = PythonOperator(
        task_id="log_streaming_completion",
        python_callable=log_streaming_completion,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Define task dependencies
    validate_config >> run_streaming_prediction >> log_completion
