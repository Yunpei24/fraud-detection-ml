"""
DAG 00: Transaction Producer (Kafka Generator)

Generates synthetic transactions and sends them to Kafka for real-time processing.
This DAG runs BEFORE or IN PARALLEL with 00_realtime_streaming.py (consumer).

Schedule: Every 5 seconds (continuous production)
Mode: Automatic
Execution: Runs via Docker container
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

# Producer configuration
PRODUCER_MODE = "batch"  # or "stream"
TRANSACTIONS_PER_RUN = 50  # Generate 50 transactions every 5 seconds
FRAUD_RATE = 0.05  # 5% fraud rate

logger = logging.getLogger(__name__)

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["ml-alerts@frauddetection.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(seconds=15),
}


def validate_producer_config(**context):
    """
    Validate producer configuration before execution
    """
    logger.info(f" Validating Transaction Producer configuration")
    logger.info(f"  Mode: {PRODUCER_MODE}")
    logger.info(f"  Transactions per run: {TRANSACTIONS_PER_RUN}")
    logger.info(f"  Fraud rate: {FRAUD_RATE * 100}%")
    logger.info(f"  Docker image: {DOCKER_IMAGE_DATA}")
    logger.info(f"  Docker network: {DOCKER_NETWORK}")

    # Push config to XCom
    context["task_instance"].xcom_push(
        key="transactions_per_run", value=TRANSACTIONS_PER_RUN
    )
    context["task_instance"].xcom_push(key="fraud_rate", value=FRAUD_RATE)

    return True


def log_producer_completion(**context):
    """
    Log producer execution metrics
    """
    logger.info(f" Transaction Producer completed")
    logger.info(f"  Generated {TRANSACTIONS_PER_RUN} transactions")
    logger.info(f"  Sent to Kafka topic: fraud-detection-transactions")
    logger.info(f"  Check Docker logs for detailed metrics")


# Define DAG
with DAG(
    "00_transaction_producer",
    default_args=default_args,
    description="Kafka transaction producer (generates synthetic transactions every 5 seconds)",
    schedule_interval="*/5 * * * * *",  # Every 5 seconds (produces FASTER than consumer)
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,  # Only one producer at a time
    tags=["producer", "kafka", "simulation", "transactions", "docker"],
    doc_md=__doc__,
) as dag:
    # Task 1: Validate producer configuration
    validate_config = PythonOperator(
        task_id="validate_producer_config",
        python_callable=validate_producer_config,
        provide_context=True,
    )

    # Task 2: Generate and send transactions to Kafka via Docker
    generate_transactions = DockerOperator(
        task_id="generate_transactions",
        image=DOCKER_IMAGE_DATA,
        command=f"python -m src.ingestion.transaction_simulator --mode {PRODUCER_MODE} --count {TRANSACTIONS_PER_RUN} --fraud-rate {FRAUD_RATE} --kafka kafka:9092 --topic fraud-detection-transactions",
        environment=ENV_VARS,
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove=True,
        mount_tmp_dir=False,
        trigger_rule=TriggerRule.NONE_FAILED,
        execution_timeout=timedelta(seconds=20),
        doc_md="""
        Generates synthetic transactions and sends them to Kafka:
        
        1.  Generates transactions with realistic patterns:
           - 28 PCA features (V1-V28)
           - Transaction amount (log-normal distribution)
           - Time (seconds since first transaction)
           - Class (0=legitimate, 1=fraud)

        2.  Sends to Kafka topic: fraud-detection-transactions

        3.  Metrics tracked:
           - Total transactions generated
           - Fraud vs legitimate ratio
           - Average transaction amount
           - Kafka send success rate
        
        Environment variables used:
        - KAFKA_BOOTSTRAP_SERVERS (e.g., kafka:9092)
        - KAFKA_TOPIC (fraud-detection-transactions)
        
        Transaction features:
        - V1-V28: PCA-transformed features (fraud patterns)
        - amount: $0.01 to $25,691 (log-normal)
        - Time: Sequential timestamp
        - transaction_id: UUID
        """,
    )

    # Task 3: Log completion
    log_completion = PythonOperator(
        task_id="log_producer_completion",
        python_callable=log_producer_completion,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Define task dependencies
    validate_config >> generate_transactions >> log_completion
