"""
DAG 00: Real-time Streaming Prediction Pipeline

Consomme les transactions de Kafka en streaming continu toutes les 10 secondes,
fait des prédictions via l'API, et sauvegarde dans PostgreSQL.

Schedule: Toutes les 10 secondes (streaming continu)
Mode: Automatique
"""

import logging
import os
import sys
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from airflow import DAG

# Configuration
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "fraud-detection-network")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "fraud-detection-transactions")
API_URL = os.getenv("API_URL", "http://api:8000")
WEBAPP_URL = os.getenv("WEBAPP_URL", None)

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


def run_streaming_prediction(**context):
    """
    Execute streaming prediction pipeline

    Consomme les transactions depuis Kafka, fait des prédictions,
    et sauvegarde les résultats dans PostgreSQL
    """
    logger.info(" Starting streaming prediction pipeline")

    try:
        # Import realtime pipeline
        from data.src.pipelines.realtime_pipeline import RealtimePipeline

        # Create pipeline instance
        pipeline = RealtimePipeline(
            kafka_bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            kafka_topic=KAFKA_TOPIC,
            api_url=API_URL,
            webapp_url=WEBAPP_URL,
        )

        # Execute batch (consume available messages, max 100 per run)
        logger.info("Consuming messages from Kafka (max 100)")
        result = pipeline.execute_batch(count=100)

        if result["status"] == "success":
            logger.info(f"Streaming pipeline completed successfully")
            logger.info(f"   - Consumed: {result['consumed']}")
            logger.info(f"   - Frauds detected: {result['fraud_detected']}")
            logger.info(f"   - Saved: {result['saved']}")
            logger.info(f"   - Duration: {result['elapsed_seconds']:.2f}s")

            # Push metrics to XCom
            context["task_instance"].xcom_push(key="consumed", value=result["consumed"])
            context["task_instance"].xcom_push(
                key="fraud_detected", value=result["fraud_detected"]
            )
            context["task_instance"].xcom_push(key="saved", value=result["saved"])

            return result
        else:
            logger.error(f"Streaming pipeline failed: {result.get('message')}")
            raise Exception(f"Pipeline failed: {result.get('message')}")

    except Exception as e:
        logger.error(f"Streaming prediction failed: {str(e)}", exc_info=True)
        raise


def check_kafka_health(**context):
    """
    Check if Kafka is available before processing
    """
    logger.info("Checking Kafka health")

    try:
        # Try to create a consumer to test connection
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, consumer_timeout_ms=5000
        )

        # Get topics
        topics = consumer.topics()
        logger.info(f"Kafka is healthy. Topics: {topics}")

        consumer.close()
        return True

    except Exception as e:
        logger.warning(f"Kafka health check failed: {str(e)}")
        # Don't fail the DAG, just skip if Kafka is down
        return False


def log_pipeline_metrics(**context):
    """
    Log pipeline metrics for monitoring
    """
    ti = context["task_instance"]

    consumed = ti.xcom_pull(task_ids="run_streaming_prediction", key="consumed") or 0
    fraud_detected = (
        ti.xcom_pull(task_ids="run_streaming_prediction", key="fraud_detected") or 0
    )
    saved = ti.xcom_pull(task_ids="run_streaming_prediction", key="saved") or 0

    logger.info(f"Pipeline Metrics Summary:")
    logger.info(f"   - Transactions consumed: {consumed}")
    logger.info(
        f"   - Frauds detected: {fraud_detected} ({fraud_detected/consumed*100:.1f}%)"
        if consumed > 0
        else "   - Frauds detected: 0"
    )
    logger.info(f"   - Transactions saved: {saved}")

    # These metrics can be scraped by Prometheus
    print(f"fraud_detection_streaming_transactions_consumed {consumed}")
    print(f"fraud_detection_streaming_frauds_detected {fraud_detected}")
    print(f"fraud_detection_streaming_transactions_saved {saved}")


# Define DAG
with DAG(
    "00_realtime_streaming",
    default_args=default_args,
    description="Real-time streaming prediction pipeline (every 10 seconds)",
    schedule_interval="*/10 * * * * *",  # Every 10 seconds (requires Airflow 2.4+)
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,  # Only one run at a time
    tags=["streaming", "realtime", "prediction", "kafka"],
) as dag:
    # Task 1: Check Kafka health
    check_kafka = PythonOperator(
        task_id="check_kafka_health",
        python_callable=check_kafka_health,
        provide_context=True,
    )

    # Task 2: Run streaming prediction
    run_prediction = PythonOperator(
        task_id="run_streaming_prediction",
        python_callable=run_streaming_prediction,
        provide_context=True,
        execution_timeout=timedelta(seconds=30),  # Timeout after 30s
    )

    # Task 3: Log metrics
    log_metrics = PythonOperator(
        task_id="log_pipeline_metrics",
        python_callable=log_pipeline_metrics,
        provide_context=True,
    )

    # Define task dependencies
    check_kafka >> run_prediction >> log_metrics


# Documentation
dag.doc_md = """
# Real-time Streaming Prediction Pipeline

## Description
This DAG consumes transactions from Kafka in a continuous stream every 10 seconds,
makes predictions via the API, and saves the results in PostgreSQL.

## Schedule
- **Interval:** Every 10 seconds
- **Mode:** Automatic (continuous streaming)
- **Max active runs:** 1 (avoids overlap)

## Flow
1. **check_kafka_health** - Checks that Kafka is accessible
2. **run_streaming_prediction** - Consumes max 100 transactions, predicts, saves
3. **log_pipeline_metrics** - Logs metrics for Prometheus

## Configuration
- **KAFKA_BOOTSTRAP_SERVERS:** {kafka_servers}
- **KAFKA_TOPIC:** {topic}
- **API_URL:** {api_url}
- **Batch size:** 100 transactions per run

## Metrics
The following metrics are exported to Prometheus:
- `fraud_detection_streaming_transactions_consumed`
- `fraud_detection_streaming_frauds_detected`
- `fraud_detection_streaming_transactions_saved`

## Usage
This DAG starts automatically and runs continuously.
To stop it, disable the DAG in the Airflow interface.

To test manually:
```bash
docker exec fraud-airflow-scheduler airflow dags trigger 00_realtime_streaming
```

## Dependencies
- Kafka must be started: `docker compose up -d kafka`
- The simulator must publish messages: `python -m src.ingestion.transaction_simulator --mode stream`
- The API must be operational: `docker compose up -d api`
""".format(
    kafka_servers=KAFKA_BOOTSTRAP_SERVERS, topic=KAFKA_TOPIC, api_url=API_URL
)
