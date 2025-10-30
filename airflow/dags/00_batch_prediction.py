"""
DAG 00: Batch Prediction Pipeline (Manual Trigger)

Consomme 1000 transactions de Kafka en batch, fait des prédictions,
et sauvegarde dans PostgreSQL.

Schedule: Manual trigger uniquement
Mode: On-demand
"""

import logging
import os
import sys
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from airflow import DAG

# Configuration
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "fraud-detection-network")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "fraud-detection-transactions")
API_URL = os.getenv("API_URL", "http://api:8000")
WEBAPP_URL = os.getenv("WEBAPP_URL", None)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))

logger = logging.getLogger(__name__)

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["ml-alerts@frauddetection.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}


def validate_kafka_topic(**context):
    """
    Validate Kafka topic exists and has messages
    """
    logger.info(f"🔍 Validating Kafka topic: {KAFKA_TOPIC}")

    try:
        from kafka import KafkaConsumer
        from kafka.errors import KafkaError
        from kafka.structs import TopicPartition

        # Create consumer
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            consumer_timeout_ms=5000,
            auto_offset_reset="earliest",
        )

        # Check partitions
        partitions = consumer.partitions_for_topic(KAFKA_TOPIC)
        if not partitions:
            raise ValueError(f"Topic '{KAFKA_TOPIC}' has no partitions")

        logger.info(f"Topic '{KAFKA_TOPIC}' validated. Partitions: {partitions}")

        # Get approximate message count (check lag)
        consumer.poll(timeout_ms=1000)
        consumer.seek_to_beginning()
        total_messages = 0

        for partition in partitions:
            tp = TopicPartition(KAFKA_TOPIC, partition)
            consumer.assign([tp])
            consumer.seek_to_beginning(tp)
            beginning_offset = consumer.position(tp)
            consumer.seek_to_end(tp)
            end_offset = consumer.position(tp)
            partition_messages = end_offset - beginning_offset
            total_messages += partition_messages
            logger.info(f"   Partition {partition}: {partition_messages} messages")

        consumer.close()

        logger.info(f"Total available messages: {total_messages}")

        if total_messages < BATCH_SIZE:
            logger.warning(
                f" Only {total_messages} messages available, but batch size is {BATCH_SIZE}"
            )

        # Push to XCom
        context["task_instance"].xcom_push(
            key="available_messages", value=total_messages
        )

        return total_messages

    except Exception as e:
        logger.error(f"Kafka topic validation failed: {str(e)}", exc_info=True)
        raise


def run_batch_prediction(**context):
    """
    Execute batch prediction pipeline

    Consomme 1000 transactions depuis Kafka, fait des prédictions,
    et sauvegarde les résultats dans PostgreSQL
    """
    logger.info(f"Starting batch prediction pipeline (size: {BATCH_SIZE})")

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

        # Execute batch
        logger.info(f" Consuming {BATCH_SIZE} messages from Kafka")
        result = pipeline.execute_batch(count=BATCH_SIZE)

        if result["status"] == "success":
            logger.info(f"Batch pipeline completed successfully")
            logger.info(f"   - Consumed: {result['consumed']}")
            logger.info(f"   - Cleaned: {result['cleaned']}")
            logger.info(f"   - Frauds detected: {result['fraud_detected']}")
            logger.info(f"   - Fraud rate: {result['fraud_rate']:.2%}")
            logger.info(f"   - Saved: {result['saved']}")
            logger.info(f"   - Duration: {result['elapsed_seconds']:.2f}s")

            # Push metrics to XCom
            context["task_instance"].xcom_push(key="consumed", value=result["consumed"])
            context["task_instance"].xcom_push(key="cleaned", value=result["cleaned"])
            context["task_instance"].xcom_push(
                key="fraud_detected", value=result["fraud_detected"]
            )
            context["task_instance"].xcom_push(
                key="fraud_rate", value=result["fraud_rate"]
            )
            context["task_instance"].xcom_push(key="saved", value=result["saved"])
            context["task_instance"].xcom_push(
                key="elapsed_seconds", value=result["elapsed_seconds"]
            )

            return result
        else:
            logger.error(f"Batch pipeline failed: {result.get('message')}")
            raise Exception(f"Pipeline failed: {result.get('message')}")

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise


def generate_batch_report(**context):
    """
    Generate comprehensive batch prediction report
    """
    ti = context["task_instance"]

    available = (
        ti.xcom_pull(task_ids="validate_kafka_topic", key="available_messages") or 0
    )
    consumed = ti.xcom_pull(task_ids="run_batch_prediction", key="consumed") or 0
    cleaned = ti.xcom_pull(task_ids="run_batch_prediction", key="cleaned") or 0
    fraud_detected = (
        ti.xcom_pull(task_ids="run_batch_prediction", key="fraud_detected") or 0
    )
    fraud_rate = ti.xcom_pull(task_ids="run_batch_prediction", key="fraud_rate") or 0.0
    saved = ti.xcom_pull(task_ids="run_batch_prediction", key="saved") or 0
    elapsed = ti.xcom_pull(task_ids="run_batch_prediction", key="elapsed_seconds") or 0

    logger.info("=" * 80)
    logger.info("BATCH PREDICTION REPORT")
    logger.info("=" * 80)
    logger.info(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info("")
    logger.info("DATA INGESTION")
    logger.info(f"   Available in Kafka: {available:,} messages")
    logger.info(f"   Consumed: {consumed:,} transactions")
    logger.info(f"   Cleaned: {cleaned:,} transactions")
    logger.info(
        f"   Data quality: {cleaned/consumed*100:.1f}%"
        if consumed > 0
        else "   Data quality: N/A"
    )
    logger.info("")
    logger.info("FRAUD DETECTION")
    logger.info(f"   Frauds detected: {fraud_detected:,}")
    logger.info(f"   Fraud rate: {fraud_rate:.2%}")
    logger.info(f"   Legitimate: {cleaned - fraud_detected:,}")
    logger.info("")
    logger.info("DATA STORAGE")
    logger.info(f"   Saved to PostgreSQL: {saved:,} records")
    logger.info(
        f"   Save rate: {saved/cleaned*100:.1f}%"
        if cleaned > 0
        else "   Save rate: N/A"
    )
    logger.info("")
    logger.info("⚡ PERFORMANCE")
    logger.info(f"   Total duration: {elapsed:.2f}s")
    logger.info(
        f"   Throughput: {consumed/elapsed:.1f} txn/s"
        if elapsed > 0
        else "   Throughput: N/A"
    )
    logger.info("=" * 80)

    # Prometheus metrics
    print(f"fraud_detection_batch_transactions_consumed {consumed}")
    print(f"fraud_detection_batch_frauds_detected {fraud_detected}")
    print(f"fraud_detection_batch_fraud_rate {fraud_rate}")
    print(f"fraud_detection_batch_transactions_saved {saved}")
    print(f"fraud_detection_batch_duration_seconds {elapsed}")

    # Create report dictionary
    report = {
        "timestamp": datetime.now().isoformat(),
        "batch_size": BATCH_SIZE,
        "available": available,
        "consumed": consumed,
        "cleaned": cleaned,
        "fraud_detected": fraud_detected,
        "fraud_rate": fraud_rate,
        "saved": saved,
        "elapsed_seconds": elapsed,
        "throughput_tps": consumed / elapsed if elapsed > 0 else 0,
    }

    # Push report to XCom
    context["task_instance"].xcom_push(key="report", value=report)

    return report


# Define DAG
with DAG(
    "00_batch_prediction",
    default_args=default_args,
    description="Batch prediction pipeline - manual trigger (1000 transactions)",
    schedule_interval=None,  # Manual trigger only
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["batch", "prediction", "kafka", "manual"],
) as dag:
    # Task 1: Validate Kafka topic
    validate_topic = PythonOperator(
        task_id="validate_kafka_topic",
        python_callable=validate_kafka_topic,
        provide_context=True,
    )

    # Task 2: Run batch prediction
    run_prediction = PythonOperator(
        task_id="run_batch_prediction",
        python_callable=run_batch_prediction,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
    )

    # Task 3: Generate report
    generate_report = PythonOperator(
        task_id="generate_batch_report",
        python_callable=generate_batch_report,
        provide_context=True,
    )
