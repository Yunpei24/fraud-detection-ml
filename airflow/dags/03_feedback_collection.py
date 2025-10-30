"""
DAG 03: Feedback Collection Pipeline
Collects analyst labels to improve the model
Uses PostgreSQL hooks for database queries

Schedule: Daily at 1 AM
Refactored to remove dependencies on obsolete helpers/module_loader
"""

import os
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from airflow import DAG
# Import centralized configuration
from config.constants import ALERT_CONFIG, SCHEDULES, TABLE_NAMES

# Configuration from environment
ALERT_EMAIL = os.getenv("ALERT_EMAIL", ALERT_CONFIG["DEFAULT_EMAIL"])
MAX_FALSE_NEGATIVE_RATE = float(os.getenv("MAX_FALSE_NEGATIVE_RATE", "0.05"))

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": [ALERT_EMAIL],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def collect_analyst_labels(**context):
    """Collect labels confirmed by analysts via DatabaseService"""
    from plugins.hooks.postgres_hook import FraudPostgresHook

    hook = FraudPostgresHook()

    # Retrieve predictions with analyst labels (last 24 hours)
    query = f"""
        INSERT INTO {TABLE_NAMES['FEEDBACK_LABELS']} (transaction_id, predicted_label, analyst_label, confidence, feedback_quality)
        SELECT 
            p.transaction_id,
            p.prediction,
            t.is_fraud as analyst_label,
            p.probability,
            CASE 
                WHEN p.prediction = t.is_fraud THEN 'CORRECT'
                WHEN p.prediction = 1 AND t.is_fraud = 0 THEN 'FALSE_POSITIVE'
                WHEN p.prediction = 0 AND t.is_fraud = 1 THEN 'FALSE_NEGATIVE'
            END as feedback_quality
        FROM {TABLE_NAMES['PREDICTIONS']} p
        JOIN {TABLE_NAMES['TRANSACTIONS']} t ON p.transaction_id = t.transaction_id
        WHERE p.created_at >= NOW() - INTERVAL '24 hours'
        AND t.analyst_reviewed = true
        AND NOT EXISTS (
            SELECT 1 FROM {TABLE_NAMES['FEEDBACK_LABELS']} f 
            WHERE f.transaction_id = p.transaction_id
        )
    """

    rows_inserted = hook.execute_query(query)

    return {
        "status": "success",
        "labels_collected": rows_inserted,
        "timestamp": datetime.now().isoformat(),
    }


def analyze_feedback_quality(**context):
    """Analyze feedback quality via PostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook

    ti = context["task_instance"]
    collection_result = ti.xcom_pull(task_ids="collect_analyst_labels")

    hook = FraudPostgresHook()

    # Calculate feedback metrics
    query = f"""
        SELECT 
            COUNT(*) as total_feedback,
            SUM(CASE WHEN feedback_quality = 'CORRECT' THEN 1 ELSE 0 END) as correct,
            SUM(CASE WHEN feedback_quality = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) as false_positives,
            SUM(CASE WHEN feedback_quality = 'FALSE_NEGATIVE' THEN 1 ELSE 0 END) as false_negatives,
            AVG(confidence) as avg_confidence
        FROM {TABLE_NAMES['FEEDBACK_LABELS']}
        WHERE created_at >= NOW() - INTERVAL '7 days'
    """

    result = hook.fetch_one(query)

    total = result[0] or 1
    metrics = {
        "total_feedback": result[0] or 0,
        "correct": result[1] or 0,
        "false_positives": result[2] or 0,
        "false_negatives": result[3] or 0,
        "avg_confidence": float(result[4] or 0),
        "accuracy": (result[1] or 0) / total,
        "fp_rate": (result[2] or 0) / total,
        "fn_rate": (result[3] or 0) / total,
    }

    return metrics


def check_retraining_needed(**context):
    """Check if retraining is needed based on feedback"""
    ti = context["task_instance"]
    metrics = ti.xcom_pull(task_ids="analyze_feedback_quality")

    # Use thresholds from environment
    min_accuracy = 0.90
    max_fn_rate = MAX_FALSE_NEGATIVE_RATE

    needs_retraining = (
        metrics["accuracy"] < min_accuracy or metrics["fn_rate"] > max_fn_rate
    )

    if needs_retraining:
        reason = f"Accuracy {metrics['accuracy']:.2%} < {min_accuracy:.2%} or FN rate {metrics['fn_rate']:.2%} > {max_fn_rate:.2%}"

        return {"needs_retraining": True, "reason": reason, "metrics": metrics}

    return {
        "needs_retraining": False,
        "reason": "Performance acceptable",
        "metrics": metrics,
    }


def prepare_feedback_dataset(**context):
    """Prepare dataset for retraining using FraudPostgresHook"""
    import pandas as pd

    ti = context["task_instance"]
    check_result = ti.xcom_pull(task_ids="check_retraining_needed")

    if not check_result["needs_retraining"]:
        return {"status": "skipped", "reason": "No retraining needed"}

    # Use hook instead of direct SQLAlchemy
    from plugins.hooks.postgres_hook import FraudPostgresHook

    hook = FraudPostgresHook()

    # Load transactions with feedback
    query = f"""
        SELECT 
            t.*,
            f.analyst_label,
            f.feedback_quality,
            cf.num_purchases_24h,
            cf.avg_transaction_amount,
            mf.merchant_risk_score
        FROM {TABLE_NAMES['TRANSACTIONS']} t
        JOIN {TABLE_NAMES['FEEDBACK_LABELS']} f ON t.transaction_id = f.transaction_id
        LEFT JOIN {TABLE_NAMES['CUSTOMER_FEATURES']} cf ON t.customer_id = cf.customer_id
        LEFT JOIN {TABLE_NAMES['MERCHANT_FEATURES']} mf ON t.merchant_id = mf.merchant_id
        WHERE f.created_at >= NOW() - INTERVAL '30 days'
    """

    results = hook.fetch_all(query)

    # Convert to DataFrame for parquet export
    columns = [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "amount",
        "timestamp",
        "is_fraud",
        "analyst_label",
        "feedback_quality",
        "num_purchases_24h",
        "avg_transaction_amount",
        "merchant_risk_score",
    ]
    df = pd.DataFrame(results, columns=columns)

    # Save for training
    output_path = f"/tmp/feedback_dataset_{context['ds_nodash']}.parquet"
    df.to_parquet(output_path, index=False)

    return {
        "status": "prepared",
        "dataset_path": output_path,
        "num_samples": len(df),
        "fraud_rate": float(df["analyst_label"].mean()),
    }


def generate_feedback_report(**context):
    """Generate feedback report"""
    ti = context["task_instance"]

    collection_result = ti.xcom_pull(task_ids="collect_analyst_labels")
    quality_metrics = ti.xcom_pull(task_ids="analyze_feedback_quality")
    retraining_check = ti.xcom_pull(task_ids="check_retraining_needed")

    report = f"""
    === Feedback Collection Report ===
    Date: {context['ds']}
    
    COLLECTION:
    - Labels collected: {collection_result['labels_collected']}
    
    QUALITY (Last 7 days):
    - Total feedback: {quality_metrics['total_feedback']}
    - Accuracy: {quality_metrics['accuracy']:.2%}
    - False Positives: {quality_metrics['false_positives']} ({quality_metrics['fp_rate']:.2%})
    - False Negatives: {quality_metrics['false_negatives']} ({quality_metrics['fn_rate']:.2%})
    - Avg Confidence: {quality_metrics['avg_confidence']:.3f}
    
    RETRAINING:
    - Needed: {retraining_check['needs_retraining']}
    - Reason: {retraining_check['reason']}
    """

    print(report)

    # Save to database using FraudPostgresHook
    from plugins.hooks.postgres_hook import FraudPostgresHook

    hook = FraudPostgresHook()

    query = f"""
        INSERT INTO {TABLE_NAMES['PIPELINE_EXECUTION_LOG']} 
        (pipeline_name, status, execution_time_seconds, records_processed, error_message)
        VALUES (%s, %s, %s, %s, %s)
    """

    hook.execute_query(
        query,
        (
            "03_feedback_collection",
            "SUCCESS",
            0,  # To be calculated
            collection_result["labels_collected"],
            None,
        ),
    )

    return {"report": report}


# Define DAG
with DAG(
    "03_feedback_collection",
    default_args=default_args,
    description="Collect and analyze analyst feedback",
    schedule_interval=SCHEDULES["FEEDBACK_COLLECTION"],  # Daily at 4 AM
    start_date=days_ago(1),
    catchup=False,
    tags=["feedback", "quality", "monitoring"],
) as dag:
    # Task 1: Collect labels
    collect = PythonOperator(
        task_id="collect_analyst_labels", python_callable=collect_analyst_labels
    )

    # Task 2: Analyze quality
    analyze = PythonOperator(
        task_id="analyze_feedback_quality", python_callable=analyze_feedback_quality
    )

    # Task 3: Check if retraining needed
    check_retrain = PythonOperator(
        task_id="check_retraining_needed", python_callable=check_retraining_needed
    )

    # Task 4: Prepare dataset
    prepare = PythonOperator(
        task_id="prepare_feedback_dataset", python_callable=prepare_feedback_dataset
    )

    # Task 5: Generate report
    report = PythonOperator(
        task_id="generate_feedback_report", python_callable=generate_feedback_report
    )

    # Dependencies
    collect >> analyze >> check_retrain >> prepare >> report
