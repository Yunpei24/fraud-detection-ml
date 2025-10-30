"""
DAG 04: Data Quality Monitoring
Monitors data quality in production
Uses PostgresHook for validation

Schedule: Daily at 2 AM
Refactored to remove obsolete settings/module_loader imports
"""
import os
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from airflow import DAG
# Import centralized configuration
from config.constants import ALERT_CONFIG, SCHEDULES, TABLE_NAMES, THRESHOLDS

# Configuration from environment
ALERT_EMAIL = os.getenv("ALERT_EMAIL", ALERT_CONFIG["DEFAULT_EMAIL"])
MAX_DATA_AGE_HOURS = float(os.getenv("MAX_DATA_AGE_HOURS", "2"))
MIN_PREDICTION_COVERAGE = float(os.getenv("MIN_PREDICTION_COVERAGE", "95"))

default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "email": [ALERT_EMAIL],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_data_freshness(**context):
    """Check data freshness via PostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook

    hook = FraudPostgresHook()

    # Check last transaction
    query = f"""
        SELECT 
            MAX(created_at) as last_transaction,
            COUNT(*) as total_transactions_24h
        FROM {TABLE_NAMES['TRANSACTIONS']}
        WHERE created_at >= NOW() - INTERVAL '24 hours'
    """

    result = hook.fetch_one(query)

    last_transaction = result[0]
    total_24h = result[1]

    # Alert if no data since max_data_age_hours
    if last_transaction:
        hours_ago = (datetime.now() - last_transaction).total_seconds() / 3600
        is_fresh = hours_ago < MAX_DATA_AGE_HOURS
    else:
        hours_ago = None
        is_fresh = False

    return {
        "last_transaction": last_transaction.isoformat() if last_transaction else None,
        "hours_since_last": hours_ago,
        "total_24h": total_24h,
        "is_fresh": is_fresh,
        "status": "OK" if is_fresh else "STALE",
    }


def check_missing_values(**context):
    """Check missing values using FraudPostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook

    hook = FraudPostgresHook()

    # Check NULL values in critical columns
    query = f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) as null_amount,
            SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) as null_customer,
            SUM(CASE WHEN merchant_id IS NULL THEN 1 ELSE 0 END) as null_merchant
        FROM {TABLE_NAMES['TRANSACTIONS']}
        WHERE created_at >= NOW() - INTERVAL '24 hours'
    """

    result = hook.fetch_one(query)

    total = result[0] or 1

    return {
        "total_transactions": result[0],
        "null_amount": result[1],
        "null_customer": result[2],
        "null_merchant": result[3],
        "null_amount_pct": (result[1] or 0) / total * 100,
        "null_customer_pct": (result[2] or 0) / total * 100,
        "null_merchant_pct": (result[3] or 0) / total * 100,
        "status": "OK" if (result[1] or 0) == 0 else "WARNING",
    }


def check_data_ranges(**context):
    """Check value ranges using FraudPostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook

    hook = FraudPostgresHook()

    # Check outliers in amounts
    query = f"""
        SELECT 
            MIN(amount) as min_amount,
            MAX(amount) as max_amount,
            AVG(amount) as avg_amount,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount) as p99_amount,
            COUNT(*) as total
        FROM {TABLE_NAMES['TRANSACTIONS']}
        WHERE created_at >= NOW() - INTERVAL '24 hours'
    """

    result = hook.fetch_one(query)

    issues = []
    if result[4] > 0:  # negative amounts
        issues.append(f"Found {result[4]} negative amounts")
    if result[1] > 1000000:  # unrealistic max
        issues.append(f"Unrealistic max amount: {result[1]}")

    return {
        "min_amount": float(result[0] or 0),
        "max_amount": float(result[1] or 0),
        "avg_amount": float(result[2] or 0),
        "stddev_amount": float(result[3] or 0),
        "negative_amounts": result[4],
        "very_high_amounts": result[5],
        "issues": issues,
        "status": "OK" if len(issues) == 0 else "WARNING",
    }


def check_duplicates(**context):
    """Check duplicates using FraudPostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook

    hook = FraudPostgresHook()

    query = f"""
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT transaction_id) as unique_ids,
            COUNT(*) - COUNT(DISTINCT transaction_id) as duplicates
        FROM {TABLE_NAMES['TRANSACTIONS']}
        WHERE created_at >= NOW() - INTERVAL '24 hours'
    """

    result = hook.fetch_one(query)

    return {
        "total_transactions": result[0],
        "unique_ids": result[1],
        "duplicates": result[2],
        "duplicate_rate": (result[2] / (result[0] or 1)) * 100,
        "status": "OK" if result[2] == 0 else "WARNING",
    }


def check_prediction_coverage(**context):
    """Check that all transactions have predictions using FraudPostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook

    hook = FraudPostgresHook()

    query = f"""
        SELECT 
            COUNT(DISTINCT t.transaction_id) as total_transactions,
            COUNT(DISTINCT p.transaction_id) as transactions_with_predictions,
            COUNT(DISTINCT t.transaction_id) - COUNT(DISTINCT p.transaction_id) as missing_predictions
        FROM {TABLE_NAMES['TRANSACTIONS']} t
        LEFT JOIN {TABLE_NAMES['PREDICTIONS']} p ON t.transaction_id = p.transaction_id
        WHERE t.created_at >= NOW() - INTERVAL '24 hours'
    """

    result = hook.fetch_one(query)

    coverage = (result[1] / (result[0] or 1)) * 100

    # Use coverage threshold from environment
    min_coverage = MIN_PREDICTION_COVERAGE

    return {
        "total_transactions": result[0],
        "with_predictions": result[1],
        "missing_predictions": result[2],
        "coverage_pct": coverage,
        "status": "OK" if coverage >= min_coverage else "CRITICAL",
    }


def save_quality_metrics(**context):
    """Save quality metrics using FraudPostgresHook"""
    from plugins.hooks.postgres_hook import FraudPostgresHook

    ti = context["task_instance"]

    freshness = ti.xcom_pull(task_ids="check_data_freshness")
    missing = ti.xcom_pull(task_ids="check_missing_values")
    ranges = ti.xcom_pull(task_ids="check_data_ranges")
    duplicates = ti.xcom_pull(task_ids="check_duplicates")
    coverage = ti.xcom_pull(task_ids="check_prediction_coverage")

    # Use hook instead of direct SQLAlchemy
    hook = FraudPostgresHook()

    # Save to data_quality_log
    metrics = [
        (
            "data_freshness",
            "hours_since_last",
            freshness["hours_since_last"] or 999,
            freshness["status"],
        ),
        (
            "missing_values",
            "null_amount_pct",
            missing["null_amount_pct"],
            missing["status"],
        ),
        (
            "data_ranges",
            "negative_amounts",
            ranges["negative_amounts"],
            ranges["status"],
        ),
        (
            "duplicates",
            "duplicate_rate",
            duplicates["duplicate_rate"],
            duplicates["status"],
        ),
        (
            "prediction_coverage",
            "coverage_pct",
            coverage["coverage_pct"],
            coverage["status"],
        ),
    ]

    query = """
        INSERT INTO data_quality_log 
        (check_type, metric_name, metric_value, status)
        VALUES (%s, %s, %s, %s)
    """

    for metric in metrics:
        hook.execute_query(query, metric)

    # Déterminer statut global
    statuses = [
        freshness["status"],
        missing["status"],
        ranges["status"],
        duplicates["status"],
        coverage["status"],
    ]

    if "CRITICAL" in statuses:
        global_status = "CRITICAL"
    elif "WARNING" in statuses:
        global_status = "WARNING"
    else:
        global_status = "OK"

    return {
        "global_status": global_status,
        "checks_passed": statuses.count("OK"),
        "checks_total": len(statuses),
        "timestamp": datetime.now().isoformat(),
    }


def send_quality_alert(**context):
    """Envoie alerte si problèmes qualité"""
    ti = context["task_instance"]
    summary = ti.xcom_pull(task_ids="save_quality_metrics")

    if summary["global_status"] == "OK":
        return {"status": "no_alert_needed"}

    freshness = ti.xcom_pull(task_ids="check_data_freshness")
    coverage = ti.xcom_pull(task_ids="check_prediction_coverage")

    from plugins.operators.alert_operator import FraudDetectionAlertOperator

    alert_op = FraudDetectionAlertOperator(
        task_id="quality_alert_internal",
        alert_type="data_quality",
        severity=summary["global_status"],
        message=f"Data quality issues detected: {summary['checks_passed']}/{summary['checks_total']} checks passed",
        details={
            "freshness": freshness,
            "coverage": coverage,
            "global_status": summary["global_status"],
        },
    )

    return alert_op.execute(context)


# Define DAG
with DAG(
    "04_data_quality",
    default_args=default_args,
    description="Data quality monitoring and validation",
    schedule_interval=SCHEDULES["DATA_QUALITY"],  # Daily at 2 AM
    start_date=days_ago(1),
    catchup=False,
    tags=["quality", "monitoring", "data"],
) as dag:
    freshness = PythonOperator(
        task_id="check_data_freshness", python_callable=check_data_freshness
    )

    missing = PythonOperator(
        task_id="check_missing_values", python_callable=check_missing_values
    )

    ranges = PythonOperator(
        task_id="check_data_ranges", python_callable=check_data_ranges
    )

    dups = PythonOperator(task_id="check_duplicates", python_callable=check_duplicates)

    coverage = PythonOperator(
        task_id="check_prediction_coverage", python_callable=check_prediction_coverage
    )

    save = PythonOperator(
        task_id="save_quality_metrics", python_callable=save_quality_metrics
    )

    alert = PythonOperator(
        task_id="send_quality_alert", python_callable=send_quality_alert
    )

    # All checks in parallel, then save and alert
    [freshness, missing, ranges, dups, coverage] >> save >> alert
