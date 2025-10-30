"""
DAG 06: Model Performance Tracking
Daily tracking of model performance in production
Uses PostgresHook for metrics calculation

Schedule: Daily at 3 AM
Refactored to remove obsolete settings/helpers imports
"""

import os
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from plugins.hooks.postgres_hook import FraudPostgresHook

from airflow import DAG
# Import centralized configuration
from config.constants import ALERT_CONFIG, SCHEDULES, TABLE_NAMES, THRESHOLDS

# Configuration from environment
ALERT_EMAIL = os.getenv("ALERT_EMAIL", ALERT_CONFIG["DEFAULT_EMAIL"])
MIN_RECALL_THRESHOLD = THRESHOLDS["MIN_RECALL"]
MIN_PRECISION_THRESHOLD = THRESHOLDS["MIN_PRECISION"]
PERFORMANCE_DEGRADATION_THRESHOLD = THRESHOLDS["PERFORMANCE_DEGRADATION_THRESHOLD"]


# Helper functions (inline replacements for obsolete helpers.py)
def calculate_percentage_change(baseline: float, current: float) -> float:
    """Calculate percentage change between baseline and current value"""
    if baseline == 0:
        return 0.0
    return ((current - baseline) / baseline) * 100


def format_metric_value(value: float) -> str:
    """Format metric value as percentage"""
    return f"{value:.2%}"


default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": [ALERT_EMAIL],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def compute_daily_metrics(**context):
    """Calculate model metrics for last 24h using FraudPostgresHook"""
    import pandas as pd
    from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    # Use hook instead of direct SQLAlchemy
    hook = FraudPostgresHook()

    # Retrieve predictions vs actuals last 24h
    query = f"""
        SELECT 
            p.prediction,
            p.probability,
            t.is_fraud as actual,
            p.model_version,
            p.created_at
        FROM {TABLE_NAMES['PREDICTIONS']} p
        JOIN {TABLE_NAMES['TRANSACTIONS']} t ON p.transaction_id = t.transaction_id
        WHERE p.created_at >= NOW() - INTERVAL '24 hours'
        AND t.analyst_reviewed = true
    """

    # Fetch all rows into DataFrame
    results = hook.fetch_all(query)

    if not results:
        return {"status": "no_data", "message": "No reviewed transactions in last 24h"}

    # Convert to DataFrame for metrics calculation
    df = pd.DataFrame(
        results,
        columns=["prediction", "probability", "actual", "model_version", "created_at"],
    )

    y_true = df["actual"]
    y_pred = df["prediction"]
    y_prob = df["probability"]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "date": context["ds"],
        "total_predictions": len(df),
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
    }

    return metrics


def compare_to_baseline(**context):
    """Compare current metrics to baseline using FraudPostgresHook"""
    ti = context["task_instance"]
    current_metrics = ti.xcom_pull(task_ids="compute_daily_metrics")

    if current_metrics.get("status") == "no_data":
        return {"comparison": "skipped", "reason": "no_data"}

    # Use hook instead of direct SQLAlchemy
    hook = FraudPostgresHook()

    # Retrieve baseline (average last 30 days)
    query = f"""
        SELECT 
            AVG(recall) as baseline_recall,
            AVG(precision) as baseline_precision,
            AVG(f1_score) as baseline_f1,
            AVG(auc_roc) as baseline_auc
        FROM {TABLE_NAMES['MODEL_VERSIONS']}
        WHERE is_production = true
        AND registered_at >= NOW() - INTERVAL '30 days'
    """

    result = hook.fetch_one(query)

    # Use thresholds from environment variables
    baseline = {
        "recall": float(result[0] or MIN_RECALL_THRESHOLD),
        "precision": float(result[1] or MIN_PRECISION_THRESHOLD),
        "f1_score": float(result[2] or 0.77),
        "auc_roc": float(result[3] or 0.85),
    }

    # Use helper function to calculate percentage changes
    recall_pct_change = calculate_percentage_change(
        baseline["recall"], current_metrics["recall"]
    )
    precision_pct_change = calculate_percentage_change(
        baseline["precision"], current_metrics["precision"]
    )
    f1_pct_change = calculate_percentage_change(
        baseline["f1_score"], current_metrics["f1_score"]
    )
    auc_pct_change = calculate_percentage_change(
        baseline["auc_roc"], current_metrics["auc_roc"]
    )

    comparison = {
        "recall_delta": current_metrics["recall"] - baseline["recall"],
        "precision_delta": current_metrics["precision"] - baseline["precision"],
        "f1_delta": current_metrics["f1_score"] - baseline["f1_score"],
        "auc_delta": current_metrics["auc_roc"] - baseline["auc_roc"],
        "recall_pct_change": recall_pct_change,
        "precision_pct_change": precision_pct_change,
        "f1_pct_change": f1_pct_change,
        "auc_pct_change": auc_pct_change,
        "baseline": baseline,
        "current": {
            "recall": current_metrics["recall"],
            "precision": current_metrics["precision"],
            "f1_score": current_metrics["f1_score"],
            "auc_roc": current_metrics["auc_roc"],
        },
    }

    # Use degradation threshold from environment
    degradation_threshold = PERFORMANCE_DEGRADATION_THRESHOLD

    # Déterminer si dégradation significative
    degraded = (
        comparison["recall_delta"] < -degradation_threshold
        or comparison["f1_delta"] < -degradation_threshold
        or comparison["auc_delta"] < -degradation_threshold
    )

    comparison["is_degraded"] = degraded
    comparison["severity"] = "CRITICAL" if degraded else "OK"

    return comparison


def check_performance_thresholds(**context):
    """Vérifie seuils de performance using centralized thresholds and helpers"""
    ti = context["task_instance"]
    metrics = ti.xcom_pull(task_ids="compute_daily_metrics")

    if metrics.get("status") == "no_data":
        return {"status": "skipped"}

    # Use thresholds from environment variables
    min_recall = MIN_RECALL_THRESHOLD
    min_precision = MIN_PRECISION_THRESHOLD
    max_fpr = 0.05  # Max false positive rate

    violations = []

    if metrics["recall"] < min_recall:
        violations.append(
            {
                "metric": "recall",
                "value": metrics["recall"],
                "threshold": min_recall,
                "message": f"Recall {format_metric_value(metrics['recall'])} < {min_recall}",
            }
        )

    if metrics["precision"] < min_precision:
        violations.append(
            {
                "metric": "precision",
                "value": metrics["precision"],
                "threshold": min_precision,
                "message": f"Precision {format_metric_value(metrics['precision'])} < {min_precision}",
            }
        )

    if metrics["false_positive_rate"] > max_fpr:
        violations.append(
            {
                "metric": "false_positive_rate",
                "value": metrics["false_positive_rate"],
                "threshold": max_fpr,
                "message": f"FPR {format_metric_value(metrics['false_positive_rate'])} > {max_fpr}",
            }
        )

    return {
        "violations": violations,
        "has_violations": len(violations) > 0,
        "severity": "CRITICAL" if len(violations) > 0 else "OK",
    }


def save_performance_metrics(**context):
    """Sauvegarde métriques dans database using FraudPostgresHook"""
    import json

    ti = context["task_instance"]
    metrics = ti.xcom_pull(task_ids="compute_daily_metrics")

    if metrics.get("status") == "no_data":
        return {"status": "skipped"}

    # Use hook instead of direct SQLAlchemy
    hook = FraudPostgresHook()

    # Sauvegarder dans pipeline_execution_log
    query = """
        INSERT INTO pipeline_execution_log 
        (pipeline_name, status, execution_time_seconds, records_processed, metrics)
        VALUES (%s, %s, %s, %s, %s::jsonb)
    """

    metrics_json = json.dumps(
        {
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "f1_score": metrics["f1_score"],
            "auc_roc": metrics["auc_roc"],
            "fpr": metrics["false_positive_rate"],
            "fnr": metrics["false_negative_rate"],
        }
    )

    hook.execute_query(
        query,
        (
            "06_model_performance_tracking",
            "SUCCESS",
            0,
            metrics["total_predictions"],
            metrics_json,
        ),
    )

    return {"status": "saved", "metrics_count": 6}


def send_performance_alert(**context):
    """Envoie alerte si performance dégradée using helper and AlertOperator"""
    ti = context["task_instance"]

    comparison = ti.xcom_pull(task_ids="compare_to_baseline")
    threshold_check = ti.xcom_pull(task_ids="check_performance_thresholds")

    # Skip si pas de données
    if comparison.get("comparison") == "skipped":
        return {"status": "no_alert"}

    # Alert si dégradation ou violations
    needs_alert = comparison.get("is_degraded", False) or threshold_check.get(
        "has_violations", False
    )

    if not needs_alert:
        return {"status": "no_alert_needed"}

    from plugins.operators.alert_operator import FraudDetectionAlertOperator

    # Create alert message
    violations = threshold_check.get("violations", [])
    violations_msg = "\n".join([f"  - {v['message']}" for v in violations])

    alert_message = f"""
    Model Performance Alert
    =======================
    Type: model_performance
    Summary: Model performance degradation detected ({len(violations)} violations)
    
    Details:
    {violations_msg}
    """

    alert_op = FraudDetectionAlertOperator(
        task_id="performance_alert_internal",
        alert_type="model_performance",
        severity="CRITICAL",
        message=alert_message,
        details={
            "comparison": comparison,
            "violations": violations,
            "baseline": comparison.get("baseline"),
            "current": comparison.get("current"),
        },
    )

    return alert_op.execute(context)


def generate_performance_report(**context):
    """Génère rapport de performance"""
    ti = context["task_instance"]

    metrics = ti.xcom_pull(task_ids="compute_daily_metrics")
    comparison = ti.xcom_pull(task_ids="compare_to_baseline")

    if metrics.get("status") == "no_data":
        return {"report": "No data available for report"}

    report = f"""
    === Model Performance Report ===
    Date: {context['ds']}
    
    CURRENT METRICS (24h):
    - Total Predictions: {metrics['total_predictions']}
    - Recall: {metrics['recall']:.3f}
    - Precision: {metrics['precision']:.3f}
    - F1-Score: {metrics['f1_score']:.3f}
    - AUC-ROC: {metrics['auc_roc']:.3f}
    - False Positive Rate: {metrics['false_positive_rate']:.3f}
    - False Negative Rate: {metrics['false_negative_rate']:.3f}
    
    CONFUSION MATRIX:
    - True Positives: {metrics['true_positives']}
    - False Positives: {metrics['false_positives']}
    - True Negatives: {metrics['true_negatives']}
    - False Negatives: {metrics['false_negatives']}
    
    COMPARISON TO BASELINE (30 days):
    - Recall Δ: {comparison.get('recall_delta', 0):.3f}
    - Precision Δ: {comparison.get('precision_delta', 0):.3f}
    - F1 Δ: {comparison.get('f1_delta', 0):.3f}
    - AUC Δ: {comparison.get('auc_delta', 0):.3f}
    
    STATUS: {comparison.get('severity', 'OK')}
    """

    print(report)
    return {"report": report, "status": "generated"}


# Define DAG
with DAG(
    "06_model_performance_tracking",
    default_args=default_args,
    description="Daily model performance tracking and alerting",
    schedule_interval=SCHEDULES["PERFORMANCE_TRACKING"],  # Daily at 3 AM
    start_date=days_ago(1),
    catchup=False,
    tags=["monitoring", "performance", "ml"],
) as dag:
    compute = PythonOperator(
        task_id="compute_daily_metrics", python_callable=compute_daily_metrics
    )

    compare = PythonOperator(
        task_id="compare_to_baseline", python_callable=compare_to_baseline
    )

    check = PythonOperator(
        task_id="check_performance_thresholds",
        python_callable=check_performance_thresholds,
    )

    save = PythonOperator(
        task_id="save_performance_metrics", python_callable=save_performance_metrics
    )

    alert = PythonOperator(
        task_id="send_performance_alert", python_callable=send_performance_alert
    )

    report = PythonOperator(
        task_id="generate_performance_report",
        python_callable=generate_performance_report,
    )

    # Dependencies
    compute >> [compare, check]
    [compare, check] >> save
    save >> alert >> report
