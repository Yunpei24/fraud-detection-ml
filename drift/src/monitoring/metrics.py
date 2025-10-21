"""
Prometheus metrics for drift detection monitoring.

This module exposes drift detection metrics in Prometheus format.
"""

from prometheus_client import Gauge, Counter, Histogram, start_http_server
from typing import Dict, Any, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)

NAMESPACE = "drift_detection"

# -----------------------------------------------------------------------------
# Pipeline Health Metrics
# -----------------------------------------------------------------------------

PIPELINE_RUNS_TOTAL = Counter(
    f"{NAMESPACE}_pipeline_runs_total",
    "Total number of drift detection pipeline runs",
    ["pipeline_name", "status"]  # status: 'success' or 'failed'
)

PIPELINE_DURATION_SECONDS = Gauge(
    f"{NAMESPACE}_pipeline_duration_seconds",
    "Duration of the last drift detection run in seconds",
    ["pipeline_name"]
)

PIPELINE_LAST_SUCCESS_TIMESTAMP = Gauge(
    f"{NAMESPACE}_last_success_timestamp",
    "Timestamp of the last successful drift detection run",
    ["pipeline_name"]
)

# -----------------------------------------------------------------------------
# Drift Score Metrics
# -----------------------------------------------------------------------------

DATA_DRIFT_SCORE = Gauge(
    f"{NAMESPACE}_data_drift_score",
    "Drift score for a specific feature (e.g., PSI, KS-test)",
    ["feature_name", "metric_type"]  # metric_type: 'psi', 'ks_statistic', 'chi_square'
)

CONCEPT_DRIFT_SCORE = Gauge(
    f"{NAMESPACE}_concept_drift_score",
    "Model performance metric on recent data (indicates concept drift)",
    ["performance_metric"]  # performance_metric: 'recall', 'precision', 'f1', 'fpr'
)

TARGET_DRIFT_SCORE = Gauge(
    f"{NAMESPACE}_target_drift_score",
    "Drift score for the target variable distribution",
    ["target_variable"]
)

# Legacy metrics (kept for backward compatibility)
drift_score_gauge = Gauge(
    'drift_detection_score',
    'Current drift detection score',
    ['drift_type']
)

fraud_rate_gauge = Gauge(
    'fraud_rate_current',
    'Current fraud rate in production data'
)

model_recall_gauge = Gauge(
    'model_performance_recall',
    'Current model recall score'
)

model_precision_gauge = Gauge(
    'model_performance_precision',
    'Current model precision score'
)

model_fpr_gauge = Gauge(
    'model_performance_fpr',
    'Current model false positive rate'
)

model_f1_gauge = Gauge(
    'model_performance_f1',
    'Current model F1 score'
)

# -----------------------------------------------------------------------------
# Drift Status & Actions
# -----------------------------------------------------------------------------

DRIFT_DETECTED = Gauge(
    f"{NAMESPACE}_drift_detected",
    "Indicates if drift was detected for a feature (1=detected, 0=not detected)",
    ["feature_name", "drift_type"]  # drift_type: 'data', 'concept', 'target'
)

RETRAINING_TRIGGERED_TOTAL = Counter(
    f"{NAMESPACE}_retraining_triggered_total",
    "Total number of times model retraining was triggered due to drift"
)

ALERTS_SENT_TOTAL = Counter(
    f"{NAMESPACE}_alerts_sent_total",
    "Total number of drift alerts sent",
    ["severity"]  # severity: 'info', 'warning', 'critical'
)

# Legacy metrics (kept for backward compatibility)
alert_counter = Counter(
    'drift_alerts_total',
    'Total number of drift alerts triggered',
    ['alert_type', 'severity']
)

drift_detection_duration = Histogram(
    'drift_detection_duration_seconds',
    'Time spent detecting drift',
    ['drift_type']
)

predictions_processed = Counter(
    'drift_predictions_processed_total',
    'Total number of predictions processed for drift detection'
)


def setup_prometheus_metrics(port: int = 9091) -> None:
    """
    Start Prometheus metrics server for drift detection.
    
    Args:
        port: Port number for metrics server (default: 9091)
    """
    try:
        start_http_server(port)
        logger.info("prometheus_metrics_server_started", port=port, module="drift_detection")
    except Exception as e:
        logger.error("prometheus_server_start_failed", error=str(e), module="drift_detection")


def record_pipeline_success(pipeline_name: str, duration: float) -> None:
    """
    Record a successful drift detection pipeline run.
    
    Args:
        pipeline_name: Name of the pipeline
        duration: Duration in seconds
    """
    PIPELINE_RUNS_TOTAL.labels(pipeline_name=pipeline_name, status="success").inc()
    PIPELINE_DURATION_SECONDS.labels(pipeline_name=pipeline_name).set(duration)
    PIPELINE_LAST_SUCCESS_TIMESTAMP.labels(pipeline_name=pipeline_name).set(
        datetime.utcnow().timestamp()
    )
    logger.debug("drift_pipeline_success_recorded", pipeline_name=pipeline_name, duration=duration)


def record_pipeline_failure(pipeline_name: str) -> None:
    """
    Record a failed drift detection pipeline run.
    
    Args:
        pipeline_name: Name of the pipeline
    """
    PIPELINE_RUNS_TOTAL.labels(pipeline_name=pipeline_name, status="failed").inc()
    logger.debug("drift_pipeline_failure_recorded", pipeline_name=pipeline_name)


def update_drift_metrics(drift_results: Dict[str, Any]) -> None:
    """
    Update Prometheus metrics with drift detection results.
    
    Args:
        drift_results: Results from drift detection
    """
    try:
        # Data drift metrics
        if "data_drift" in drift_results:
            data_drift = drift_results["data_drift"]
            drift_score_gauge.labels(drift_type="data").set(
                data_drift.get("avg_psi", 0)
            )
        
        # Target drift metrics
        if "target_drift" in drift_results:
            target_drift = drift_results["target_drift"]
            fraud_rate_gauge.set(target_drift.get("current_fraud_rate", 0))
            drift_score_gauge.labels(drift_type="target").set(
                abs(target_drift.get("relative_change", 0))
            )
        
        # Concept drift metrics
        if "concept_drift" in drift_results:
            concept_drift = drift_results["concept_drift"]
            metrics = concept_drift.get("metrics", {})
            
            model_recall_gauge.set(metrics.get("recall", 0))
            model_precision_gauge.set(metrics.get("precision", 0))
            model_fpr_gauge.set(metrics.get("fpr", 0))
            model_f1_gauge.set(metrics.get("f1_score", 0))
            
            # Drift score based on performance degradation
            drift_score = max(
                abs(metrics.get("recall_change", 0)),
                abs(metrics.get("fpr_change", 0))
            )
            drift_score_gauge.labels(drift_type="concept").set(drift_score)
        
        logger.debug("drift_metrics_updated")
    
    except Exception as e:
        logger.error("failed_to_update_drift_metrics", error=str(e))


def record_alert(alert_type: str, severity: str) -> None:
    """
    Record an alert in Prometheus metrics.
    
    Args:
        alert_type: Type of alert
        severity: Severity level
    """
    alert_counter.labels(alert_type=alert_type, severity=severity).inc()
    logger.debug("alert_recorded", alert_type=alert_type, severity=severity)


def record_detection_duration(drift_type: str, duration: float) -> None:
    """
    Record drift detection duration.
    
    Args:
        drift_type: Type of drift detected
        duration: Duration in seconds
    """
    drift_detection_duration.labels(drift_type=drift_type).observe(duration)


def increment_predictions_processed(count: int = 1) -> None:
    """
    Increment counter for processed predictions.
    
    Args:
        count: Number of predictions processed
    """
    predictions_processed.inc(count)
