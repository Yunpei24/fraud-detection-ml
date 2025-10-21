"""
Prometheus metrics for drift detection monitoring.

This module exposes drift detection metrics in Prometheus format.
"""

from prometheus_client import Gauge, Counter, Histogram, start_http_server
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

# Prometheus metrics
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
    Start Prometheus metrics server.
    
    Args:
        port: Port number for metrics server
    """
    try:
        start_http_server(port)
        logger.info("prometheus_metrics_server_started", port=port)
    except Exception as e:
        logger.error("prometheus_server_start_failed", error=str(e))


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
