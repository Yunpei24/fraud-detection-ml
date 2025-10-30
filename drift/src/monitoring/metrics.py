"""
Prometheus metrics for drift detection monitoring.

This module exposes drift detection metrics in Prometheus format.
Updated to align with API-based drift detection architecture.
"""

from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = structlog.get_logger(__name__)

NAMESPACE = "drift_detection"

# -----------------------------------------------------------------------------
# API Integration Metrics
# -----------------------------------------------------------------------------

API_REQUESTS_TOTAL = Counter(
    f"{NAMESPACE}_api_requests_total",
    "Total number of API requests made to drift detection service",
    ["endpoint", "method", "status"],  # status: 'success', 'error', 'timeout'
)

API_REQUEST_DURATION_SECONDS = Histogram(
    f"{NAMESPACE}_api_request_duration_seconds",
    "Duration of API requests to drift detection service",
    ["endpoint", "method"],
)

API_LAST_SUCCESS_TIMESTAMP = Gauge(
    f"{NAMESPACE}_api_last_success_timestamp",
    "Timestamp of the last successful API call",
    ["endpoint"],
)

# -----------------------------------------------------------------------------
# Pipeline Health Metrics
# -----------------------------------------------------------------------------

PIPELINE_RUNS_TOTAL = Counter(
    f"{NAMESPACE}_pipeline_runs_total",
    "Total number of drift detection pipeline runs",
    ["pipeline_name", "status"],  # status: 'success' or 'failed'
)

PIPELINE_DURATION_SECONDS = Gauge(
    f"{NAMESPACE}_pipeline_duration_seconds",
    "Duration of the last drift detection run in seconds",
    ["pipeline_name"],
)

PIPELINE_LAST_SUCCESS_TIMESTAMP = Gauge(
    f"{NAMESPACE}_last_success_timestamp",
    "Timestamp of the last successful drift detection run",
    ["pipeline_name"],
)

# -----------------------------------------------------------------------------
# Overall Drift Status Metrics
# -----------------------------------------------------------------------------

OVERALL_DRIFT_DETECTED = Gauge(
    f"{NAMESPACE}_overall_drift_detected",
    "Overall drift detection status from API (1=drift detected, 0=no drift)",
    ["severity"],  # severity: 'low', 'medium', 'high', 'critical'
)

DRIFT_SEVERITY_SCORE = Gauge(
    f"{NAMESPACE}_drift_severity_score",
    "Overall drift severity score from API (0-1 scale)",
)

DRIFT_TYPES_DETECTED = Gauge(
    f"{NAMESPACE}_drift_types_detected",
    "Number of different drift types detected",
    ["drift_type"],  # drift_type: 'data', 'target', 'concept', 'multivariate'
)

# -----------------------------------------------------------------------------
# Drift Score Metrics (API-based)
# -----------------------------------------------------------------------------

DATA_DRIFT_SCORE = Gauge(
    f"{NAMESPACE}_data_drift_score",
    "Data drift score from API (PSI or other metric)",
    ["feature_name"],
)

TARGET_DRIFT_SCORE = Gauge(
    f"{NAMESPACE}_target_drift_score",
    "Target drift score from API",
    ["metric"],  # metric: 'fraud_rate', 'relative_change', etc.
)

CONCEPT_DRIFT_SCORE = Gauge(
    f"{NAMESPACE}_concept_drift_score",
    "Concept drift score from API",
    ["metric"],  # metric: 'performance_degradation', 'recall_change', etc.
)

MULTIVARIATE_DRIFT_SCORE = Gauge(
    f"{NAMESPACE}_multivariate_drift_score", "Multivariate drift score from API"
)

# -----------------------------------------------------------------------------
# Operational Metrics
# -----------------------------------------------------------------------------

ALERTS_TRIGGERED_TOTAL = Counter(
    f"{NAMESPACE}_alerts_triggered_total",
    "Total number of drift alerts triggered",
    ["alert_type", "severity"],  # alert_type: 'drift_detected', 'data_drift', etc.
)

RETRAINING_TRIGGERED_TOTAL = Counter(
    f"{NAMESPACE}_retraining_triggered_total",
    "Total number of times model retraining was triggered due to drift",
    ["trigger_reason", "priority"],  # priority: 'low', 'medium', 'high', 'critical'
)

THRESHOLDS_EXCEEDED_TOTAL = Counter(
    f"{NAMESPACE}_thresholds_exceeded_total",
    "Total number of times drift thresholds were exceeded",
    ["threshold_type"],
)

# Legacy metrics (kept for backward compatibility)
drift_score_gauge = Gauge(
    "drift_detection_score", "Current drift detection score", ["drift_type"]
)

fraud_rate_gauge = Gauge("fraud_rate_current", "Current fraud rate in production data")

model_recall_gauge = Gauge("model_performance_recall", "Current model recall score")

model_precision_gauge = Gauge(
    "model_performance_precision", "Current model precision score"
)

model_fpr_gauge = Gauge("model_performance_fpr", "Current model false positive rate")

model_f1_gauge = Gauge("model_performance_f1", "Current model F1 score")

alert_counter = Counter(
    "drift_alerts_total",
    "Total number of drift alerts triggered",
    ["alert_type", "severity"],
)

drift_detection_duration = Histogram(
    "drift_detection_duration_seconds", "Time spent detecting drift", ["drift_type"]
)

predictions_processed = Counter(
    "drift_predictions_processed_total",
    "Total number of predictions processed for drift detection",
)


def setup_prometheus_metrics(port: int = 9091) -> None:
    """
    Start Prometheus metrics server for drift detection.

    Args:
        port: Port number for metrics server (default: 9091)
    """
    try:
        start_http_server(port)
        logger.info(
            "prometheus_metrics_server_started", port=port, module="drift_detection"
        )
    except Exception as e:
        logger.error(
            "prometheus_server_start_failed", error=str(e), module="drift_detection"
        )


def record_api_request(
    endpoint: str, method: str, status: str, duration: float
) -> None:
    """
    Record an API request to the drift detection service.

    Args:
        endpoint: API endpoint called
        method: HTTP method used
        status: Request status ('success', 'error', 'timeout')
        duration: Request duration in seconds
    """
    API_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status=status).inc()
    API_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint, method=method).observe(
        duration
    )

    if status == "success":
        API_LAST_SUCCESS_TIMESTAMP.labels(endpoint=endpoint).set(
            datetime.utcnow().timestamp()
        )

    logger.debug(
        "api_request_recorded",
        endpoint=endpoint,
        method=method,
        status=status,
        duration=duration,
    )


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
    logger.debug(
        "drift_pipeline_success_recorded",
        pipeline_name=pipeline_name,
        duration=duration,
    )


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
    Update Prometheus metrics with API drift detection results.

    Args:
        drift_results: Results from API drift detection
    """
    try:
        # Extract overall drift status from API response
        drift_summary = drift_results.get("drift_summary", {})
        overall_drift_detected = drift_summary.get("overall_drift_detected", False)
        severity_score = drift_summary.get("severity_score", 0)
        drift_types_detected = drift_summary.get("drift_types_detected", [])

        # Update overall drift metrics
        severity_label = _get_severity_label(severity_score)
        OVERALL_DRIFT_DETECTED.labels(severity=severity_label).set(
            1 if overall_drift_detected else 0
        )
        DRIFT_SEVERITY_SCORE.set(severity_score)

        # Update drift types detected
        for drift_type in ["data", "target", "concept", "multivariate"]:
            detected = drift_type in drift_types_detected
            DRIFT_TYPES_DETECTED.labels(drift_type=drift_type).set(1 if detected else 0)

        # Update specific drift scores
        _update_data_drift_metrics(drift_results.get("data_drift", {}))
        _update_target_drift_metrics(drift_results.get("target_drift", {}))
        _update_concept_drift_metrics(drift_results.get("concept_drift", {}))
        _update_multivariate_drift_metrics(drift_results.get("multivariate_drift", {}))

        # Legacy metrics for backward compatibility
        _update_legacy_metrics(drift_results)

        logger.debug(
            "drift_metrics_updated",
            overall_drift=overall_drift_detected,
            severity=severity_score,
        )

    except Exception as e:
        logger.error("failed_to_update_drift_metrics", error=str(e))


def _get_severity_label(score: float) -> str:
    """Convert severity score to label."""
    if score >= 4:
        return "critical"
    elif score >= 3:
        return "high"
    elif score >= 2:
        return "medium"
    else:
        return "low"


def _update_data_drift_metrics(data_drift: Dict[str, Any]) -> None:
    """Update data drift specific metrics."""
    if not data_drift:
        return

    # Use PSI score or average PSI if available
    psi_score = data_drift.get("psi_score") or data_drift.get("avg_psi", 0)
    if psi_score > 0:
        DATA_DRIFT_SCORE.labels(feature_name="overall").set(psi_score)

    # Feature-specific PSI scores
    feature_scores = data_drift.get("feature_psi_scores", {})
    for feature, score in feature_scores.items():
        DATA_DRIFT_SCORE.labels(feature_name=feature).set(score)


def _update_target_drift_metrics(target_drift: Dict[str, Any]) -> None:
    """Update target drift specific metrics."""
    if not target_drift:
        return

    # Fraud rate
    fraud_rate = target_drift.get("current_fraud_rate", 0)
    if fraud_rate > 0:
        TARGET_DRIFT_SCORE.labels(metric="fraud_rate").set(fraud_rate)

    # Relative change
    relative_change = abs(target_drift.get("relative_change", 0))
    TARGET_DRIFT_SCORE.labels(metric="relative_change").set(relative_change)


def _update_concept_drift_metrics(concept_drift: Dict[str, Any]) -> None:
    """Update concept drift specific metrics."""
    if not concept_drift:
        return

    metrics = concept_drift.get("metrics", {})

    # Performance metrics
    for metric_name in ["recall", "precision", "fpr", "f1_score"]:
        value = metrics.get(metric_name, 0)
        if value > 0:
            CONCEPT_DRIFT_SCORE.labels(metric=metric_name).set(value)

    # Performance changes
    for change_metric in ["recall_change", "fpr_change"]:
        change_value = abs(metrics.get(change_metric, 0))
        CONCEPT_DRIFT_SCORE.labels(metric=f"{change_metric}_abs").set(change_value)


def _update_multivariate_drift_metrics(multivariate_drift: Dict[str, Any]) -> None:
    """Update multivariate drift specific metrics."""
    if not multivariate_drift:
        return

    drift_score = multivariate_drift.get("drift_score", 0)
    MULTIVARIATE_DRIFT_SCORE.set(drift_score)


def _update_legacy_metrics(drift_results: Dict[str, Any]) -> None:
    """Update legacy metrics for backward compatibility."""
    # Data drift
    if "data_drift" in drift_results:
        data_drift = drift_results["data_drift"]
        psi_score = data_drift.get("psi_score") or data_drift.get("avg_psi", 0)
        drift_score_gauge.labels(drift_type="data").set(psi_score)

    # Target drift
    if "target_drift" in drift_results:
        target_drift = drift_results["target_drift"]
        fraud_rate = target_drift.get("current_fraud_rate", 0)
        if fraud_rate > 0:
            fraud_rate_gauge.set(fraud_rate)

        relative_change = abs(target_drift.get("relative_change", 0))
        drift_score_gauge.labels(drift_type="target").set(relative_change)

    # Concept drift
    if "concept_drift" in drift_results:
        concept_drift = drift_results["concept_drift"]
        metrics = concept_drift.get("metrics", {})

        model_recall_gauge.set(metrics.get("recall", 0))
        model_precision_gauge.set(metrics.get("precision", 0))
        model_fpr_gauge.set(metrics.get("fpr", 0))
        model_f1_gauge.set(metrics.get("f1_score", 0))

        drift_score = max(
            abs(metrics.get("recall_change", 0)), abs(metrics.get("fpr_change", 0))
        )
        drift_score_gauge.labels(drift_type="concept").set(drift_score)

    # Multivariate drift
    if "multivariate_drift" in drift_results:
        multivariate_drift = drift_results["multivariate_drift"]
        multivariate_score = multivariate_drift.get("drift_score", 0)
        drift_score_gauge.labels(drift_type="multivariate").set(multivariate_score)


def record_alert(alert_type: str, severity: str) -> None:
    """
    Record an alert in Prometheus metrics.

    Args:
        alert_type: Type of alert
        severity: Severity level
    """
    ALERTS_TRIGGERED_TOTAL.labels(alert_type=alert_type, severity=severity).inc()
    alert_counter.labels(alert_type=alert_type, severity=severity).inc()
    logger.debug("alert_recorded", alert_type=alert_type, severity=severity)


def record_retraining_trigger(reason: str, priority: str) -> None:
    """
    Record a retraining trigger.

    Args:
        reason: Reason for retraining
        priority: Priority level
    """
    RETRAINING_TRIGGERED_TOTAL.labels(trigger_reason=reason, priority=priority).inc()
    logger.debug("retraining_trigger_recorded", reason=reason, priority=priority)


def record_threshold_exceeded(threshold_type: str) -> None:
    """
    Record when a drift threshold was exceeded.

    Args:
        threshold_type: Type of threshold exceeded
    """
    THRESHOLDS_EXCEEDED_TOTAL.labels(threshold_type=threshold_type).inc()
    logger.debug("threshold_exceeded_recorded", threshold_type=threshold_type)


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
