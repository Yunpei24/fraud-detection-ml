"""
Prometheus metrics for data pipeline monitoring.

This module exposes data pipeline metrics in Prometheus format.
"""

import logging
from datetime import datetime

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

NAMESPACE = "data_pipeline"

# -----------------------------------------------------------------------------
# Pipeline Health Metrics
# -----------------------------------------------------------------------------

PIPELINE_RUNS_TOTAL = Counter(
    f"{NAMESPACE}_runs_total",
    "Total number of pipeline runs",
    ["pipeline_name", "status"],  # status: 'success' or 'failed'
)

PIPELINE_DURATION_SECONDS = Gauge(
    f"{NAMESPACE}_duration_seconds",
    "Duration of the last pipeline run in seconds",
    ["pipeline_name"],
)

PIPELINE_LAST_SUCCESS_TIMESTAMP = Gauge(
    f"{NAMESPACE}_last_success_timestamp",
    "Timestamp of the last successful pipeline run",
    ["pipeline_name"],
)

# -----------------------------------------------------------------------------
# Data Volume Metrics
# -----------------------------------------------------------------------------

TRANSACTIONS_INGESTED_TOTAL = Counter(
    f"{NAMESPACE}_transactions_ingested_total",
    "Total number of transactions ingested from the source",
    ["source"],  # e.g., 'azure_sql', 'event_hub', 'file'
)

TRANSACTIONS_PROCESSED_TOTAL = Counter(
    f"{NAMESPACE}_transactions_processed_total",
    "Total number of transactions successfully processed",
    ["destination"],  # e.g., 'feature_store', 's3', 'database'
)

TRANSACTIONS_REJECTED_TOTAL = Counter(
    f"{NAMESPACE}_transactions_rejected_total",
    "Total number of transactions rejected during processing",
    ["reason"],  # e.g., 'validation_error', 'duplicate', 'missing_critical_field'
)

# -----------------------------------------------------------------------------
# Data Quality Metrics
# -----------------------------------------------------------------------------

VALIDATION_ERRORS_TOTAL = Counter(
    f"{NAMESPACE}_validation_errors_total",
    "Total number of data validation errors",
    [
        "feature_name",
        "error_type",
    ],  # error_type: 'missing', 'out_of_range', 'invalid_format'
)

MISSING_VALUES_TOTAL = Counter(
    f"{NAMESPACE}_missing_values_total",
    "Total number of missing values found",
    ["feature_name"],
)

# -----------------------------------------------------------------------------
# Performance Metrics
# -----------------------------------------------------------------------------

STEP_LATENCY_SECONDS = Histogram(
    f"{NAMESPACE}_step_latency_seconds",
    "Latency of individual pipeline steps in seconds",
    ["step_name"],  # e.g., 'ingestion', 'validation', 'feature_engineering', 'storage'
    buckets=[1, 5, 15, 30, 60, 120, 300, 600, 1200],  # 1s to 20min
)


def setup_prometheus_metrics(port: int = 9090) -> None:
    """
    Start Prometheus metrics server for data pipeline.

    Args:
        port: Port number for metrics server (default: 9090)
    """
    try:
        start_http_server(port)
        logger.info(
            f"Prometheus metrics server started on port {port} (module: data_pipeline)"
        )
    except Exception as e:
        logger.error(f"Prometheus server start failed: {e} (module: data_pipeline)")


def record_pipeline_success(pipeline_name: str, duration: float) -> None:
    """
    Record a successful pipeline run.

    Args:
        pipeline_name: Name of the pipeline
        duration: Duration in seconds
    """
    PIPELINE_RUNS_TOTAL.labels(pipeline_name=pipeline_name, status="success").inc()
    PIPELINE_DURATION_SECONDS.labels(pipeline_name=pipeline_name).set(duration)
    PIPELINE_LAST_SUCCESS_TIMESTAMP.labels(pipeline_name=pipeline_name).set(
        datetime.utcnow().timestamp()
    )
    logger.debug(f"Pipeline success recorded: {pipeline_name}, duration={duration}s")


def record_pipeline_failure(pipeline_name: str) -> None:
    """
    Record a failed pipeline run.

    Args:
        pipeline_name: Name of the pipeline
    """
    PIPELINE_RUNS_TOTAL.labels(pipeline_name=pipeline_name, status="failed").inc()
    logger.debug(f"Pipeline failure recorded: {pipeline_name}")
