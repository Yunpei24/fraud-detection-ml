#!/usr/bin/env python3
"""
Standalone Prometheus metrics server for training monitoring.

This script starts an HTTP server on port 9093 to expose metrics to Prometheus.
The server runs continuously until manually stopped.

Metrics exposed:
- Training duration per model
- Model accuracy, precision, recall
- Hyperparameter tuning iterations
- Data loading time
- Feature engineering time
- Memory usage during training

Usage:
    python metrics_server.py

The metrics will be available at: http://localhost:9093/metrics
"""

import os
import signal
import sys
import time

import structlog
from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

logger = structlog.get_logger(__name__)

# Global flag for graceful shutdown
running = True

# ==============================================================================
# PROMETHEUS METRICS DEFINITIONS
# ==============================================================================

# Training metrics
TRAINING_DURATION = Histogram(
    "fraud_training_duration_seconds",
    "Time spent training a model",
    ["model_name", "stage"],
)

TRAINING_RUNS_TOTAL = Counter(
    "fraud_training_runs_total",
    "Total number of training runs",
    ["model_name", "status"],
)

# Model performance metrics
MODEL_ACCURACY = Gauge(
    "fraud_model_accuracy",
    "Model accuracy on test set",
    ["model_name", "model_version"],
)

MODEL_PRECISION = Gauge(
    "fraud_model_precision",
    "Model precision on test set",
    ["model_name", "model_version"],
)

MODEL_RECALL = Gauge(
    "fraud_model_recall", "Model recall on test set", ["model_name", "model_version"]
)

MODEL_F1_SCORE = Gauge(
    "fraud_model_f1_score",
    "Model F1 score on test set",
    ["model_name", "model_version"],
)

MODEL_AUC_ROC = Gauge(
    "fraud_model_auc_roc", "Model AUC-ROC score", ["model_name", "model_version"]
)

# Hyperparameter tuning metrics
HYPERPARAMETER_TRIALS_TOTAL = Counter(
    "fraud_hyperparameter_trials_total",
    "Total number of hyperparameter tuning trials",
    ["model_name"],
)

HYPERPARAMETER_BEST_SCORE = Gauge(
    "fraud_hyperparameter_best_score",
    "Best score achieved during hyperparameter tuning",
    ["model_name", "metric"],
)

# Data processing metrics
DATA_LOADING_DURATION = Histogram(
    "fraud_data_loading_duration_seconds", "Time spent loading data", ["data_source"]
)

FEATURE_ENGINEERING_DURATION = Histogram(
    "fraud_feature_engineering_duration_seconds",
    "Time spent on feature engineering",
    ["stage"],
)

DATA_SAMPLES_PROCESSED = Counter(
    "fraud_data_samples_processed_total",
    "Total number of data samples processed",
    ["split"],
)

# System metrics
TRAINING_MEMORY_USAGE_MB = Gauge(
    "fraud_training_memory_usage_mb",
    "Memory usage during training in MB",
    ["model_name"],
)

TRAINING_CPU_PERCENT = Gauge(
    "fraud_training_cpu_percent", "CPU usage percentage during training", ["model_name"]
)

# MLflow integration
MLFLOW_EXPERIMENTS_TOTAL = Counter(
    "fraud_mlflow_experiments_total", "Total number of MLflow experiments created"
)

MLFLOW_RUNS_TOTAL = Counter(
    "fraud_mlflow_runs_total",
    "Total number of MLflow runs",
    ["experiment_name", "status"],
)

MLFLOW_MODELS_REGISTERED = Counter(
    "fraud_mlflow_models_registered_total",
    "Total number of models registered in MLflow",
    ["model_name"],
)

# Training info
TRAINING_INFO = Info(
    "fraud_training_info", "General information about training environment"
)

# Set training environment info
TRAINING_INFO.info(
    {
        "python_version": sys.version.split()[0],
        "environment": os.getenv("ENVIRONMENT", "development"),
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        "model_name": os.getenv("MODEL_NAME", "fraud_detection_xgboost"),
    }
)


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info("shutdown_signal_received", signal=sig)
    running = False
    sys.exit(0)


def setup_prometheus_metrics(port: int = 9095) -> None:
    """
    Start Prometheus HTTP server to expose metrics.

    Args:
        port: Port number to expose metrics on (default: 9095)
    """
    try:
        start_http_server(port)
        logger.info(
            "prometheus_metrics_server_started",
            port=port,
            endpoint=f"http://localhost:{port}/metrics",
        )
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(
                "port_already_in_use",
                port=port,
                error=f"Another process is using port {port}",
            )
            raise
        else:
            logger.error("server_startup_failed", error=str(e), exc_info=True)
            raise


def main():
    """Start the Prometheus metrics server for training monitoring."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    port = int(os.getenv("PROMETHEUS_PORT", 9095))

    logger.info(
        "starting_training_metrics_server",
        port=port,
        model_name=os.getenv("MODEL_NAME", "fraud_detection_xgboost"),
    )

    try:
        # Start Prometheus HTTP server
        setup_prometheus_metrics(port=port)
        logger.info(
            "metrics_server_started_successfully",
            endpoint=f"http://localhost:{port}/metrics",
            metrics_exposed=[
                "fraud_training_duration_seconds",
                "fraud_training_runs_total",
                "fraud_model_accuracy",
                "fraud_model_precision",
                "fraud_model_recall",
                "fraud_model_f1_score",
                "fraud_model_auc_roc",
                "fraud_hyperparameter_trials_total",
                "fraud_hyperparameter_best_score",
                "fraud_data_loading_duration_seconds",
                "fraud_feature_engineering_duration_seconds",
                "fraud_data_samples_processed_total",
                "fraud_training_memory_usage_mb",
                "fraud_training_cpu_percent",
                "fraud_mlflow_experiments_total",
                "fraud_mlflow_runs_total",
                "fraud_mlflow_models_registered_total",
            ],
        )

        # Keep the server running
        logger.info("metrics_server_ready", message="Waiting for training jobs...")
        while running:
            time.sleep(60)  # Sleep for 1 minute

    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(
                "port_already_in_use",
                port=port,
                error=f"Another process is using port {port}",
            )
            sys.exit(1)
        else:
            logger.error("server_startup_failed", error=str(e), exc_info=True)
            sys.exit(1)
    except Exception as e:
        logger.error("unexpected_error", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
