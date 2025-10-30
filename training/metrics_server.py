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

import time
import signal
import sys
import os
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
import structlog

logger = structlog.get_logger(__name__)

# Global flag for graceful shutdown
running = True

# ==============================================================================
# PROMETHEUS METRICS DEFINITIONS
# ==============================================================================

# Training metrics
TRAINING_DURATION = Histogram(
    'training_duration_seconds',
    'Time spent training a model',
    ['model_name', 'stage']
)

TRAINING_RUNS_TOTAL = Counter(
    'training_runs_total',
    'Total number of training runs',
    ['model_name', 'status']
)

# Model performance metrics
MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model accuracy on test set',
    ['model_name', 'model_version']
)

MODEL_PRECISION = Gauge(
    'model_precision',
    'Model precision on test set',
    ['model_name', 'model_version']
)

MODEL_RECALL = Gauge(
    'model_recall',
    'Model recall on test set',
    ['model_name', 'model_version']
)

MODEL_F1_SCORE = Gauge(
    'model_f1_score',
    'Model F1 score on test set',
    ['model_name', 'model_version']
)

MODEL_AUC_ROC = Gauge(
    'model_auc_roc',
    'Model AUC-ROC score',
    ['model_name', 'model_version']
)

# Hyperparameter tuning metrics
HYPERPARAMETER_TRIALS_TOTAL = Counter(
    'hyperparameter_trials_total',
    'Total number of hyperparameter tuning trials',
    ['model_name']
)

HYPERPARAMETER_BEST_SCORE = Gauge(
    'hyperparameter_best_score',
    'Best score achieved during hyperparameter tuning',
    ['model_name', 'metric']
)

# Data processing metrics
DATA_LOADING_DURATION = Histogram(
    'data_loading_duration_seconds',
    'Time spent loading data',
    ['data_source']
)

FEATURE_ENGINEERING_DURATION = Histogram(
    'feature_engineering_duration_seconds',
    'Time spent on feature engineering',
    ['stage']
)

DATA_SAMPLES_PROCESSED = Counter(
    'data_samples_processed_total',
    'Total number of data samples processed',
    ['split']
)

# System metrics
TRAINING_MEMORY_USAGE_MB = Gauge(
    'training_memory_usage_mb',
    'Memory usage during training in MB',
    ['model_name']
)

TRAINING_CPU_PERCENT = Gauge(
    'training_cpu_percent',
    'CPU usage percentage during training',
    ['model_name']
)

# MLflow integration
MLFLOW_EXPERIMENTS_TOTAL = Counter(
    'mlflow_experiments_total',
    'Total number of MLflow experiments created'
)

MLFLOW_RUNS_TOTAL = Counter(
    'mlflow_runs_total',
    'Total number of MLflow runs',
    ['experiment_name', 'status']
)

MLFLOW_MODELS_REGISTERED = Counter(
    'mlflow_models_registered_total',
    'Total number of models registered in MLflow',
    ['model_name']
)

# Training info
TRAINING_INFO = Info(
    'training_info',
    'General information about training environment'
)

# Set training environment info
TRAINING_INFO.info({
    'python_version': sys.version.split()[0],
    'environment': os.getenv('ENVIRONMENT', 'development'),
    'mlflow_tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'),
    'model_name': os.getenv('MODEL_NAME', 'fraud_detection_xgboost'),
})


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info("shutdown_signal_received", signal=sig)
    running = False
    sys.exit(0)


def setup_prometheus_metrics(port: int = 9093) -> None:
    """
    Start Prometheus HTTP server to expose metrics.
    
    Args:
        port: Port number to expose metrics on (default: 9093)
    """
    try:
        start_http_server(port)
        logger.info(
            "prometheus_metrics_server_started",
            port=port,
            endpoint=f"http://localhost:{port}/metrics"
        )
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(
                "port_already_in_use",
                port=port,
                error=f"Another process is using port {port}"
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
    
    port = int(os.getenv('PROMETHEUS_PORT', 9093))
    
    logger.info(
        "starting_training_metrics_server",
        port=port,
        model_name=os.getenv('MODEL_NAME', 'fraud_detection_xgboost')
    )
    
    try:
        # Start Prometheus HTTP server
        setup_prometheus_metrics(port=port)
        logger.info(
            "metrics_server_started_successfully",
            endpoint=f"http://localhost:{port}/metrics",
            metrics_exposed=[
                'training_duration_seconds',
                'training_runs_total',
                'model_accuracy',
                'model_precision',
                'model_recall',
                'model_f1_score',
                'model_auc_roc',
                'hyperparameter_trials_total',
                'hyperparameter_best_score',
                'data_loading_duration_seconds',
                'feature_engineering_duration_seconds',
                'data_samples_processed_total',
                'training_memory_usage_mb',
                'training_cpu_percent',
                'mlflow_experiments_total',
                'mlflow_runs_total',
                'mlflow_models_registered_total',
            ]
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
                error=f"Another process is using port {port}"
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
