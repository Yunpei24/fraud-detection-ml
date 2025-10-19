# training/src/mlflow_utils/tracking.py
from __future__ import annotations

import mlflow
from typing import Any, Dict


def log_run_params(params: Dict[str, Any]) -> None:
    """Log a dictionary of parameters."""
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_run_metrics(metrics: Dict[str, float]) -> None:
    """Log a dictionary of metrics."""
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))


def start_mlflow_run(name: str, *, nested: bool = False):
    """Start and yield an MLflow run context."""
    return mlflow.start_run(run_name=name, nested=nested)
