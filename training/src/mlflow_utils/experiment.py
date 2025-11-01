# training/src/mlflow_utils/experiment.py
from __future__ import annotations

import os
from contextlib import contextmanager

try:
    import mlflow  # type: ignore

    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False


def setup_experiment(name: str, tags: dict | None = None) -> None:
    """
    Idempotent setup. If MLflow isn't available/configured, it's a no-op.
    """
    if not MLFLOW_AVAILABLE:
        return

    # Respect env var if provided, otherwise MLflow uses its default (./mlruns)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(name)
    if tags:
        try:
            mlflow.set_tags(tags)
        except Exception:
            # set_tags requires an active run; ignore if none yet
            pass


@contextmanager
def start_run(run_name: str | None = None):
    """
    Context manager that works even without MLflow installed.
    Usage:
        with start_run("my_run"):
            ...
    """
    if not MLFLOW_AVAILABLE:
        yield None
        return

    active = mlflow.active_run()
    if active is None:
        with mlflow.start_run(run_name=run_name):
            yield mlflow.active_run()
    else:
        # Nested usage: just yield without creating a new run
        yield active


def end_run() -> None:
    if not MLFLOW_AVAILABLE:
        return
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        pass
