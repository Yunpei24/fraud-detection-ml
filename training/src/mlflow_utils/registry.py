# training/src/mlflow_utils/registry.py
from __future__ import annotations

from typing import Any, Optional

try:
    from mlflow.tracking import MlflowClient  # type: ignore

    import mlflow  # type: ignore

    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None  # type: ignore
    MlflowClient = object  # type: ignore
    MLFLOW_AVAILABLE = False


def register_model(model: Any, name: str, stage: str = "staging") -> Optional[str]:
    """
    Registers the current run's logged model under `name`.
    Returns model version string if available.
    Safe no-op without MLflow.
    """
    if not (MLFLOW_AVAILABLE and mlflow.active_run()):
        return None

    try:
        # If the model wasn't logged yet, attempt to log quickly.
        # Caller often already called tracking.log_model(...)
        # Here we only ensure a registration exists.
        run_id = mlflow.active_run().info.run_id  # type: ignore
        client = MlflowClient()
        # Register whatever is logged at 'model' artifact path
        mv = mlflow.register_model(f"runs:/{run_id}/model", name)
        # Optionally transition stage
        if stage:
            client.transition_model_version_stage(name, mv.version, stage)
        return str(mv.version)
    except Exception:
        return None


def transition_stage(model_name: str, version: str | int, new_stage: str) -> bool:
    if not MLFLOW_AVAILABLE:
        return False
    try:
        client = MlflowClient()
        client.transition_model_version_stage(model_name, str(version), new_stage)
        return True
    except Exception:
        return False


def get_latest_model(model_name: str, stage: str = "production"):
    """
    Returns the latest model version meta for the given stage, or None.
    """
    if not MLFLOW_AVAILABLE:
        return None
    try:
        client = MlflowClient()
        vers = client.get_latest_versions(model_name, stages=[stage])
        return vers[0] if vers else None
    except Exception:
        return None
