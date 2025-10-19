# training/src/mlflow_utils/registry.py
from __future__ import annotations

import mlflow
from typing import Any


def register_model(run_id: str, model_path: str, model_name: str) -> None:
    """
    Register a model in the MLflow Model Registry.
    """
    model_uri = f"runs:/{run_id}/{model_path}"
    mlflow.register_model(model_uri, model_name)
    print(f"Registered model '{model_name}' from run {run_id}")


def promote_model(model_name: str, stage: str = "Staging") -> None:
    """
    Promote a registered model to a specific stage.
    """
    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions(model_name, stages=["None"])
    if not latest:
        print(f"No un-staged version found for {model_name}")
        return
    client.transition_model_version_stage(
        name=model_name,
        version=latest[0].version,
        stage=stage,
    )
    print(f"Model '{model_name}' promoted to stage '{stage}'")
