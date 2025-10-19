# training/src/mlflow_utils/experiment.py
from __future__ import annotations

import mlflow


def set_experiment(name: str) -> None:
    """
    Set or create an MLflow experiment.
    """
    mlflow.set_experiment(name)
    exp = mlflow.get_experiment_by_name(name)
    print(f"Experiment '{name}' active (ID={exp.experiment_id})")


def list_experiments() -> None:
    """
    Print available MLflow experiments.
    """
    exps = mlflow.list_experiments()
    for e in exps:
        print(f"- {e.name} (id={e.experiment_id}, artifact_location={e.artifact_location})")
