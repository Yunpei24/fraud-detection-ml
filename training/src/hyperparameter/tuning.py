# training/src/hyperparameter/tuning.py
from __future__ import annotations

import mlflow
import numpy as np
from typing import Any, Dict, Tuple
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score


def tune_model(
    estimator: Any,
    param_space: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_iter: int = 25,
    cv: int = 3,
    scoring: str = "f1",
    random_state: int = 42,
    experiment_name: str = "hyperparameter_tuning",
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Generic hyperparameter tuning with RandomizedSearchCV and MLflow tracking.

    Returns
    -------
    best_estimator, best_params, best_score
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{estimator.__class__.__name__}_tuning"):
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )

        search.fit(X_train, y_train)

        best_params = search.best_params_
        best_score = search.best_score_
        best_model = search.best_estimator_

        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_best_score", float(best_score))
        mlflow.sklearn.log_model(best_model, artifact_path="best_model")

        return best_model, best_params, best_score
