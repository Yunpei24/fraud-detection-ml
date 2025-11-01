# training/src/models/random_forest.py
"""
Random Forest model wrapper with SMOTE support for imbalanced fraud detection.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from src.config.logging_config import get_logger

from .hyperparameter_tuning import tune_random_forest_hyperparameters

logger = get_logger(__name__)


class RandomForestModel:
    """
    Random Forest classifier with optional SMOTE for handling class imbalance.

    Parameters:
    -----------
    use_smote : bool
        Whether to apply SMOTE oversampling during training
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning
    random_state : int
        Random seed for reproducibility
    params : Dict[str, Any]
        Additional RandomForest hyperparameters
    """

    def __init__(
        self,
        use_smote: bool = True,
        tune_hyperparams: bool = False,
        random_state: int = 42,
        n_estimators: int = 300,
        n_jobs: int = -1,
        oob_score: bool = False,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.use_smote = use_smote
        self.tune_hyperparams = tune_hyperparams
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.params = params or {}

        # Default parameters optimized for fraud detection
        default_params = {
            "n_estimators": n_estimators,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": None if use_smote else "balanced",
            "random_state": random_state,
            "n_jobs": n_jobs,
            "oob_score": oob_score,
        }

        # Merge user params
        default_params.update(self.params)

        self.model = RandomForestClassifier(**default_params)
        self.smote = SMOTE(random_state=random_state) if use_smote else None
        self.is_fitted = False

        logger.info(
            f"RandomForestModel initialized (use_smote={use_smote}, tune_hyperparams={tune_hyperparams})"
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RandomForestModel":
        """
        Train the Random Forest model with optional SMOTE resampling and hyperparameter tuning.
        """
        X_train, y_train = X, y

        if self.use_smote and self.smote is not None:
            logger.info(
                f"Applying SMOTE: original class distribution: {np.bincount(y_train.astype(int))}"
            )
            X_train, y_train = self.smote.fit_resample(X, y)
            logger.info(f"After SMOTE: {np.bincount(y_train.astype(int))}")

        if self.tune_hyperparams and X_val is not None and y_val is not None:
            logger.info("Performing hyperparameter tuning...")
            tuning_results = tune_random_forest_hyperparameters(
                X_train, y_train, X_val, y_val, random_state=self.random_state
            )

            # Update model with best parameters
            best_params = tuning_results["best_params"]
            self.model = RandomForestClassifier(**best_params)

            logger.info(f"Using tuned hyperparameters: {best_params}")

        logger.info(f"Training Random Forest on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("Random Forest training completed")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities. Returns shape (n_samples, 2).
        """
        return self.model.predict_proba(X)

    @property
    def oob_score_(self):
        """Access the out-of-bag score from the underlying model."""
        return getattr(self.model, "oob_score_", None)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances from the trained model.
        """
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model must be fitted before getting feature importance")

        # Return as dict for compatibility with tests
        return {
            f"feature_{i}": float(imp)
            for i, imp in enumerate(self.model.feature_importances_)
        }
