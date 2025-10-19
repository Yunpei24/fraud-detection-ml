# training/src/hyperparameter/spaces.py
from __future__ import annotations

from typing import Any, Dict
from scipy.stats import loguniform, uniform, randint


def xgb_search_space() -> Dict[str, Any]:
    """
    Define hyperparameter search space for XGBoost.
    """
    return {
        "n_estimators": randint(100, 600),
        "max_depth": randint(3, 10),
        "learning_rate": loguniform(1e-3, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "gamma": uniform(0, 0.5),
        "min_child_weight": uniform(0, 5),
    }


def rf_search_space() -> Dict[str, Any]:
    """Random Forest search space."""
    return {
        "n_estimators": randint(100, 600),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
    }


def mlp_search_space() -> Dict[str, Any]:
    """MLP classifier search space."""
    return {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "alpha": loguniform(1e-5, 1e-2),
        "learning_rate_init": loguniform(1e-4, 1e-1),
        "activation": ["relu", "tanh"],
    }
