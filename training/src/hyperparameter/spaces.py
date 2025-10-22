# training/src/hyperparameter/spaces.py
from __future__ import annotations
from typing import Dict, Any

try:
    import optuna  # type: ignore
    _HAS_OPTUNA = True
except Exception:
    optuna = None  # type: ignore
    _HAS_OPTUNA = False


# Static describers (useful for docs/printing)
XGB_PARAM_SPACE: Dict[str, Any] = {
    "n_estimators": (200, 800),
    "max_depth": (3, 10),
    "learning_rate": (1e-3, 0.3, "log"),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_lambda": (1e-3, 10.0, "log"),
}

RF_PARAM_SPACE: Dict[str, Any] = {
    "n_estimators": (200, 1000),
    "max_depth": (3, 20),
    "max_features": ("sqrt", "log2"),
    "min_samples_split": (2, 10),
    "min_samples_leaf": (1, 10),
}

NN_PARAM_SPACE: Dict[str, Any] = {
    "hidden_layers": (1, 3),
    "hidden_units_min": 32,
    "hidden_units_max": 256,
    "alpha": (1e-5, 1e-1, "log"),
    "learning_rate_init": (1e-4, 1e-2, "log"),
}

IF_PARAM_SPACE: Dict[str, Any] = {
    "n_estimators": (100, 600),
    "max_samples": (0.5, 1.0),
    "contamination": (0.001, 0.02),  # 0.1% to 2%
    "max_features": (0.5, 1.0),
}

ENSEMBLE_PARAM_SPACE: Dict[str, Any] = {
    "threshold": (0.05, 0.95),
    "weights": (0.1, 1.0),  # uniform range for each model weight
}


# Trial-driven suggestors (called inside Optuna objective)
def suggest_xgb_params(trial) -> Dict[str, Any]:
    assert _HAS_OPTUNA, "Optuna not installed"
    return {
        "n_estimators": trial.suggest_int("xgb_n_estimators", *XGB_PARAM_SPACE["n_estimators"]),
        "max_depth": trial.suggest_int("xgb_max_depth", *XGB_PARAM_SPACE["max_depth"]),
        "learning_rate": trial.suggest_float("xgb_learning_rate", *XGB_PARAM_SPACE["learning_rate"][:2], log=True),
        "subsample": trial.suggest_float("xgb_subsample", *XGB_PARAM_SPACE["subsample"]),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", *XGB_PARAM_SPACE["colsample_bytree"]),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", *XGB_PARAM_SPACE["reg_lambda"][:2], log=True),
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": 0,
        "random_state": 42,
    }


def suggest_rf_params(trial) -> Dict[str, Any]:
    assert _HAS_OPTUNA, "Optuna not installed"
    return {
        "n_estimators": trial.suggest_int("rf_n_estimators", *RF_PARAM_SPACE["n_estimators"]),
        "max_depth": trial.suggest_int("rf_max_depth", *RF_PARAM_SPACE["max_depth"]),
        "max_features": trial.suggest_categorical("rf_max_features", RF_PARAM_SPACE["max_features"]),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", *RF_PARAM_SPACE["min_samples_split"]),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", *RF_PARAM_SPACE["min_samples_leaf"]),
        "n_jobs": 0,
        "random_state": 42,
    }


def suggest_nn_params(trial) -> Dict[str, Any]:
    assert _HAS_OPTUNA, "Optuna not installed"
    layers = trial.suggest_int("nn_hidden_layers", *NN_PARAM_SPACE["hidden_layers"])
    hidden = []
    for i in range(layers):
        hidden.append(
            trial.suggest_int(f"nn_units_{i+1}", NN_PARAM_SPACE["hidden_units_min"], NN_PARAM_SPACE["hidden_units_max"])
        )
    return {
        "hidden_layer_sizes": tuple(hidden),
        "alpha": trial.suggest_float("nn_alpha", *NN_PARAM_SPACE["alpha"][:2], log=True),
        "learning_rate_init": trial.suggest_float("nn_lr_init", *NN_PARAM_SPACE["learning_rate_init"][:2], log=True),
        "random_state": 42,
        "max_iter": 150,
        "early_stopping": True,
        "n_iter_no_change": 10,
    }


def suggest_if_params(trial) -> Dict[str, Any]:
    assert _HAS_OPTUNA, "Optuna not installed"
    return {
        "n_estimators": trial.suggest_int("if_n_estimators", *IF_PARAM_SPACE["n_estimators"]),
        "max_samples": trial.suggest_float("if_max_samples", *IF_PARAM_SPACE["max_samples"]),
        "contamination": trial.suggest_float("if_contamination", *IF_PARAM_SPACE["contamination"]),
        "max_features": trial.suggest_float("if_max_features", *IF_PARAM_SPACE["max_features"]),
        "random_state": 42,
        "n_jobs": 0,
    }


def suggest_ensemble_params(trial, n_models: int = 3) -> Dict[str, Any]:
    assert _HAS_OPTUNA, "Optuna not installed"
    weights = [trial.suggest_float(f"ens_w_{i}", *ENSEMBLE_PARAM_SPACE["weights"]) for i in range(n_models)]
    thr = trial.suggest_float("ens_threshold", *ENSEMBLE_PARAM_SPACE["threshold"])
    return {"weights": weights, "threshold": thr}
