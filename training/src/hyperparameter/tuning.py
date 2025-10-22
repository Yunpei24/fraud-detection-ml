# training/src/hyperparameter/tuning.py
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils import shuffle

try:
    import optuna  # type: ignore
    _HAS_OPTUNA = True
except Exception:
    optuna = None  # type: ignore
    _HAS_OPTUNA = False

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from ..evaluation.metrics import calculate_all_metrics
from ..features.scaling import standard_scale_fit, standard_scale_apply
from . import spaces


# --------- utilities ---------
def _supervised_objective_proba(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Primary objective = PR-AUC, with a mild AUC tie-breaker."""
    pr = average_precision_score(y_true, y_proba)
    # Small bonus for ROC-AUC to help rank close PR-AUCs
    roc = roc_auc_score(y_true, y_proba)
    return pr + 0.05 * roc


def _threshold_binarize(y_proba: np.ndarray, thr: float) -> np.ndarray:
    return (y_proba >= thr).astype(int)


def _iforest_to_proba(scores: np.ndarray) -> np.ndarray:
    """Map ISOF decision_function (higher=normal) to [0,1] anomaly probability via rank."""
    ranks = np.argsort(np.argsort(scores)) / (len(scores) - 1 + 1e-9)
    return 1.0 - ranks  # lower score -> higher anomaly prob


def _guard_optuna():
    if not _HAS_OPTUNA:
        raise RuntimeError("Optuna is not installed. Please add optuna to your environment.")


# --------- XGBoost ----------
def optimize_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 40
) -> Dict[str, Any]:
    _guard_optuna()

    def objective(trial):
        params = spaces.suggest_xgb_params(trial)
        model = XGBClassifier(**params)
        # Scale *optional*; XGB is robust, but scaling helps Amount-like features a bit
        y_tr = y_train
        X_tr = X_train
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]
        return _supervised_objective_proba(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_value": study.best_value, "study": study}


# --------- Random Forest ----------
def optimize_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 40
) -> Dict[str, Any]:
    _guard_optuna()

    def objective(trial):
        params = spaces.suggest_rf_params(trial)
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        return _supervised_objective_proba(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_value": study.best_value, "study": study}


# --------- MLP (scikit-learn) ----------
def optimize_neural_network(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 40
) -> Dict[str, Any]:
    _guard_optuna()

    # MLP benefits from scaling
    scaler = standard_scale_fit(X_train)
    X_tr = standard_scale_apply(scaler, X_train)
    X_va = standard_scale_apply(scaler, X_val)

    def objective(trial):
        params = spaces.suggest_nn_params(trial)
        model = MLPClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_train)
        proba = model.predict_proba(X_va)[:, 1]
        return _supervised_objective_proba(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_value": study.best_value, "study": study}


# --------- Isolation Forest (unsupervised) ----------
def optimize_isolation_forest(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 40
) -> Dict[str, Any]:
    _guard_optuna()

    # Use only X for training (unsupervised). Evaluate via PR-AUC on mapped anomaly proba.
    def objective(trial):
        params = spaces.suggest_if_params(trial)
        model = IsolationForest(**params)
        model.fit(X_train)
        scores = model.decision_function(X_val)  # higher=more normal
        proba = _iforest_to_proba(scores)
        return _supervised_objective_proba(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_value": study.best_value, "study": study}


# --------- Simple Ensemble ----------
def optimize_ensemble(
    models: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 40,
    model_order: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """
    Optimizes a probability-weighted average + single threshold.
    `models` should map name -> fitted model exposing predict_proba(X)[:,1]
      (for IsolationForest we internally map decision_function to [0,1]).
    """
    _guard_optuna()

    if model_order is None:
        model_order = tuple(models.keys())

    # Precompute each model's proba vector on validation
    probas = []
    for name in model_order:
        m = models[name]
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X_val)[:, 1]
        else:
            # IsolationForest, or anything with decision_function
            scores = m.decision_function(X_val)
            p = _iforest_to_proba(scores)
        probas.append(p)
    probas = np.vstack(probas)  # shape: (n_models, n_samples)

    n_models = probas.shape[0]

    def objective(trial):
        pars = spaces.suggest_ensemble_params(trial, n_models=n_models)
        w = np.array(pars["weights"], dtype=float)
        w = w / (w.sum() + 1e-12)
        thr = float(pars["threshold"])

        y_hat = (np.average(probas, axis=0, weights=w) >= thr).astype(int)
        # Business constraints: prioritize recall >= 0.95, then maximize F1
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_hat, average="binary", zero_division=0)
        # Penalize heavily if recall is below target
        target_recall = 0.95
        penalty = 0.0 if rec >= target_recall else -5.0 * (target_recall - rec)
        return f1 + penalty

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_value": study.best_value, "study": study}


# --------- Generic runner ----------
def run_trials(study, n_trials: int = 100):
    """
    Thin wrapper to run an external study object, mainly for CLI convenience.
    """
    _guard_optuna()
    study.optimize(lambda t: 0.0, n_trials=0)  # no-op; placeholder if you wire your own objective
    return study
