# training/src/evaluation/validation.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def validate_recall(y_true: np.ndarray, y_pred: np.ndarray, min_recall: float = 0.95) -> bool:
    """
    Business rule: detect at least 95% of frauds.
    """
    _, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return bool(recall >= min_recall)


def validate_fpr(y_true: np.ndarray, y_pred: np.ndarray, max_fpr: float = 0.02) -> bool:
    """
    Business rule: keep false positive rate under 2%.
    """
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    denom = max(tn + fp, 1)
    fpr = fp / denom
    return bool(fpr <= max_fpr)


def cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model,
    cv: int = 5,
    *,
    use_proba: bool = True,
) -> Dict[str, float]:
    """
    Stratified CV for imbalanced data. Returns mean metrics over folds.
    - If model has predict_proba and use_proba=True, uses it for AUC/PR-AUC.
    - Else tries decision_function; else falls back to predictions.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    aucs, pr_aucs, precisions, recalls, f1s = [], [], [], [], []

    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        cloned = _clone_estimator(model)
        cloned.fit(Xtr, ytr)

        if hasattr(cloned, "predict_proba") and use_proba:
            scores = cloned.predict_proba(Xte)[:, 1]
        elif hasattr(cloned, "decision_function"):
            scores = cloned.decision_function(Xte)
        else:
            # last resort: 0/1 labels as "scores"
            scores = cloned.predict(Xte)

        # Choose a default threshold 0.5 for supervised; for anomaly scores this is just a rough cut
        ypred = (scores >= 0.5).astype(int)

        try:
            aucs.append(roc_auc_score(yte, scores))
        except Exception:
            aucs.append(np.nan)

        try:
            pr_aucs.append(average_precision_score(yte, scores))
        except Exception:
            pr_aucs.append(np.nan)

        p, r, f1, _ = precision_recall_fscore_support(
            yte, ypred, average="binary", zero_division=0
        )
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "auc_mean": float(np.nanmean(aucs)),
        "pr_auc_mean": float(np.nanmean(pr_aucs)),
        "precision_mean": float(np.nanmean(precisions)),
        "recall_mean": float(np.nanmean(recalls)),
        "f1_mean": float(np.nanmean(f1s)),
    }


def validate_all_models(
    models_results: Dict[str, Dict[str, float]],
    *,
    min_recall: float = 0.95,
    max_fpr: float = 0.02,
) -> bool:
    """
    Check that every model in `models_results` meets business constraints.
    Expect each entry to contain at least: {"recall": ..., "fpr": ...}
    Returns True only if ALL pass.
    """
    ok = True
    for name, metrics in models_results.items():
        r_ok = metrics.get("recall", 0.0) >= min_recall
        f_ok = metrics.get("fpr", 1.0) <= max_fpr
        ok = ok and r_ok and f_ok
    return ok


# ---- helpers ----
def _clone_estimator(model):
    """
    Cheap clone for simple sklearn/XGB estimators without pulling sklearn.clone
    (keeps dependencies light). Works for common estimators.
    """
    cls = model.__class__
    try:
        return cls(**getattr(model, "get_params", lambda: {})())
    except Exception:
        # fallback: try bare constructor
        return cls()
