# training/src/evaluation/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


@dataclass(frozen=True)
class BinaryMetrics:
    auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int


def binary_classification_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    *,
    threshold: Optional[float] = 0.5,
) -> BinaryMetrics:
    """
    Compute common binary metrics. If threshold is None, only AUC/PR-AUC are reliable.
    """
    auc = float(roc_auc_score(y_true, y_scores))
    pr_auc = float(average_precision_score(y_true, y_scores))

    if threshold is None:
        # derive labels using 0.5 just to avoid errors, but caller should pass threshold
        y_pred = (y_scores >= 0.5).astype(int)
    else:
        y_pred = (y_scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()

    return BinaryMetrics(
        auc=auc,
        pr_auc=pr_auc,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tn=tn, fp=fp, fn=fn, tp=tp,
    )


def find_best_threshold_for_f1(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """
    Sweep thresholds on [0,1] and return the best F1 (and the threshold).
    """
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for thr in np.linspace(0.01, 0.99, 99):
        y_pred = (y_scores >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {"threshold": float(thr), "f1": float(f1), "precision": float(p), "recall": float(r)}
    return best
