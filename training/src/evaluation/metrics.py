# training/src/evaluation/metrics.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)


def calculate_auc_roc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    AUC over ROC; expects probability-like scores for the positive class (fraud=1).
    If not computable, returns NaN.
    """
    try:
        return float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        return float("nan")


def calculate_precision_recall(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Precision and recall for binary classification (fraud is positive class=1).
    """
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(precision), float(recall)


def calculate_f2_score(precision: float, recall: float) -> float:
    """
    F-beta with beta=2 (recall-weighted).
    """
    beta2 = 2.0**2
    denom = beta2 * precision + recall
    if denom <= 0:
        return 0.0
    return float((1 + beta2) * (precision * recall) / denom)


def confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Return confusion matrix counts + useful rates.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / max(tp + fn, 1)  # recall
    fpr = fp / max(fp + tn, 1)
    tnr = tn / max(tn + fp, 1)  # specificity
    fnr = fn / max(fn + tp, 1)
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tnr": float(tnr),
        "fnr": float(fnr),
    }


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Consolidated metrics (good for MLflow logging & thresholds checks).
    """
    auc = calculate_auc_roc(y_true, y_pred_proba)
    try:
        pr_auc = float(average_precision_score(y_true, y_pred_proba))
    except Exception:
        pr_auc = float("nan")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    f2 = calculate_f2_score(precision, recall)
    cm = confusion_matrix_dict(y_true, y_pred)

    # Get PR curve points to compute best F1 threshold if useful later
    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        # sometimes thresholds length != precisions length
        best_f1 = 0.0
        if len(thresholds) > 0:
            f1s = (2 * precisions[:-1] * recalls[:-1]) / np.clip(
                precisions[:-1] + recalls[:-1], 1e-12, None
            )
            best_f1 = float(np.nanmax(f1s))
    except Exception:
        best_f1 = float("nan")

    return {
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "fpr": cm["fpr"],
        "tnr": cm["tnr"],
        "tn": cm["tn"],
        "fp": cm["fp"],
        "fn": cm["fn"],
        "tp": cm["tp"],
        "best_f1_from_curve": best_f1,
    }
