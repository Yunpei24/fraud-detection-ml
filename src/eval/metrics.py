"""
metrics.py
------------
This module provides reusable functions for evaluating supervised and unsupervised
fraud detection models. It supports AUC, PR-AUC, precision, recall, F1, and
confusion matrix generation in a standardized format.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix
)


def compute_classification_metrics(y_true, y_pred, y_score=None):
    """
    Compute classification metrics for binary classification (fraud vs legitimate).
    For unsupervised models (like Isolation Forest), y_score can be a continuous anomaly score.

    Parameters
    ----------
    y_true : array-like
        True labels (0 = legitimate, 1 = fraud)
    y_pred : array-like
        Predicted binary labels
    y_score : array-like, optional
        Continuous scores or probabilities (used for AUC and PR-AUC)

    Returns
    -------
    metrics : dict
        Dictionary containing all computed metrics.
    """
    metrics = {}

    # Compute basic metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # Compute AUCs if scores are provided
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = np.nan
        try:
            pr_auc = average_precision_score(y_true, y_score)
        except Exception:
            pr_auc = np.nan
    else:
        auc, pr_auc = np.nan, np.nan

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Save to dictionary
    metrics.update({
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc) if np.isfinite(auc) else None,
        "pr_auc": float(pr_auc) if np.isfinite(pr_auc) else None,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    })

    return metrics


def metrics_to_dataframe(metrics_dict):
    """
    Convert metrics dictionary into a one-row pandas DataFrame for easy logging or saving.

    Example:
        df = metrics_to_dataframe(metrics)
    """
    return pd.DataFrame([metrics_dict])


def print_metrics(metrics_dict):
    """
    Nicely print the metrics for CLI visibility.
    """
    print("\nModel Evaluation Results")
    print("-" * 40)
    for k, v in metrics_dict.items():
        print(f"{k:<20}: {v}")
    print("-" * 40)
