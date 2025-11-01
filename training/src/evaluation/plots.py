# training/src/evaluation/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def plot_roc_auc(
    y_true: np.ndarray, y_pred_proba: np.ndarray, *, title: str = "ROC AUC"
):
    """
    Returns a Matplotlib figure with ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], ls="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


def plot_precision_recall_curve_plot(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    *,
    title: str = "Precision-Recall Curve",
):
    """
    Returns a Matplotlib figure with PR curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


def plot_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Confusion Matrix",
    normalize: Optional[str] = None,  # "true", "pred", or None
):
    """
    Returns a Matplotlib figure with the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    model,
    feature_names: Iterable[str],
    *,
    top_n: int = 20,
    title: str = "Feature Importance",
):
    """
    Handles:
      - XGBoost/sklearn tree models via .feature_importances_
      - Linear models via .coef_
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        importances = np.abs(coef).ravel()
    else:
        raise ValueError("Model has no feature_importances_ or coef_.")

    feature_names = list(feature_names)
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx[::-1]], importances[idx][::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    return fig


def save_plots(output_dir: str, *figs):
    """
    Save sequentially named figures to output_dir as PNGs.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, fig in enumerate(figs, start=1):
        fig.savefig(out / f"plot_{i:02d}.png", dpi=160, bbox_inches="tight")
