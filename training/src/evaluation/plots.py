# training/src/evaluation/plots.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


def plot_roc(y_true: np.ndarray, y_scores: np.ndarray, *, title: str = "ROC Curve") -> None:
    RocCurveDisplay.from_predictions(y_true, y_scores)
    plt.title(title)
    plt.tight_layout()


def plot_pr(y_true: np.ndarray, y_scores: np.ndarray, *, title: str = "Precision-Recall Curve") -> None:
    PrecisionRecallDisplay.from_predictions(y_true, y_scores)
    plt.title(title)
    plt.tight_layout()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, *, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title(title)
    plt.tight_layout()
