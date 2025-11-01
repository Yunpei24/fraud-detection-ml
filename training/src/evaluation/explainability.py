# training/src/evaluation/explainability.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def generate_shap_values(model, X_sample: np.ndarray):
    """
    Returns (explainer, shap_values) using the best available SHAP explainer.
    - TreeExplainer for tree-based (XGBoost, RandomForest, etc.)
    - LinearExplainer for linear models
    - KernelExplainer fallback (slower) for others

    X_sample should be a small slice for speed (e.g., 2k rows).
    """
    import shap  # heavy import

    # Pick the right explainer
    explainer = None
    try:
        if _is_tree_model(model):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        elif _is_linear_model(model):
            explainer = shap.LinearExplainer(
                model, X_sample, feature_perturbation="interventional"
            )
            shap_values = explainer.shap_values(X_sample)
        else:
            # Kernel (model-agnostic). Use a small background to speed up.
            background = (
                shap.sample(X_sample, 200) if X_sample.shape[0] > 200 else X_sample
            )
            explainer = (
                shap.KernelExplainer(model.predict_proba, background)
                if hasattr(model, "predict_proba")
                else shap.KernelExplainer(model.predict, background)
            )
            shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        raise RuntimeError(f"Failed to generate SHAP values: {e}") from e

    return explainer, shap_values


def plot_shap_summary(
    shap_values,
    X: np.ndarray,
    feature_names: Optional[list] = None,
    *,
    title: str = "SHAP Summary",
):
    """
    Summary dot plot. Returns the Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import shap

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(title)
    return fig


def save_explainer(explainer, path: str):
    """
    Persist SHAP explainer (pickle). Note: KernelExplainer objects can be large.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(explainer, f)


def create_explanation_report(
    shap_values, X: np.ndarray, feature_names: Optional[list] = None, top_n: int = 20
) -> Dict[str, float]:
    """
    Produce a simple global importance report: mean(|SHAP|) per feature.
    Works for both list-like outputs (multi-class) and arrays.
    """
    sv = shap_values
    if isinstance(sv, list):  # some explainers return a list per class
        sv = sv[1] if len(sv) > 1 else sv[0]  # use positive class if available
    sv = np.array(sv)
    # shape: [n_samples, n_features]
    importance = np.mean(np.abs(sv), axis=0)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    order = np.argsort(importance)[::-1][:top_n]
    return {feature_names[i]: float(importance[i]) for i in order}


def create_explanation_report_from_model(
    model, X_sample: np.ndarray, feature_names: Optional[list] = None, top_n: int = 20
) -> Tuple[Dict[str, float], object]:
    """
    Convenience wrapper: build SHAP explainer, compute values, and return (report, explainer).
    """
    explainer, shap_values = generate_shap_values(model, X_sample)
    report = create_explanation_report(
        shap_values, X_sample, feature_names=feature_names, top_n=top_n
    )
    return report, explainer


# ---- helpers ----
def _is_tree_model(model) -> bool:
    name = model.__class__.__name__.lower()
    return any(
        k in name
        for k in ["xgb", "xgboost", "forest", "gbm", "gradientboost", "lightgbm"]
    )


def _is_linear_model(model) -> bool:
    name = model.__class__.__name__.lower()
    return any(k in name for k in ["logistic", "linear", "sgd"])
