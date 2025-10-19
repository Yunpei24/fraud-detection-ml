# training/src/evaluation/explainability.py
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


def shap_summary_tree(model: Any, X: pd.DataFrame, *, max_display: int = 20) -> Optional[np.ndarray]:
    """
    SHAP summary for tree models (XGBoost/LightGBM/RandomForest). Safe no-op if shap missing.
    Returns SHAP values for potential further use.
    """
    if not _HAS_SHAP:
        print("shap not installed - skipping explainability")
        return None
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    # The following call draws the plot if running in a notebook
    try:
        shap.summary_plot(sv, X, max_display=max_display, show=False)
    except Exception:
        pass
    return np.array(sv.values)


def permutation_feature_importance(model: Any, X: pd.DataFrame, y: np.ndarray, *, n_repeats: int = 5) -> pd.DataFrame:
    """
    Lightweight permutation importance wrapper (works for sklearn-compatible estimators).
    """
    from sklearn.inspection import permutation_importance

    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    imp = pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
    return imp.sort_values("importance_mean", ascending=False).reset_index(drop=True)
