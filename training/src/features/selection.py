# training/src/features/selection.py
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


def top_k_by_mutual_information(
    X: pd.DataFrame, y: np.ndarray, k: int = 20, random_state: int = 42
) -> List[str]:
    """
    Rank features by mutual information with the target.
    """
    mi = mutual_info_classif(X.values, y, random_state=random_state, discrete_features="auto")
    order = np.argsort(mi)[::-1]
    k = min(k, X.shape[1])
    return [X.columns[i] for i in order[:k]]


def top_k_by_rf_importance(
    X: pd.DataFrame, y: np.ndarray, k: int = 20, random_state: int = 42
) -> List[str]:
    """
    Rank features using a quick RandomForest importance pass.
    """
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=random_state
    )
    rf.fit(X, y)
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    k = min(k, X.shape[1])
    return [X.columns[i] for i in order[:k]]


def column_subset(X: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Select a safe subset (ignores missing columns)."""
    cols = [c for c in cols if c in X.columns]
    return X[cols].copy()
