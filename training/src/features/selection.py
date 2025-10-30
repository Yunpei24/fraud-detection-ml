# training/src/features/selection.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.linear_model import LogisticRegression


def variance_filter(
    X: pd.DataFrame,
    *,
    threshold: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop near-constant columns (variance <= threshold).
    Returns (filtered_df, kept_columns).
    """
    selector = VarianceThreshold(threshold=threshold)
    Xt = selector.fit_transform(X.values)
    kept_mask = selector.get_support()
    kept_cols = list(X.columns[kept_mask])
    X_filtered = pd.DataFrame(Xt, columns=kept_cols, index=X.index)
    return X_filtered, kept_cols


def mutual_information_score(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> pd.Series:
    """
    Compute mutual information per feature with the target (classification).
    Returns a pd.Series sorted descending (high -> more informative).
    """
    mi = mutual_info_classif(
        X.values,
        y.values.astype(int),
        n_neighbors=n_neighbors,
        random_state=random_state,
        discrete_features=False,
    )
    scores = pd.Series(mi, index=X.columns, name="mutual_information")
    return scores.sort_values(ascending=False)


def select_k_best_mutual_info(
    X: pd.DataFrame,
    *,
    target: np.ndarray,
    k: int = 20,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select top k features based on mutual information with target.

    Args:
        X: Input features DataFrame
        target: Target variable (binary labels)
        k: Number of top features to select
        n_neighbors: Number of neighbors for MI calculation
        random_state: Random seed

    Returns:
        Tuple of (filtered DataFrame, list of selected feature names)
    """
    y = pd.Series(target, name="target")
    scores = mutual_information_score(
        X, y, n_neighbors=n_neighbors, random_state=random_state
    )

    # Select top k features
    top_k = min(k, len(scores))
    selected_features = scores.head(top_k).index.tolist()

    X_selected = X[selected_features]
    return X_selected, selected_features


def correlation_analysis(
    X: pd.DataFrame,
    *,
    method: str = "pearson",
    abs_threshold: float = 0.9,
) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Find pairs of highly correlated features. Returns a dict with:
      - 'pairs': list of (feat_i, feat_j, corr)
      - 'drop_candidates': list of features you might drop (heuristic)
    """
    corr = X.corr(method=method)
    pairs: List[Tuple[str, str, float]] = []
    drop_candidates = set()

    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if abs(c) >= abs_threshold:
                fi, fj = cols[i], cols[j]
                pairs.append((fi, fj, float(c)))
                # heuristic: mark one side (later pruned by importance)
                drop_candidates.add(fj)

    return {
        "pairs": pairs,
        "drop_candidates": list(drop_candidates),
    }


def remove_collinear_features(
    X: pd.DataFrame,
    *,
    method: str = "pearson",
    abs_threshold: float = 0.95,
    keep: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly collinear columns (greedy). If `keep` provided, those
    columns will be preserved preferentially when possible.
    """
    keep = set(keep or [])
    corr = X.corr(method=method).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()

    for col in upper.columns:
        high = upper[col][upper[col] >= abs_threshold]
        if not high.empty:
            # prefer to drop `col` unless it is in keep
            if col not in keep:
                to_drop.add(col)
            else:
                # drop partners not in keep
                for partner in high.index:
                    if partner not in keep:
                        to_drop.add(partner)

    X_out = X.drop(columns=list(to_drop), errors="ignore")
    return X_out, list(X_out.columns)


def get_important_features(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    top_n: Optional[int] = 20,
    backend: str = "rf",
    random_state: int = 42,
    max_iter: int = 200,
) -> pd.Series:
    """
    Model-based importance ranking. Backends:
      - "rf": RandomForestClassifier (default, robust)
      - "logreg": LogisticRegression (liblinear, absolute coef)
    Returns a Series sorted descending. If top_n is provided, it is truncated.
    """
    backend = backend.lower()
    if backend == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(X.values, y.values.astype(int))
        imp = pd.Series(model.feature_importances_, index=X.columns, name="importance")
        imp = imp.sort_values(ascending=False)
    elif backend == "logreg":
        # scale before logistic outside this function if needed
        lr = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state,
        )
        lr.fit(X.values, y.values.astype(int))
        coef = np.abs(lr.coef_).ravel()
        imp = pd.Series(coef, index=X.columns, name="importance").sort_values(
            ascending=False
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if top_n is not None:
        return imp.head(top_n)
    return imp
