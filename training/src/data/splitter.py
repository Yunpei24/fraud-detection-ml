# training/src/data/splitter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Split:
    X_train: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    feature_names: list[str]


def supervised_train_valid_split(
    df: pd.DataFrame,
    *,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Split:
    """
    Split a labeled dataset into train and validation sets, with optional stratification.
    """
    if target_col not in df.columns:
        raise KeyError(f"target column '{target_col}' not in dataframe")

    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col])
    feats = list(X.columns)

    Xtr, Xva, ytr, yva = train_test_split(
        X.values,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )
    return Split(Xtr, Xva, ytr, yva, feats)


def unsupervised_train_valid_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    For models that do not use labels (e.g., Isolation Forest).
    """
    X = df.values
    feats = list(df.columns)
    Xtr, Xva = train_test_split(X, test_size=test_size, random_state=random_state)
    return Xtr, Xva, feats
