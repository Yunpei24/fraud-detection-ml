# training/src/features/scaling.py
from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


ScalerT = Union[StandardScaler, MinMaxScaler]


def make_scaler(strategy: str = "standard") -> ScalerT:
    """
    Factory:
      - "standard" -> StandardScaler (zero mean, unit variance)
      - "minmax"   -> MinMaxScaler (0..1), handy for neural nets
    """
    s = strategy.lower()
    if s == "standard":
        return StandardScaler()
    if s == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaling strategy: {strategy}")


def fit_scaler(
    X_train: Union[pd.DataFrame, np.ndarray],
    *,
    cols: Optional[Iterable[str]] = None,
    strategy: str = "standard",
) -> Tuple[ScalerT, Optional[list]]:
    """
    Fit a scaler on training data. If DataFrame and `cols` given, fit on those columns
    and remember the column order. Returns (scaler, col_list).
    """
    scaler = make_scaler(strategy)
    if isinstance(X_train, pd.DataFrame):
        if cols is None:
            cols = list(X_train.columns)
        Xsub = X_train[cols].values
        scaler.fit(Xsub)
        return scaler, list(cols)
    else:
        scaler.fit(X_train)
        return scaler, None


def transform_with_scaler(
    X: Union[pd.DataFrame, np.ndarray],
    scaler: ScalerT,
    *,
    cols: Optional[Iterable[str]] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Apply a fitted scaler. If DataFrame and `cols` provided, transform only those
    columns and return a DataFrame with the same shape and column order.
    """
    if isinstance(X, pd.DataFrame):
        X_out = X.copy()
        if cols is None:
            cols = list(X_out.columns)
        X_out.loc[:, cols] = scaler.transform(X_out[cols].values)
        return X_out
    else:
        return scaler.transform(X)


def save_scaler(scaler: ScalerT, path: str) -> None:
    """
    Persist a fitted scaler (joblib).
    """
    joblib.dump(scaler, path)


def load_scaler(path: str) -> ScalerT:
    """
    Load a fitted scaler (joblib).
    """
    return joblib.load(path)
