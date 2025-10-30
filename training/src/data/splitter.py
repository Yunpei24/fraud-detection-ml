"""
Dataset splitting utilities.

- stratified_split: standard stratified train val test
- time_aware_split: preserves temporal order using the 'Time' column
- save_splits: write to parquet for reuse
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.logging_config import get_logger

logger = get_logger(__name__)


def stratified_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame | np.ndarray]:
    """
    Return stratified train, val, test splits.
    val_size is relative to the full dataset, not to the train chunk.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # compute val share relative to remaining pool
    val_rel = val_size / (1.0 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_rel), stratify=y_temp, random_state=random_state
    )

    logger.info(
        f"Stratified split -> "
        f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def time_aware_split(
    df: pd.DataFrame,
    target_col: str = "Class",
    time_col: str = "Time",
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Dict[str, pd.DataFrame | np.ndarray]:
    """
    Split by time. Keeps earlier rows for train, later rows for val and test.

    Assumes `time_col` increases with time. Does not do stratification.
    """
    if time_col not in df.columns:
        raise ValueError(f"time_aware_split requires column '{time_col}'")

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError("Invalid sizes. Choose smaller test_size or val_size.")

    train_df = df_sorted.iloc[:n_train]
    val_df = df_sorted.iloc[n_train:n_train + n_val]
    test_df = df_sorted.iloc[n_train + n_val:]

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col].values
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values

    logger.info(
        f"Time aware split -> "
        f"train={train_df.shape}, val={val_df.shape}, test={test_df.shape}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_splits(
    splits: Dict[str, pd.DataFrame | np.ndarray],
    out_dir: str | Path = "training/artifacts/splits",
    prefix: str = "creditcard",
) -> Dict[str, str]:
    """
    Persist splits to parquet and return file map.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    for key in ["X_train", "X_val", "X_test"]:
        p = out / f"{prefix}_{key}.parquet"
        splits[key].to_parquet(p, index=False)
        paths[key] = str(p)

    for key in ["y_train", "y_val", "y_test"]:
        p = out / f"{prefix}_{key}.parquet"
        pd.DataFrame({key: splits[key]}).to_parquet(p, index=False)
        paths[key] = str(p)

    logger.info(f"Wrote splits to {out.resolve()}")
    return paths
