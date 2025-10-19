# training/src/evaluation/validation.py
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def assert_no_missing(df: pd.DataFrame, cols: Iterable[str] | None = None) -> None:
    """
    Raise if any column in `cols` (or all columns) contains missing values.
    """
    focus = df.columns if cols is None else list(cols)
    na = df[focus].isnull().sum()
    offenders = na[na > 0]
    if len(offenders) > 0:
        raise ValueError(f"missing values found in columns: {list(offenders.index)}")


def assert_value_ranges(df: pd.DataFrame, rules: Dict[str, Tuple[float, float]]) -> None:
    """
    Validate that columns lie within [min, max] ranges, raises with offenders.
    Example: {"Amount": (0, 100000)}.
    """
    bad = {}
    for col, (mn, mx) in rules.items():
        if col not in df.columns:
            continue
        s = df[col]
        mask = (s < mn) | (s > mx)
        if mask.any():
            bad[col] = int(mask.sum())
    if bad:
        raise ValueError(f"value range violations: {bad}")
