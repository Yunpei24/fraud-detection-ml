# src/drift/ks_test.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal, Tuple, Dict
from scipy.stats import ks_2samp

Tail = Literal["two-sided", "less", "greater"]

def ks_test_1d(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    alternative: Tail = "two-sided",
) -> Tuple[float, float]:
    """
    2-sample Kolmogorov-Smirnov test for a single numeric feature.

    Parameters
    ----------
    expected : array-like
        Baseline sample (e.g., training distribution).
    actual : array-like
        Current sample (e.g., production distribution).
    alternative : {"two-sided","less","greater"}
        Hypothesis type. "two-sided" is standard for drift.

    Returns
    -------
    stat : float
        KS statistic (0..1). Larger means more separation.
    p_value : float
        p value for the chosen alternative.
    """
    x = np.asarray(pd.Series(expected).dropna(), dtype=float)
    y = np.asarray(pd.Series(actual).dropna(), dtype=float)
    if x.size == 0 or y.size == 0:
        # if no data, return NaNs so callers can decide what to do
        return float("nan"), float("nan")
    stat, p = ks_2samp(x, y, alternative=alternative, mode="auto")
    return float(stat), float(p)


def ks_test_df(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    cols: list[str] | None = None,
    alternative: Tail = "two-sided",
) -> pd.DataFrame:
    """
    Apply KS test column-wise for numeric features shared by both DataFrames.

    Parameters
    ----------
    expected_df : DataFrame
        Baseline data (e.g., training).
    actual_df : DataFrame
        Current data (e.g., prod window).
    cols : list[str] | None
        Subset of columns to test. Defaults to numeric intersection.
    alternative : {"two-sided","less","greater"}
        Hypothesis type.

    Returns
    -------
    DataFrame with index=column name, columns=["ks_stat","p_value","n_expected","n_actual"].
    """
    if cols is None:
        cols = sorted(
            set(expected_df.select_dtypes(include="number").columns)
            & set(actual_df.select_dtypes(include="number").columns)
        )

    rows = []
    for c in cols:
        stat, p = ks_test_1d(expected_df[c], actual_df[c], alternative=alternative)
        rows.append({
            "feature": c,
            "ks_stat": stat,
            "p_value": p,
            "n_expected": int(expected_df[c].dropna().shape[0]),
            "n_actual": int(actual_df[c].dropna().shape[0]),
        })
    return pd.DataFrame(rows).set_index("feature")
