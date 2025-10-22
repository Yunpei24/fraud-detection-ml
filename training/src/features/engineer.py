# training/src/features/engineer.py
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def _safe_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    return df[col] if col in df.columns else None


def add_behavioral_features(
    df: pd.DataFrame,
    *,
    amount_col: str = "Amount",
    time_col: Optional[str] = "Time",
    roll_window: int = 10,
    add_ewm: bool = True,
) -> pd.DataFrame:
    """
    Behavioral signals from Amount:
      - rolling mean/std over last `roll_window` txns (global, time-ordered)
      - z-score (global)
      - optional exponentially-weighted mean/std (EWM)

    Notes:
      * We sort by time_col (if available) to reduce leakage.
      * Because the dataset has no account identifier, these are global stats.
    """
    out = df.copy()

    amt = _safe_series(out, amount_col)
    if amt is None:
        return out  # nothing to do

    # sort by time if present (stable sort)
    if time_col and time_col in out.columns:
        out = out.sort_values(time_col, kind="mergesort").reset_index(drop=True)
        amt = out[amount_col]

    # rolling stats
    roll = amt.rolling(roll_window, min_periods=1)
    out[f"amt_rollmean_{roll_window}"] = roll.mean()
    out[f"amt_rollstd_{roll_window}"] = roll.std(ddof=0).fillna(0.0)

    # global z-score (center & scale)
    out["amt_z"] = (amt - amt.mean()) / (amt.std(ddof=0) + 1e-8)

    if add_ewm:
        ewm = amt.ewm(span=max(3, roll_window // 2), adjust=False)
        out["amt_ewm_mean"] = ewm.mean()
        out["amt_ewm_std"] = ewm.std().fillna(0.0)

    return out


def add_temporal_features(
    df: pd.DataFrame,
    *,
    ts_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    If you have a real timestamp column (not present in creditcard.csv), extract hour/dayofweek.
    """
    out = df.copy()
    if ts_col and ts_col in out.columns:
        ts = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
        out["hour"] = ts.dt.hour.astype("Int16")
        out["dayofweek"] = ts.dt.day_of_week.astype("Int16")
    return out


def add_geo_risk(
    df: pd.DataFrame,
    *,
    country_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Simple geo risk encoding from frequency. Not used for creditcard.csv (no country).
    """
    out = df.copy()
    if country_col and country_col in out.columns:
        freq = out[country_col].value_counts(normalize=True)
        out["geo_risk"] = out[country_col].map(lambda c: 1.0 - float(freq.get(c, 0.0)))
    return out


def build_feature_frame(
    df: pd.DataFrame,
    *,
    amount_col: str = "Amount",
    time_col: Optional[str] = "Time",
    ts_col: Optional[str] = None,
    country_col: Optional[str] = None,
    roll_window: int = 10,
) -> pd.DataFrame:
    """
    End-to-end feature builder that composes the blocks above.
    Order:
      1) behavioral on Amount (time-ordered)
      2) optional temporal from timestamps
      3) optional geo risk
    """
    out = add_behavioral_features(
        df,
        amount_col=amount_col,
        time_col=time_col,
        roll_window=roll_window,
    )
    out = add_temporal_features(out, ts_col=ts_col)
    out = add_geo_risk(out, country_col=country_col)

    # fill NaNs introduced by rolling/EWM
    return out.fillna(0.0)
