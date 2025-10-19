# training/src/features/engineer.py
from __future__ import annotations

import pandas as pd


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example behavioral signals:
    - amount to running mean
    - z-score of amount
    """
    out = df.copy()
    if "Amount" in out.columns:
        out["amt_rolling_mean_10"] = out["Amount"].rolling(10, min_periods=1).mean()
        out["amt_z"] = (out["Amount"] - out["Amount"].mean()) / (out["Amount"].std() + 1e-8)
    return out


def add_temporal_features(df: pd.DataFrame, *, ts_col: str | None = None) -> pd.DataFrame:
    """
    If you have a timestamp column (e.g., unix or ISO8601), extract hour, dow.
    """
    out = df.copy()
    if ts_col and ts_col in out.columns:
        ts = pd.to_datetime(out[ts_col], errors="coerce")
        out["hour"] = ts.dt.hour
        out["dayofweek"] = ts.dt.dayofweek
    return out


def add_geo_risk(df: pd.DataFrame, *, country_col: str | None = None) -> pd.DataFrame:
    """
    Simple geo risk encoding: minority frequency.
    """
    out = df.copy()
    if country_col and country_col in out.columns:
        freq = out[country_col].value_counts(normalize=True)
        out["geo_risk"] = out[country_col].map(lambda c: 1.0 - float(freq.get(c, 0.0)))
    return out


def build_feature_frame(
    df: pd.DataFrame,
    *,
    ts_col: str | None = None,
    country_col: str | None = None,
) -> pd.DataFrame:
    """
    End-to-end feature builder that composes the blocks above.
    """
    out = add_behavioral_features(df)
    out = add_temporal_features(out, ts_col=ts_col)
    out = add_geo_risk(out, country_col=country_col)
    # fill any freshly created NaNs from rolling ops
    return out.fillna(0.0)
