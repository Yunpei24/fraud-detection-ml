# training/src/data/loader.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import pandas as pd


def read_table(
    path: str | os.PathLike,
    *,
    fmt: Optional[Literal["csv", "parquet"]] = None,
    **pd_kwargs,
) -> pd.DataFrame:
    """
    Load a tabular dataset from CSV or Parquet.

    Parameters
    ----------
    path : file path to dataset.
    fmt : override format detection. if None, inferred from extension.
    pd_kwargs : forwarded to pandas reader.

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {p}")

    ext = (fmt or p.suffix.lstrip(".").lower())
    if ext in ("csv", "gz", "bz2", "zip", "xz"):
        return pd.read_csv(p, **pd_kwargs)
    if ext in ("parquet",):
        return pd.read_parquet(p, **pd_kwargs)
    raise ValueError(f"unsupported format: {ext} (path={p})")


def write_parquet(df: pd.DataFrame, path: str | os.PathLike, *, index: bool = False) -> None:
    """Save dataframe to Parquet, ensuring parent dir exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=index)


def safe_cast_dtypes(df: pd.DataFrame, schema: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Optionally cast columns to desired pandas dtypes (e.g., {"Class": "int8"}).
    Unknown columns are ignored.
    """
    if not schema:
        return df
    out = df.copy()
    for col, dtype in schema.items():
        if col in out.columns:
            out[col] = out[col].astype(dtype)
    return out
