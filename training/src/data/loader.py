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

# Data loading utilities for the creditcard fraud dataset.

from pathlib import Path
from typing import Iterable, List
import pandas as pd

from training.src.config.logging_config import get_logger

logger = get_logger(__name__)

# canonical order for the Kaggle creditcard.csv
EXPECTED_COLUMNS: List[str] = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Class"]
)


def validate_schema(df: pd.DataFrame, required: Iterable[str] = EXPECTED_COLUMNS) -> None:
    """Raise if required columns are missing. Warn if extras are present."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    extras = [c for c in df.columns if c not in required]
    if extras:
        logger.warning(f"Extra unexpected columns found: {extras}")


def load_local_csv(path: str | Path, validate: bool = True) -> pd.DataFrame:
    """
    Load creditcard.csv from a local path.

    Args:
        path: path to CSV file
        validate: if True, enforce expected schema

    Returns:
        pandas DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path.resolve()}")
    logger.info(f"Loading CSV from {path}")
    df = pd.read_csv(path)
    if validate:
        validate_schema(df)
    logger.info(f"Loaded shape={df.shape}")
    return df
