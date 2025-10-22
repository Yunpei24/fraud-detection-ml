# training/src/data/preprocessor.py
from __future__ import annotations  # Move this to the very top

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from training.src.data.schema_validation import validate_schema 
from training.src.data.utils import fill_na


def drop_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Drop columns if they exist."""
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

# Continue with the rest of the functions...


def drop_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Drop columns if they exist."""
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")


def fill_na(
    df: pd.DataFrame,
    *,
    num_strategy: str = "median",  # Fixed the unterminated string literal here
    cat_strategy: str = "most_frequent",
    cat_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Simple NA imputation for numeric and categorical columns.
    """
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if cat_cols is None:
        cat_cols = [c for c in out.columns if c not in num_cols]

    if num_strategy == "median":
        for c in num_cols:
            out[c] = out[c].fillna(out[c].median())
    elif num_strategy == "mean":
        for c in num_cols:
            out[c] = out[c].fillna(out[c].mean())

    if cat_strategy == "most_frequent":
        for c in cat_cols:
            out[c] = out[c].fillna(out[c].mode().iloc[0] if not out[c].mode().empty else out[c].iloc[0])

    return out


def encode_categoricals_onehot(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """One-hot encode given categorical columns."""
    if not cols:
        return df
    return pd.get_dummies(df, columns=[c for c in cols if c in df.columns], drop_first=True)

# Preprocessing helpers for creditcard.csv.

# Fix dtypes
# Optional outlier handling (IQR)
# Drop columns (commonly Time)
# Scale selected columns (commonly Amount)
# Persist fitted transformers for production

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from training.src.config.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class DataPreprocessor:
    drop_columns: Iterable[str] = field(default_factory=lambda: ["Time"])
    scale_columns: Iterable[str] = field(default_factory=lambda: ["Amount"])
    outlier_columns: Iterable[str] = field(default_factory=list)
    outlier_method: Optional[str] = None  # "IQR" or None
    scaler_: Optional[StandardScaler] = None

    def _fix_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Class" in df.columns:
            df["Class"] = df["Class"].astype(int)
        return df

    def _remove_outliers_iqr(self, df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
        """Clip outliers using IQR fences. Keeps shape stable."""
        for c in cols:
            if c not in df.columns:
                logger.warning(f"Outlier column '{c}' not in frame, skipping")
                continue
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            before = df[c].copy()
            df[c] = df[c].clip(lower=low, upper=high)
            changed = int((before != df[c]).sum())
            if changed:
                logger.info(f"IQR clipped {changed} values in {c}")
        return df

    def _scale_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.scale_columns:
            return df
        self.scaler_ = StandardScaler()
        df[list(self.scale_columns)] = self.scaler_.fit_transform(df[list(self.scale_columns)])
        logger.info(f"Fitted StandardScaler on columns {list(self.scale_columns)}")
        return df

    def _scale_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None or not self.scale_columns:
            return df
        df[list(self.scale_columns)] = self.scaler_.transform(df[list(self.scale_columns)])
        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        persist_dir: Optional[str | Path] = None,
        artifacts_prefix: str = "preprocessor",
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        df = df.copy()

        # schema and dtypes
        df = self._fix_dtypes(df)

        # drop columns
        for c in self.drop_columns:
            if c in df.columns:
                df = df.drop(columns=[c])

        # outliers
        if self.outlier_method == "IQR" and self.outlier_columns:
            df = self._remove_outliers_iqr(df, self.outlier_columns)

        # scaling
        df = self._scale_fit(df)

        artifacts = {}
        if persist_dir:
            persist_dir = Path(persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            if self.scaler_ is not None:
                p = persist_dir / f"{artifacts_prefix}_scaler.joblib"
                joblib.dump(self.scaler_, p)
                artifacts["scaler"] = str(p)
                logger.info(f"Saved scaler to {p}")

        return df, artifacts

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._fix_dtypes(df)
        for c in self.drop_columns:
            if c in df.columns:
                df = df.drop(columns=[c])
        df = self._scale_transform(df)
        return df

# training/src/data/preprocessor.py

def check_data_quality(df: pd.DataFrame) -> Dict[str, int]:
    """Returns a report on data quality including row count, column count, and null counts."""
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "null_counts": df.isna().sum().to_dict()
    }
