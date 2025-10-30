# common/src/fraud_detection_common/preprocessor.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataPreprocessor:
    drop_columns: Iterable[str] = field(default_factory=lambda: ["Time"])
    scale_columns: Iterable[str] = field(default_factory=lambda: ["amount"])
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
