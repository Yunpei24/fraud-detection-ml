# training/src/features/scaling.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


class ColumnScaler:
    """
    Fit / transform a selected set of numeric columns and persist the scaler.

    Usage:
        scaler = ColumnScaler(cols=["Amount", "V1", "V2"])
        df_scaled = scaler.fit_transform(df)
        scaler.save("artifacts/scaler.joblib")
        ...
        scaler = ColumnScaler.load("artifacts/scaler.joblib")
        df2 = scaler.transform(df2)
    """

    def __init__(self, cols: Sequence[str]):
        self.cols = list(cols)
        self._scaler = StandardScaler()

    def fit(self, df: pd.DataFrame) -> "ColumnScaler":
        self._scaler.fit(df[self.cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.cols] = self._scaler.transform(out[self.cols])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        dump({"cols": self.cols, "scaler": self._scaler}, p)

    @staticmethod
    def load(path: str | Path) -> "ColumnScaler":
        blob = load(path)
        obj = ColumnScaler(cols=blob["cols"])
        obj._scaler = blob["scaler"]
        return obj
