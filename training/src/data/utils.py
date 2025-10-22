import numpy as np
import pandas as pd
from typing import Sequence

def fill_na(
    df: pd.DataFrame,
    *,
    num_strategy: str = "median",
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
