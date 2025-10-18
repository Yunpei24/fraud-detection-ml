"""
Population Stability Index (PSI) utilities.

- Computes PSI per feature between a reference (training/baseline) dataset
  and a current (production/batch) dataset.
- Uses quantile bins derived from the reference distribution (standard practice).
- Handles zero counts with an epsilon to avoid log(0).
- Outputs a per-feature report and an overall summary.

Typical usage:
    python -m src.drift.psi \
        --ref data/processed/train.parquet \
        --cur data/processed/live_batch.parquet \
        --out data/interim/psi_report.csv \
        --bins 10 --threshold 0.2
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


EPS = 1e-6  # to avoid log(0)


def _make_bins_from_reference(ref: pd.Series, bins: int) -> np.ndarray:
    """Create monotonic quantile bin edges from the reference series."""
    # quantiles from 0..1 inclusive
    qs = np.linspace(0, 1, bins + 1)
    edges = ref.quantile(qs, interpolation="linear").values
    # ensure strictly increasing (deduplicate)
    edges = np.unique(edges)
    # if ref is constant, expand tiny epsilon around the constant value
    if edges.size == 1:
        v = edges[0]
        edges = np.array([v - 1e-9, v + 1e-9])
    return edges


def _percent_in_bins(s: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Return percentage per bin given edges, with epsilon clipping."""
    # cut rightmost inclusive to ensure coverage
    labels = range(len(edges) - 1)
    b = pd.cut(s, bins=edges, labels=labels, include_lowest=True, right=True)
    counts = b.value_counts(sort=False).reindex(labels, fill_value=0).astype(float)
    pct = counts / max(len(s), 1.0)
    # smooth to avoid zero
    pct = np.clip(pct.values, EPS, None)
    # renormalize after clipping
    pct = pct / pct.sum()
    return pct


def psi_for_series(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """
    Compute PSI for a single numeric feature.
    PSI = sum( (cur_pct - ref_pct) * ln(cur_pct / ref_pct) ) over bins.
    """
    edges = _make_bins_from_reference(ref.dropna(), bins=bins)
    ref_pct = _percent_in_bins(ref.dropna(), edges)
    cur_pct = _percent_in_bins(cur.dropna(), edges)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_psi_frame(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    features: Iterable[str],
    bins: int = 10,
) -> pd.DataFrame:
    """Compute PSI per feature and return a DataFrame with columns: feature, psi."""
    rows = []
    for f in features:
        if f not in df_ref.columns or f not in df_cur.columns:
            continue
        # skip non-numeric quickly
        if not (np.issubdtype(df_ref[f].dtype, np.number) and np.issubdtype(df_cur[f].dtype, np.number)):
            continue
        try:
            val = psi_for_series(df_ref[f], df_cur[f], bins=bins)
            rows.append({"feature": f, "psi": val})
        except Exception as e:
            rows.append({"feature": f, "psi": np.nan, "error": str(e)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def summarize_psi(df_psi: pd.DataFrame, threshold: float = 0.2) -> Dict[str, float | int]:
    """
    Provide quick summary counts for monitoring.
    Common rule of thumb:
      - PSI < 0.1 : no significant shift
      - 0.1 - 0.2 : slight shift
      - > 0.2     : significant shift (actionable)
    """
    s = df_psi["psi"].fillna(0.0)
    return {
        "n_features": int(len(s)),
        "mean_psi": float(s.mean()) if len(s) else 0.0,
        "max_psi": float(s.max()) if len(s) else 0.0,
        "n_above_threshold": int((s >= threshold).sum()),
        "threshold": float(threshold),
    }


def load_parquet(path: str) -> pd.DataFrame:
    # Keep engine implicit or set explicitly if you standardized on pyarrow/fastparquet
    return pd.read_parquet(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference parquet (e.g., training baseline)")
    ap.add_argument("--cur", required=True, help="Current parquet (e.g., recent prod batch)")
    ap.add_argument("--out", default="data/interim/psi_report.csv", help="Output CSV path for per-feature PSI")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.2, help="Alert threshold per feature")
    ap.add_argument("--features", nargs="*", default=None, help="Optional explicit feature list (defaults: all numeric except 'Class')")
    args = ap.parse_args()

    df_ref = load_parquet(args.ref)
    df_cur = load_parquet(args.cur)

    if args.features:
        features = args.features
    else:
        # all numeric features except the target
        features = [c for c in df_ref.select_dtypes(include=[np.number]).columns if c != "Class"]

    psi_df = compute_psi_frame(df_ref, df_cur, features, bins=args.bins)
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    psi_df.to_csv(args.out, index=False)

    summary = summarize_psi(psi_df, threshold=args.threshold)
    # also write a JSON sidecar for programmatic checks in CI/CD
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"PSI report saved -> {args.out}")
    print(f"Summary -> {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
