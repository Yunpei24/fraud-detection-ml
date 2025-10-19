
from __future__ import annotations
import numpy as np
from typing import Literal, Tuple

def _make_bins(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int,
    strategy: Literal["quantile", "uniform"] = "quantile",
) -> np.ndarray:
    """Create bin edges shared by expected and actual."""
    x = np.asarray(expected).astype(float)
    y = np.asarray(actual).astype(float)

    if strategy == "quantile":
        # Quantile edges on expected, then unique to avoid duplicates
        qs = np.linspace(0, 1, bins + 1)
        edges = np.quantile(x, qs)
        edges = np.unique(edges)
        if len(edges) - 1 < bins:
            # fallback to uniform if too many ties
            mn = min(x.min(), y.min())
            mx = max(x.max(), y.max())
            edges = np.linspace(mn, mx, bins + 1)
    elif strategy == "uniform":
        mn = min(x.min(), y.min())
        mx = max(x.max(), y.max())
        edges = np.linspace(mn, mx, bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    # If all values are identical, widen edges minimally
    if np.allclose(edges[0], edges[-1]):
        edges = np.array([edges[0] - 0.5, edges[-1] + 0.5])

    return edges

def _hist_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram to proportions with safe zero handling."""
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        # all zero -> return uniform tiny proportions to avoid div-by-zero later
        return np.full_like(counts, 1.0 / len(counts), dtype=float)
    return counts.astype(float) / float(total)

def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
    epsilon: float = 1e-12,
) -> float:
    """
    Population Stability Index (PSI) between expected (baseline) and actual (current).

    PSI = sum over bins of (p_i - q_i) * ln(p_i / q_i),
    where p_i is expected proportion, q_i is actual proportion.

    Parameters
    ----------
    expected : array-like
        Reference (training or baseline) sample.
    actual : array-like
        Current (production) sample.
    bins : int, default=10
        Number of bins.
    strategy : {"quantile","uniform"}, default="quantile"
        How to create shared bin edges.
    epsilon : float, default=1e-12
        Small value to avoid log(0) or division by zero.

    Returns
    -------
    float
        PSI value (0 to inf). Rough rule of thumb:
          < 0.1  : no shift
          0.1-0.2: small shift
          > 0.2  : significant shift
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = _make_bins(expected, actual, bins=bins, strategy=strategy)
    p = _hist_proportions(expected, edges)
    q = _hist_proportions(actual, edges)

    # numerical safety
    p_safe = np.clip(p, epsilon, 1.0)
    q_safe = np.clip(q, epsilon, 1.0)

    return float(np.sum((p_safe - q_safe) * np.log(p_safe / q_safe)))
