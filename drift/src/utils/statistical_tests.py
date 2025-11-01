"""
Statistical tests utilities for drift detection.
"""

from typing import Tuple

import numpy as np
from scipy import stats


def ks_test_2sample(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
    """
    Perform two-sample Kolmogorov-Smirnov test.

    Args:
        sample1: First sample
        sample2: Second sample

    Returns:
        Tuple of (statistic, p_value)
    """
    statistic, p_value = stats.ks_2samp(sample1, sample2)
    return float(statistic), float(p_value)


def chi_squared_test(observed: np.ndarray, expected: np.ndarray) -> Tuple[float, float]:
    """
    Perform chi-squared test.

    Args:
        observed: Observed frequencies
        expected: Expected frequencies

    Returns:
        Tuple of (statistic, p_value)
    """
    statistic, p_value = stats.chisquare(observed, expected)
    return float(statistic), float(p_value)


def compute_psi_score(
    current: np.ndarray, baseline: np.ndarray, bins: int = 10
) -> float:
    """
    Compute Population Stability Index (PSI).

    Args:
        current: Current distribution
        baseline: Baseline distribution
        bins: Number of bins for binning

    Returns:
        PSI score
    """
    # Create bins
    min_val = min(current.min(), baseline.min())
    max_val = max(current.max(), baseline.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Compute frequencies
    current_freq, _ = np.histogram(current, bins=bin_edges)
    baseline_freq, _ = np.histogram(baseline, bins=bin_edges)

    # Normalize
    current_freq = current_freq / len(current) + 1e-10
    baseline_freq = baseline_freq / len(baseline) + 1e-10

    # Compute PSI
    psi = np.sum((current_freq - baseline_freq) * np.log(current_freq / baseline_freq))

    return float(psi)


def z_score_anomaly(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect anomalies using Z-score method.

    Args:
        values: Array of values
        threshold: Z-score threshold

    Returns:
        Boolean array indicating anomalies
    """
    mean = np.mean(values)
    std = np.std(values)

    z_scores = np.abs((values - mean) / (std + 1e-10))
    anomalies = z_scores > threshold

    return anomalies
