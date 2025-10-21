"""Unit tests for Statistical Tests utilities."""

import pytest
import numpy as np
from scipy import stats

from src.utils.statistical_tests import (
    ks_test_2sample,
    chi_squared_test,
    compute_psi_score,
    z_score_anomaly
)


@pytest.mark.unit
class TestStatisticalTests:
    """Test suite for statistical test utilities."""

    def test_ks_test_identical_samples(self):
        """Test KS test with identical samples."""
        sample1 = np.random.normal(0, 1, 1000)
        sample2 = sample1.copy()
        
        statistic, p_value = ks_test_2sample(sample1, sample2)
        
        assert statistic == 0.0
        assert p_value == 1.0

    def test_ks_test_different_samples(self):
        """Test KS test with different distributions."""
        sample1 = np.random.normal(0, 1, 1000)
        sample2 = np.random.normal(5, 1, 1000)  # Different mean
        
        statistic, p_value = ks_test_2sample(sample1, sample2)
        
        assert statistic > 0
        assert p_value < 0.05  # Statistically significant difference

    def test_ks_test_similar_samples(self):
        """Test KS test with similar distributions."""
        np.random.seed(42)
        sample1 = np.random.normal(0, 1, 1000)
        np.random.seed(43)
        sample2 = np.random.normal(0, 1, 1000)
        
        statistic, p_value = ks_test_2sample(sample1, sample2)
        
        # Should not be significantly different
        assert p_value > 0.05

    def test_chi_squared_test_identical(self):
        """Test chi-squared test with identical distributions."""
        observed = np.array([10, 20, 30, 40])
        expected = np.array([10, 20, 30, 40])
        
        statistic, p_value = chi_squared_test(observed, expected)
        
        assert statistic == 0.0
        assert p_value == 1.0

    def test_chi_squared_test_different(self):
        """Test chi-squared test with different distributions."""
        observed = np.array([10, 20, 30, 40])
        expected = np.array([40, 30, 20, 10])
        
        statistic, p_value = chi_squared_test(observed, expected)
        
        assert statistic > 0
        assert p_value < 0.05

    def test_compute_psi_no_drift(self):
        """Test PSI computation with no drift."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        np.random.seed(43)
        actual = np.random.normal(0, 1, 1000)
        
        psi = compute_psi_score(expected, actual, bins=10)
        
        assert psi >= 0
        assert psi < 0.1  # Low PSI indicates no drift

    def test_compute_psi_with_drift(self):
        """Test PSI computation with drift."""
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(2, 1, 1000)  # Shifted distribution
        
        psi = compute_psi_score(expected, actual, bins=10)
        
        assert psi > 0.25  # High PSI indicates drift

    def test_compute_psi_different_bins(self):
        """Test PSI with different bin counts."""
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0.5, 1, 1000)
        
        psi_5 = compute_psi_score(expected, actual, bins=5)
        psi_10 = compute_psi_score(expected, actual, bins=10)
        psi_20 = compute_psi_score(expected, actual, bins=20)
        
        # All should detect drift
        assert psi_5 > 0
        assert psi_10 > 0
        assert psi_20 > 0

    def test_z_score_no_anomalies(self):
        """Test z-score anomaly detection with no anomalies."""
        data = np.random.normal(0, 1, 1000)
        
        anomalies = z_score_anomaly(data, threshold=3.0)
        
        # Should have very few anomalies (< 1%)
        assert np.sum(anomalies) < 10

    def test_z_score_with_anomalies(self):
        """Test z-score anomaly detection with outliers."""
        data = np.concatenate([
            np.random.normal(0, 1, 990),
            np.array([10, 10, 10, 10, 10, -10, -10, -10, -10, -10])  # Outliers
        ])
        
        anomalies = z_score_anomaly(data, threshold=3.0)
        
        assert np.sum(anomalies) > 0

    def test_z_score_threshold_sensitivity(self):
        """Test z-score threshold sensitivity."""
        data = np.random.normal(0, 1, 1000)
        
        anomalies_strict = z_score_anomaly(data, threshold=4.0)
        anomalies_lenient = z_score_anomaly(data, threshold=2.0)
        
        # Lenient threshold should find more anomalies
        assert np.sum(anomalies_lenient) >= np.sum(anomalies_strict)

    def test_ks_test_empty_arrays(self):
        """Test KS test with empty arrays."""
        with pytest.raises(Exception):
            ks_test_2sample(np.array([]), np.array([1, 2, 3]))

    def test_chi_squared_mismatched_sums(self):
        """Test chi-squared raises error when sum(observed) != sum(expected)."""
        observed = np.array([10, 20, 30])  # Sum = 60
        expected = np.array([5, 5, 5])     # Sum = 15 (different!)
        
        # scipy.stats.chisquare requires sum(observed) == sum(expected)
        # Should raise ValueError when sums don't match
        with pytest.raises(ValueError, match="sum of the observed"):
            chi_squared_test(observed, expected)

    def test_psi_single_value(self):
        """Test PSI with constant values."""
        expected = np.array([5.0] * 100)
        actual = np.array([5.0] * 100)
        
        psi = compute_psi_score(expected, actual, bins=10)
        
        # Should handle gracefully
        assert psi >= 0

    def test_z_score_constant_data(self):
        """Test z-score with constant data (zero variance)."""
        data = np.array([5.0] * 100)
        
        anomalies = z_score_anomaly(data, threshold=3.0)
        
        # No anomalies in constant data
        assert np.sum(anomalies) == 0

    def test_psi_small_samples(self):
        """Test PSI with small sample sizes."""
        expected = np.array([1, 2, 3, 4, 5])
        actual = np.array([1, 2, 3, 4, 6])
        
        psi = compute_psi_score(expected, actual, bins=3)
        
        assert psi >= 0

    def test_ks_test_different_sizes(self):
        """Test KS test with different sample sizes."""
        sample1 = np.random.normal(0, 1, 1000)
        sample2 = np.random.normal(0, 1, 500)
        
        statistic, p_value = ks_test_2sample(sample1, sample2)
        
        assert 0 <= statistic <= 1
        assert 0 <= p_value <= 1

    def test_chi_squared_mismatched_lengths(self):
        """Test chi-squared with mismatched array lengths."""
        observed = np.array([10, 20, 30])
        expected = np.array([10, 20])
        
        with pytest.raises(Exception):
            chi_squared_test(observed, expected)

    def test_psi_zero_bins_handling(self):
        """Test PSI handling of bins with zero counts."""
        # Sparse data
        expected = np.array([0] * 95 + [10] * 5)
        actual = np.array([0] * 98 + [10] * 2)
        
        psi = compute_psi_score(expected, actual, bins=10)
        
        assert not np.isnan(psi)
        assert psi >= 0

    def test_z_score_extreme_values(self):
        """Test z-score with extreme values."""
        data = np.concatenate([
            np.random.normal(0, 1, 100),
            np.array([1000])  # Extreme outlier
        ])
        
        anomalies = z_score_anomaly(data, threshold=3.0)
        
        # Should detect the extreme outlier
        assert np.sum(anomalies) > 0
        assert 1000 in data[anomalies]

    def test_ks_test_return_types(self):
        """Test that KS test returns correct types."""
        sample1 = np.random.normal(0, 1, 100)
        sample2 = np.random.normal(0, 1, 100)
        
        statistic, p_value = ks_test_2sample(sample1, sample2)
        
        assert isinstance(statistic, float)
        assert isinstance(p_value, float)

    def test_psi_return_type(self):
        """Test that PSI returns correct type."""
        expected = np.random.normal(0, 1, 100)
        actual = np.random.normal(0, 1, 100)
        
        psi = compute_psi_score(expected, actual, bins=10)
        
        assert isinstance(psi, float)

    def test_z_score_return_type(self):
        """Test that z-score returns correct type."""
        data = np.random.normal(0, 1, 100)
        
        anomalies = z_score_anomaly(data, threshold=3.0)
        
        assert isinstance(anomalies, np.ndarray)
