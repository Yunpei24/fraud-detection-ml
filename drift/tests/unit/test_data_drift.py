"""Unit tests for Data Drift Detection."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from src.detection.data_drift import DataDriftDetector
from src.config.constants import PSI_NO_CHANGE, PSI_MODERATE_CHANGE, PSI_SIGNIFICANT_CHANGE


@pytest.mark.unit
class TestDataDriftDetector:
    """Test suite for DataDriftDetector class."""

    def test_initialization_with_baseline(self, baseline_data):
        """Test detector initialization with baseline data."""
        detector = DataDriftDetector(baseline_data)
        
        assert detector.baseline_data is not None
        assert detector.baseline_stats is not None
        assert len(detector.baseline_stats) == len(baseline_data.columns)
        
    def test_initialization_without_baseline(self):
        """Test detector initialization without baseline data."""
        detector = DataDriftDetector()
        
        assert detector.baseline_data is None
        assert detector.baseline_stats is None

    def test_compute_baseline_stats_numeric(self):
        """Test baseline statistics computation for numeric features."""
        baseline = pd.DataFrame({
            'V1': np.random.normal(0, 1, 100),
            'V2': np.random.normal(5, 2, 100)
        })
        
        detector = DataDriftDetector(baseline)
        
        assert 'V1' in detector.baseline_stats
        assert 'V2' in detector.baseline_stats
        assert detector.baseline_stats['V1']['type'] == 'numeric'
        assert 'mean' in detector.baseline_stats['V1']
        assert 'std' in detector.baseline_stats['V1']
        assert 'min' in detector.baseline_stats['V1']
        assert 'max' in detector.baseline_stats['V1']

    def test_compute_psi_no_drift(self):
        """Test PSI computation with no drift."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        np.random.seed(43)
        current = np.random.normal(0, 1, 1000)
        
        detector = DataDriftDetector()
        psi_score = detector.compute_psi(current, baseline)
        
        assert isinstance(psi_score, float)
        assert psi_score >= 0
        assert psi_score < PSI_MODERATE_CHANGE  # No significant drift

    def test_compute_psi_with_drift(self):
        """Test PSI computation with significant drift."""
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(3, 1, 1000)  # Shifted distribution
        
        detector = DataDriftDetector()
        psi_score = detector.compute_psi(current, baseline)
        
        assert isinstance(psi_score, float)
        assert psi_score > PSI_SIGNIFICANT_CHANGE  # Significant drift

    def test_compute_psi_custom_bins(self):
        """Test PSI computation with custom bin count."""
        baseline = np.random.normal(0, 1, 500)
        current = np.random.normal(0.5, 1, 500)
        
        detector = DataDriftDetector()
        psi_5_bins = detector.compute_psi(current, baseline, bins=5)
        psi_20_bins = detector.compute_psi(current, baseline, bins=20)
        
        assert isinstance(psi_5_bins, float)
        assert isinstance(psi_20_bins, float)
        assert psi_5_bins >= 0
        assert psi_20_bins >= 0

    def test_ks_test_no_drift(self):
        """Test KS test with no drift."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 500)
        current = np.random.normal(0, 1, 500)
        
        detector = DataDriftDetector()
        statistic, p_value, drift_detected = detector.ks_test(current, baseline)
        
        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert isinstance(drift_detected, (bool, np.bool_))
        assert 0 <= statistic <= 1
        assert 0 <= p_value <= 1
        assert drift_detected == False

    def test_ks_test_with_drift(self):
        """Test KS test with drift."""
        baseline = np.random.normal(0, 1, 500)
        current = np.random.normal(2, 1, 500)  # Shifted distribution
        
        detector = DataDriftDetector()
        statistic, p_value, drift_detected = detector.ks_test(current, baseline)
        
        assert statistic > 0.3  # High KS statistic
        assert p_value < 0.05  # Statistically significant
        assert drift_detected == True

    def test_chi_squared_test_no_drift(self):
        """Test Chi-squared test for categorical features without drift."""
        baseline = pd.Series(['A'] * 100 + ['B'] * 100 + ['C'] * 100)
        current = pd.Series(['A'] * 95 + ['B'] * 105 + ['C'] * 100)
        
        detector = DataDriftDetector()
        statistic, p_value, drift_detected = detector.chi_squared_test(current, baseline)
        
        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert isinstance(drift_detected, (bool, np.bool_))
        assert statistic >= 0
        assert drift_detected == False

    def test_chi_squared_test_with_drift(self):
        """Test Chi-squared test with drift."""
        baseline = pd.Series(['A'] * 200 + ['B'] * 100)
        current = pd.Series(['A'] * 100 + ['B'] * 200)  # Reversed distribution
        
        detector = DataDriftDetector()
        statistic, p_value, drift_detected = detector.chi_squared_test(current, baseline)
        
        assert statistic > 10  # High chi-squared statistic
        assert p_value < 0.05  # Statistically significant
        assert drift_detected == True

    def test_detect_drift_no_drift(self, baseline_data, current_data_no_drift):
        """Test drift detection with no drift."""
        detector = DataDriftDetector(baseline_data)
        
        results = detector.detect_drift(current_data_no_drift)
        
        assert 'drift_detected' in results
        assert 'avg_psi' in results
        assert 'drifted_features' in results
        assert 'psi_scores' in results
        assert isinstance(results['drifted_features'], list)
        assert isinstance(results['psi_scores'], dict)
        assert results['drift_detected'] is False
        assert results['avg_psi'] < PSI_SIGNIFICANT_CHANGE

    def test_detect_drift_with_drift(self, baseline_data, current_data_with_drift):
        """Test drift detection with significant drift."""
        detector = DataDriftDetector(baseline_data)
        
        results = detector.detect_drift(current_data_with_drift)
        
        assert results['drift_detected'] is True
        assert len(results['drifted_features']) > 0
        assert results['avg_psi'] > PSI_NO_CHANGE
        assert 'num_features_drifted' in results
        assert results['num_features_drifted'] > 0

    def test_compute_drift_score_psi_method(self, baseline_data, current_data_no_drift):
        """Test drift score computation using PSI method."""
        detector = DataDriftDetector(baseline_data)
        
        scores = detector.compute_drift_score(current_data_no_drift, method="psi")
        
        assert isinstance(scores, dict)
        assert len(scores) > 0
        for feature, score in scores.items():
            assert isinstance(score, float)
            assert score >= 0

    def test_compute_drift_score_ks_method(self, baseline_data, current_data_no_drift):
        """Test drift score computation using KS method."""
        detector = DataDriftDetector(baseline_data)
        
        scores = detector.compute_drift_score(current_data_no_drift, method="ks")
        
        assert isinstance(scores, dict)
        assert len(scores) > 0
        for feature, score in scores.items():
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_detect_drift_without_baseline_raises_error(self):
        """Test that detect_drift raises error without baseline data."""
        detector = DataDriftDetector()
        current = pd.DataFrame({'V1': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            detector.compute_drift_score(current)

    def test_single_value_column_handling(self):
        """Test handling of columns with single value (no variance)."""
        detector = DataDriftDetector()
        
        baseline = np.array([1.0] * 100)
        current = np.array([1.0] * 50)
        
        psi_score = detector.compute_psi(current, baseline)
        
        # Should handle gracefully without errors
        assert isinstance(psi_score, float)
        assert not np.isnan(psi_score)

    def test_small_sample_handling(self):
        """Test PSI computation with small sample size."""
        detector = DataDriftDetector()
        
        baseline = np.random.normal(0, 1, 10)
        current = np.random.normal(0, 1, 5)
        
        psi_score = detector.compute_psi(current, baseline)
        
        assert isinstance(psi_score, float)
        assert psi_score >= 0

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values."""
        detector = DataDriftDetector()
        
        # Very large values
        baseline = np.array([1e10, 1e10 + 1, 1e10 + 2] * 30)
        current = np.array([1e10, 1e10 + 1, 1e10 + 3] * 30)
        
        psi_score = detector.compute_psi(current, baseline)
        
        assert not np.isnan(psi_score)
        assert not np.isinf(psi_score)
        assert psi_score >= 0

    def test_zero_bins_handling(self):
        """Test handling when PSI bins have zero counts."""
        detector = DataDriftDetector()
        
        # Sparse data that may create empty bins
        baseline = np.array([0] * 95 + [10] * 5)
        current = np.array([0] * 98 + [10] * 2)
        
        psi_score = detector.compute_psi(current, baseline, bins=10)
        
        assert not np.isnan(psi_score)
        assert not np.isinf(psi_score)
        assert psi_score >= 0

    def test_get_feature_importance_drift(self, baseline_data, current_data_no_drift):
        """Test drift computation for important features only."""
        detector = DataDriftDetector(baseline_data)
        
        # Mock feature importance
        feature_importance = {col: np.random.random() for col in baseline_data.columns}
        
        drift_scores = detector.get_feature_importance_drift(
            current_data_no_drift,
            feature_importance,
            top_n=5
        )
        
        assert isinstance(drift_scores, dict)
        assert len(drift_scores) <= 5  # At most top 5 features

    @patch('src.detection.data_drift.logger')
    def test_logging_on_drift_detection(self, mock_logger, baseline_data, current_data_with_drift):
        """Test that drift detection logs appropriately."""
        detector = DataDriftDetector(baseline_data)
        
        results = detector.detect_drift(current_data_with_drift)
        
        # Verify logging was called
        assert mock_logger.info.called
