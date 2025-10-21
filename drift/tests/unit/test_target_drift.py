"""Unit tests for Target Drift Detection."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.detection.target_drift import TargetDriftDetector
from src.config.settings import Settings


@pytest.mark.unit
class TestTargetDriftDetector:
    """Test suite for TargetDriftDetector class."""

    def test_initialization(self, test_settings):
        """Test detector initialization."""
        detector = TargetDriftDetector(test_settings)
        
        assert detector.settings == test_settings
        assert detector.threshold == test_settings.drift_target_threshold

    def test_compute_fraud_rate(self, baseline_data, test_settings):
        """Test fraud rate computation."""
        detector = TargetDriftDetector(test_settings)
        
        fraud_rate = detector.compute_fraud_rate(baseline_data['Class'].values)
        
        assert isinstance(fraud_rate, float)
        assert 0 <= fraud_rate <= 1
        assert fraud_rate == baseline_data['Class'].mean()

    def test_no_target_drift(self, test_settings):
        """Test detection when no drift exists."""
        detector = TargetDriftDetector(test_settings)
        
        # Create data with same fraud rate (no drift)
        baseline = np.array([0] * 998 + [1] * 2)     # 0.2% fraud rate
        current = np.array([0] * 499 + [1, 0])       # 0.2% fraud rate (1/500 is close to 0.2%)
        
        results = detector.detect(baseline, current)
        
        # With same fraud rate, relative change should be minimal
        assert abs(results['relative_change']) < test_settings.drift_target_threshold

    def test_target_drift_detected(self, baseline_data, current_data_with_drift, test_settings):
        """Test detection when drift exists."""
        detector = TargetDriftDetector(test_settings)
        
        results = detector.detect(
            baseline_labels=baseline_data['Class'].values,
            current_labels=current_data_with_drift['Class'].values
        )
        
        assert results['drift_detected'] is True
        assert abs(results['relative_change']) > test_settings.drift_target_threshold
        assert 'severity' in results

    def test_fraud_rate_increase(self, test_settings):
        """Test detection of fraud rate increase."""
        detector = TargetDriftDetector(test_settings)
        
        baseline = np.array([0] * 998 + [1] * 2)  # 0.2% fraud rate
        current = np.array([0] * 995 + [1] * 5)   # 0.5% fraud rate
        
        results = detector.detect(baseline, current)
        
        assert results['current_fraud_rate'] > results['baseline_fraud_rate']
        assert results['relative_change'] > 0

    def test_fraud_rate_decrease(self, test_settings):
        """Test detection of fraud rate decrease."""
        detector = TargetDriftDetector(test_settings)
        
        baseline = np.array([0] * 995 + [1] * 5)  # 0.5% fraud rate
        current = np.array([0] * 998 + [1] * 2)   # 0.2% fraud rate
        
        results = detector.detect(baseline, current)
        
        assert results['current_fraud_rate'] < results['baseline_fraud_rate']
        assert results['relative_change'] < 0

    def test_severity_classification_low(self, test_settings):
        """Test severity classification for low drift."""
        detector = TargetDriftDetector(test_settings)
        
        baseline = np.array([0] * 998 + [1] * 2)
        current = np.array([0] * 997 + [1] * 3)  # Slight increase
        
        results = detector.detect(baseline, current)
        
        if results['drift_detected']:
            assert results['severity'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

    def test_severity_classification_high(self, test_settings):
        """Test severity classification for high drift."""
        detector = TargetDriftDetector(test_settings)
        
        baseline = np.array([0] * 998 + [1] * 2)   # 0.2%
        current = np.array([0] * 990 + [1] * 10)   # 1.0% - 5x increase
        
        results = detector.detect(baseline, current)
        
        assert results['drift_detected'] is True
        assert results['severity'] in ['HIGH', 'CRITICAL']

    def test_empty_labels(self, test_settings):
        """Test handling of empty label arrays."""
        detector = TargetDriftDetector(test_settings)
        
        with pytest.raises(Exception):
            detector.detect(np.array([]), np.array([1, 0, 1]))

    def test_all_zeros(self, test_settings):
        """Test handling when all labels are 0 (no fraud)."""
        detector = TargetDriftDetector(test_settings)
        
        baseline = np.zeros(1000)
        current = np.zeros(500)
        
        results = detector.detect(baseline, current)
        
        assert results['baseline_fraud_rate'] == 0.0
        assert results['current_fraud_rate'] == 0.0
        assert results['drift_detected'] is False

    def test_all_ones(self, test_settings):
        """Test handling when all labels are 1 (all fraud)."""
        detector = TargetDriftDetector(test_settings)
        
        baseline = np.ones(1000)
        current = np.ones(500)
        
        results = detector.detect(baseline, current)
        
        assert results['baseline_fraud_rate'] == 1.0
        assert results['current_fraud_rate'] == 1.0
        assert results['drift_detected'] is False

    def test_baseline_zero_fraud(self, test_settings):
        """Test when baseline has zero fraud but current has some."""
        detector = TargetDriftDetector(test_settings)
        
        baseline = np.zeros(1000)
        current = np.array([0] * 995 + [1] * 5)
        
        results = detector.detect(baseline, current)
        
        # Should detect drift from 0% to 0.5%
        assert results['current_fraud_rate'] > 0
        assert results['baseline_fraud_rate'] == 0

    def test_relative_change_calculation(self, test_settings):
        """Test relative change calculation accuracy."""
        detector = TargetDriftDetector(test_settings)
        
        baseline = np.array([0] * 998 + [1] * 2)  # 0.2%
        current = np.array([0] * 996 + [1] * 4)   # 0.4% - 100% increase
        
        results = detector.detect(baseline, current)
        
        expected_change = (0.004 - 0.002) / 0.002  # 100% increase
        assert abs(results['relative_change'] - expected_change) < 0.01

    def test_small_sample_size(self, test_settings):
        """Test with small sample sizes."""
        detector = TargetDriftDetector(test_settings)
        
        baseline = np.array([0, 1, 0, 1, 0])
        current = np.array([0, 1, 1])
        
        results = detector.detect(baseline, current)
        
        assert 'drift_detected' in results
        assert 'current_fraud_rate' in results

    def test_results_structure(self, baseline_data, current_data_with_drift, test_settings):
        """Test that results have expected structure."""
        detector = TargetDriftDetector(test_settings)
        
        results = detector.detect(
            baseline_data['Class'].values,
            current_data_with_drift['Class'].values
        )
        
        required_keys = [
            'drift_detected',
            'current_fraud_rate',
            'baseline_fraud_rate',
            'relative_change',
            'severity'
        ]
        
        for key in required_keys:
            assert key in results

    @patch('src.detection.target_drift.logger')
    def test_logging_on_detection(self, mock_logger, baseline_data, current_data_with_drift, test_settings):
        """Test logging behavior."""
        detector = TargetDriftDetector(test_settings)
        
        results = detector.detect(
            baseline_data['Class'].values,
            current_data_with_drift['Class'].values
        )
        
        # Verify some logging occurred
        assert mock_logger.info.called or mock_logger.warning.called

    def test_extreme_drift(self, test_settings):
        """Test handling of extreme drift scenarios."""
        detector = TargetDriftDetector(test_settings)
        
        baseline = np.array([0] * 999 + [1])       # 0.1%
        current = np.array([0] * 500 + [1] * 500)  # 50% - extreme increase
        
        results = detector.detect(baseline, current)
        
        assert results['drift_detected'] is True
        assert results['severity'] == 'CRITICAL'
        assert results['relative_change'] > 10  # Over 1000% increase
