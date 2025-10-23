"""Unit tests for Concept Drift Detection."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.detection.concept_drift import ConceptDriftDetector
from src.config.settings import Settings


@pytest.mark.unit
class TestConceptDriftDetector:
    """Test suite for ConceptDriftDetector class."""

    def test_initialization(self, test_settings):
        """Test detector initialization."""
        detector = ConceptDriftDetector(test_settings)
        
        assert detector.settings == test_settings
        assert detector.recall_threshold == test_settings.drift_recall_threshold
        assert detector.fpr_threshold == test_settings.drift_fpr_threshold

    def test_compute_metrics(self, predictions_and_labels, test_settings):
        """Test metrics computation."""
        detector = ConceptDriftDetector(test_settings)
        y_true, y_pred = predictions_and_labels
        
        metrics = detector.compute_metrics(y_true, y_pred)
        
        assert 'recall' in metrics
        assert 'precision' in metrics
        assert 'fpr' in metrics
        assert 'f1_score' in metrics
        
        # Check valid ranges
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['fpr'] <= 1
        assert 0 <= metrics['f1_score'] <= 1

    def test_no_concept_drift(self, predictions_and_labels, test_settings):
        """Test when no concept drift exists."""
        detector = ConceptDriftDetector(test_settings)
        y_true, y_pred = predictions_and_labels
        
        # Use same metrics for baseline and current
        results = detector.detect(
            baseline_y_true=y_true,
            baseline_y_pred=y_pred,
            current_y_true=y_true,
            current_y_pred=y_pred
        )
        
        assert results['drift_detected'] is False

    def test_concept_drift_detected(self, predictions_and_labels, degraded_predictions, test_settings):
        """Test when concept drift is detected."""
        detector = ConceptDriftDetector(test_settings)
        y_true_good, y_pred_good = predictions_and_labels
        y_true_bad, y_pred_bad = degraded_predictions
        
        results = detector.detect(
            baseline_y_true=y_true_good,
            baseline_y_pred=y_pred_good,
            current_y_true=y_true_bad,
            current_y_pred=y_pred_bad
        )
        
        assert results['drift_detected'] is True
        assert 'severity' in results

    def test_recall_degradation(self, test_settings):
        """Test detection of recall degradation."""
        detector = ConceptDriftDetector(test_settings)
        
        # Good baseline: high recall
        baseline_y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        baseline_y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # 100% recall
        
        # Current: lower recall
        current_y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        current_y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])  # 50% recall
        
        results = detector.detect(
            baseline_y_true, baseline_y_pred,
            current_y_true, current_y_pred
        )
        
        assert results['drift_detected'] is True
        assert results['metrics']['recall'] < results['metrics']['baseline_recall']

    def test_fpr_increase(self, test_settings):
        """Test detection of false positive rate increase."""
        detector = ConceptDriftDetector(test_settings)
        
        # Good baseline: low FPR
        baseline_y_true = np.array([0] * 100 + [1] * 10)
        baseline_y_pred = np.array([0] * 100 + [1] * 10)  # 0% FPR
        
        # Current: higher FPR
        current_y_true = np.array([0] * 100 + [1] * 10)
        current_y_pred = np.array([1] * 20 + [0] * 80 + [1] * 10)  # 20% FPR
        
        results = detector.detect(
            baseline_y_true, baseline_y_pred,
            current_y_true, current_y_pred
        )
        
        assert results['drift_detected'] is True
        assert results['metrics']['fpr'] > results['metrics']['baseline_fpr']

    def test_severity_classification(self, test_settings):
        """Test severity classification."""
        detector = ConceptDriftDetector(test_settings)
        
        # Moderate degradation
        baseline_y_true = np.array([1] * 100 + [0] * 900)
        baseline_y_pred = np.array([1] * 95 + [0] * 5 + [0] * 900)  # 95% recall
        
        current_y_true = np.array([1] * 100 + [0] * 900)
        current_y_pred = np.array([1] * 85 + [0] * 15 + [0] * 900)  # 85% recall
        
        results = detector.detect(
            baseline_y_true, baseline_y_pred,
            current_y_true, current_y_pred
        )
        
        assert results['severity'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

    def test_perfect_predictions(self, test_settings):
        """Test with perfect predictions."""
        detector = ConceptDriftDetector(test_settings)
        
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        
        metrics = detector.compute_metrics(y_true, y_pred)
        
        assert metrics['recall'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['fpr'] == 0.0
        assert metrics['f1_score'] == 1.0

    def test_all_false_negatives(self, test_settings):
        """Test when model misses all frauds."""
        detector = ConceptDriftDetector(test_settings)
        
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # Miss all frauds
        
        metrics = detector.compute_metrics(y_true, y_pred)
        
        assert metrics['recall'] == 0.0

    def test_all_false_positives(self, test_settings):
        """Test when model flags everything as fraud."""
        detector = ConceptDriftDetector(test_settings)
        
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # Flag everything
        
        metrics = detector.compute_metrics(y_true, y_pred)
        
        assert metrics['fpr'] == 1.0  # 100% false positive rate

    def test_no_positive_predictions(self, test_settings):
        """Test when model makes no positive predictions."""
        detector = ConceptDriftDetector(test_settings)
        
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        
        metrics = detector.compute_metrics(y_true, y_pred)
        
        assert metrics['precision'] == 0.0 or np.isnan(metrics['precision'])
        assert metrics['recall'] == 0.0

    def test_no_true_positives(self, test_settings):
        """Test when there are no actual positive cases."""
        detector = ConceptDriftDetector(test_settings)
        
        y_true = np.array([0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        
        metrics = detector.compute_metrics(y_true, y_pred)
        
        # Recall should be undefined or 0 when no true positives exist
        assert metrics['recall'] == 0.0 or np.isnan(metrics['recall'])

    def test_results_structure(self, predictions_and_labels, test_settings):
        """Test that results have expected structure."""
        detector = ConceptDriftDetector(test_settings)
        y_true, y_pred = predictions_and_labels
        
        results = detector.detect(y_true, y_pred, y_true, y_pred)
        
        required_keys = [
            'drift_detected',
            'severity',
            'metrics'
        ]
        
        for key in required_keys:
            assert key in results
        
        metric_keys = ['recall', 'precision', 'fpr', 'f1_score']
        for key in metric_keys:
            assert key in results['metrics']

    def test_empty_arrays(self, test_settings):
        """Test handling of empty arrays."""
        detector = ConceptDriftDetector(test_settings)
        
        with pytest.raises(Exception):
            detector.compute_metrics(np.array([]), np.array([]))

    def test_mismatched_lengths(self, test_settings):
        """Test handling of mismatched array lengths."""
        detector = ConceptDriftDetector(test_settings)
        
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1])
        
        with pytest.raises(Exception):
            detector.compute_metrics(y_true, y_pred)

    @patch('src.detection.concept_drift.logger')
    def test_logging_on_drift(self, mock_logger, predictions_and_labels, degraded_predictions, test_settings):
        """Test logging behavior on drift detection."""
        detector = ConceptDriftDetector(test_settings)
        y_true_good, y_pred_good = predictions_and_labels
        y_true_bad, y_pred_bad = degraded_predictions
        
        results = detector.detect(
            y_true_good, y_pred_good,
            y_true_bad, y_pred_bad
        )
        
        assert mock_logger.warning.called or mock_logger.error.called

    def test_f1_score_calculation(self, test_settings):
        """Test F1 score calculation accuracy."""
        detector = ConceptDriftDetector(test_settings)
        
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1])
        
        metrics = detector.compute_metrics(y_true, y_pred)
        
        # Manual calculation
        precision = 3 / 4  # 3 TP, 1 FP
        recall = 3 / 4     # 3 TP, 1 FN
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        
        assert abs(metrics['f1_score'] - expected_f1) < 0.01

    def test_slight_improvement_not_drift(self, test_settings):
        """Test that slight improvement doesn't trigger false drift."""
        detector = ConceptDriftDetector(test_settings)
        
        # Baseline
        baseline_y_true = np.array([1] * 100 + [0] * 900)
        baseline_y_pred = np.array([1] * 90 + [0] * 10 + [1] * 10 + [0] * 890)
        
        # Slightly better current
        current_y_true = np.array([1] * 100 + [0] * 900)
        current_y_pred = np.array([1] * 92 + [0] * 8 + [1] * 10 + [0] * 890)
        
        results = detector.detect(
            baseline_y_true, baseline_y_pred,
            current_y_true, current_y_pred
        )
        
        # Slight improvement shouldn't trigger drift alert
        if results['metrics']['recall'] > results['metrics']['baseline_recall']:
            assert results['drift_detected'] is False
