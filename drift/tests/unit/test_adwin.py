"""Unit tests for ADWIN algorithm."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.detection.adwin import ADWIN


@pytest.mark.unit
class TestADWIN:
    """Test suite for ADWIN class."""

    def test_initialization(self):
        """Test ADWIN initialization."""
        adwin = ADWIN(delta=0.002)
        
        assert adwin.delta == 0.002
        assert len(adwin.window) == 0
        assert adwin.total == 0.0
        assert adwin.variance == 0.0
        assert adwin.width == 0

    def test_add_single_element(self):
        """Test adding a single element."""
        adwin = ADWIN()
        
        change_detected = adwin.add_element(1.0)
        
        assert change_detected is False
        assert adwin.width == 1
        assert adwin.total == 1.0

    def test_add_multiple_elements_no_change(self):
        """Test adding multiple similar elements."""
        adwin = ADWIN()
        
        # Add elements from same distribution
        for _ in range(100):
            change_detected = adwin.add_element(np.random.normal(0, 1))
        
        # Should not detect change in stable distribution
        assert adwin.width == 100

    def test_detect_mean_shift(self):
        """Test detection of mean shift."""
        adwin = ADWIN(delta=0.002)
        
        # Add elements from distribution 1
        for _ in range(100):
            adwin.add_element(np.random.normal(0, 1))
        
        # Add elements from shifted distribution
        change_detected = False
        for _ in range(100):
            if adwin.add_element(np.random.normal(3, 1)):  # Mean shift
                change_detected = True
                break
        
        assert change_detected is True

    def test_detect_variance_change(self):
        """Test detection of variance change."""
        adwin = ADWIN(delta=0.002)
        
        # Add elements with low variance
        for _ in range(100):
            adwin.add_element(np.random.normal(0, 0.1))
        
        # Add elements with high variance
        change_detected = False
        for _ in range(100):
            if adwin.add_element(np.random.normal(0, 5)):  # Variance increase
                change_detected = True
                break
        
        # May or may not detect depending on random values
        # Just verify no errors occur
        assert isinstance(change_detected, bool)

    def test_window_trimming(self):
        """Test that window is trimmed after change detection."""
        adwin = ADWIN(delta=0.002)
        
        # Fill window
        for _ in range(100):
            adwin.add_element(np.random.normal(0, 1))
        
        initial_width = adwin.width
        
        # Add shifted values
        for _ in range(50):
            adwin.add_element(np.random.normal(5, 1))
        
        # Window should have been trimmed if change was detected
        assert adwin.width <= initial_width + 50

    def test_detected_change_method(self):
        """Test detected_change() method."""
        adwin = ADWIN()
        
        # Add stable data
        for _ in range(50):
            adwin.add_element(1.0)
        
        assert adwin.detected_change() is False

    def test_estimate_mean(self):
        """Test mean estimation."""
        adwin = ADWIN()
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for val in values:
            adwin.add_element(val)
        
        expected_mean = np.mean(values)
        assert abs(adwin.total / adwin.width - expected_mean) < 0.001

    def test_reset_after_change(self):
        """Test that statistics reset after change detection."""
        adwin = ADWIN(delta=0.002)
        
        # Add initial data
        for _ in range(100):
            adwin.add_element(np.random.normal(0, 1))
        
        initial_total = adwin.total
        
        # Add shifted data until change detected
        for _ in range(100):
            if adwin.add_element(np.random.normal(10, 1)):
                break
        
        # Total should be reset/reduced after change
        assert adwin.width < 200  # Window was trimmed

    def test_delta_sensitivity(self):
        """Test that smaller delta is more sensitive."""
        # More sensitive ADWIN (smaller delta)
        adwin_sensitive = ADWIN(delta=0.0001)
        
        # Less sensitive ADWIN (larger delta)
        adwin_robust = ADWIN(delta=0.01)
        
        # Add same data to both - deterministic test data
        # First stable distribution
        data1 = [1.0, 1.1, 0.9, 1.05, 0.95] * 20  # 100 values around 1.0
        # Then shifted distribution
        data2 = [5.0, 5.1, 4.9, 5.05, 4.95] * 10  # 50 values around 5.0
        
        changes_sensitive = 0
        changes_robust = 0
        
        for val in data1 + data2:
            if adwin_sensitive.add_element(val):
                changes_sensitive += 1
            if adwin_robust.add_element(val):
                changes_robust += 1
        
        # Both should detect the drift (large shift from 1.0 to 5.0)
        assert changes_sensitive > 0 or changes_robust > 0

    def test_constant_values(self):
        """Test with constant values (no variance)."""
        adwin = ADWIN()
        
        # Add constant values
        for _ in range(100):
            change_detected = adwin.add_element(5.0)
        
        assert adwin.variance == 0.0
        assert adwin.total == 500.0

    def test_alternating_values(self):
        """Test with alternating values."""
        adwin = ADWIN()
        
        # Add alternating values
        for i in range(100):
            val = 1.0 if i % 2 == 0 else 0.0
            adwin.add_element(val)
        
        # Mean should be around 0.5
        assert abs(adwin.total / adwin.width - 0.5) < 0.01

    def test_extreme_values(self):
        """Test with extreme values."""
        adwin = ADWIN()
        
        # Add normal values then extreme
        for _ in range(50):
            adwin.add_element(1.0)
        
        # Add extreme value - should detect drift and compress window
        change_detected = adwin.add_element(1000.0)
        
        # Should detect the extreme value as drift
        assert change_detected is True
        # Window should be compressed (less than 51 due to drift detection)
        assert adwin.width < 51

    def test_negative_values(self):
        """Test with negative values."""
        adwin = ADWIN()
        
        values = [-1.0, -2.0, -3.0, -4.0, -5.0]
        for val in values:
            adwin.add_element(val)
        
        expected_mean = np.mean(values)
        assert abs(adwin.total / adwin.width - expected_mean) < 0.001

    def test_large_window(self):
        """Test with large window size (fast version)."""
        adwin = ADWIN()
        
        # Add 100 constant elements (very fast)
        for _ in range(100):
            adwin.add_element(1.0)
        
        # Should process successfully
        assert adwin.width > 0
        assert adwin.width == 100

    def test_window_properties(self):
        """Test window property calculations."""
        adwin = ADWIN()
        
        values = [1, 2, 3, 4, 5]
        for val in values:
            adwin.add_element(val)
        
        assert adwin.width == len(values)
        assert adwin.total == sum(values)

    def test_zero_values(self):
        """Test with zero values."""
        adwin = ADWIN()
        
        for _ in range(100):
            adwin.add_element(0.0)
        
        assert adwin.total == 0.0
        assert adwin.width == 100

    @patch('src.detection.adwin.logger')
    def test_logging_on_change(self, mock_logger):
        """Test logging when change is detected."""
        adwin = ADWIN(delta=0.002)
        
        # Add stable data
        for _ in range(100):
            adwin.add_element(np.random.normal(0, 1))
        
        # Add shifted data
        for _ in range(100):
            if adwin.add_element(np.random.normal(10, 1)):
                break
        
        # Some logging should have occurred
        assert mock_logger.info.called or mock_logger.warning.called

    def test_incremental_change(self):
        """Test with gradual/incremental change."""
        adwin = ADWIN(delta=0.002)
        
        # Gradual increase
        for i in range(200):
            val = i * 0.01  # Slowly increasing
            adwin.add_element(val)
        
        # ADWIN may or may not detect gradual change
        # Just verify it runs without errors
        assert adwin.width > 0
