"""
ADWIN (Adaptive Windowing) Algorithm for Drift Detection.

This module implements the ADWIN algorithm for detecting changes in data streams
by maintaining an adaptive window of recent observations.

Reference: Bifet, A., & Gavaldà, R. (2007). Learning from time-changing data with adaptive windowing.
"""

import numpy as np
from typing import List, Tuple, Optional
import structlog

logger = structlog.get_logger(__name__)


class ADWIN:
    """
    Adaptive Windowing algorithm for drift detection.
    
    ADWIN maintains a variable-length window of recent observations and detects
    when the mean of two sub-windows differs significantly, indicating a change point.
    
    Attributes:
        delta: Confidence parameter (0 < delta < 1). Lower values = more sensitive.
        window: List of recent observations
        total: Sum of all values in window
        variance: Running variance estimate
        width: Current window size
        mint: Minimum window size before detection
    """
    
    def __init__(self, delta: float = 0.002, mint: int = 10):
        """
        Initialize ADWIN detector.
        
        Args:
            delta: Confidence parameter (default: 0.002, ~99.8% confidence)
            mint: Minimum window size before detection (default: 10)
        """
        self.delta = delta
        self.mint = mint
        self.window: List[float] = []
        self.total: float = 0.0
        self.variance: float = 0.0
        self.width: int = 0
        self.change_detected: bool = False
        self.estimation: float = 0.0
        
        logger.info("adwin_initialized", delta=delta, mint=mint)
    
    def add_element(self, value: float) -> bool:
        """
        Add a new element to the window and check for drift.
        
        Args:
            value: New observation value
            
        Returns:
            True if drift detected, False otherwise
        """
        self.window.append(value)
        self.width += 1
        self.total += value
        
        # Update running mean
        if self.width > 0:
            self.estimation = self.total / self.width
        
        # Update variance
        if self.width > 1:
            variance_sum = sum((x - self.estimation) ** 2 for x in self.window)
            self.variance = variance_sum / self.width
        
        # Check for change if window is large enough
        self.change_detected = False
        if self.width >= self.mint:
            self.change_detected = self._detect_change()
            
            if self.change_detected:
                logger.warning(
                    "adwin_drift_detected",
                    window_size=self.width,
                    mean=self.estimation,
                    variance=self.variance,
                    value=value
                )
        
        return self.change_detected
    
    def _detect_change(self) -> bool:
        """
        Check if there's a significant change between sub-windows.
        
        Returns:
            True if change detected, False otherwise
        """
        n = self.width
        
        # Try different split points
        for i in range(1, n):
            # Split window at position i
            window_0 = self.window[:i]
            window_1 = self.window[i:]
            
            n0 = len(window_0)
            n1 = len(window_1)
            
            if n0 < self.mint or n1 < self.mint:
                continue
            
            # Compute means
            mean_0 = np.mean(window_0)
            mean_1 = np.mean(window_1)
            
            # Compute difference
            diff = abs(mean_0 - mean_1)
            
            # Compute threshold using Hoeffding bound
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = np.sqrt((2.0 / m) * np.log(2.0 / self.delta))
            
            # Check if difference exceeds threshold
            if diff > epsilon:
                # Remove old data before split point
                self._compress_window(i)
                return True
        
        return False
    
    def _compress_window(self, cut_point: int) -> None:
        """
        Remove old data from the window up to the cut point.
        
        Args:
            cut_point: Index where to cut the window
        """
        # Keep only recent data
        self.window = self.window[cut_point:]
        self.width = len(self.window)
        
        # Recalculate statistics
        if self.width > 0:
            self.total = sum(self.window)
            self.estimation = self.total / self.width
            
            if self.width > 1:
                variance_sum = sum((x - self.estimation) ** 2 for x in self.window)
                self.variance = variance_sum / self.width
        else:
            self.total = 0.0
            self.estimation = 0.0
            self.variance = 0.0
        
        logger.info(
            "adwin_window_compressed",
            cut_point=cut_point,
            new_size=self.width,
            new_mean=self.estimation
        )
    
    def detected_change(self) -> bool:
        """
        Check if drift was detected in the last added element.
        
        Returns:
            True if drift detected, False otherwise
        """
        return self.change_detected
    
    def reset(self) -> None:
        """Reset the detector to initial state."""
        self.window = []
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.change_detected = False
        self.estimation = 0.0
        
        logger.info("adwin_reset")
    
    def get_estimation(self) -> float:
        """
        Get the current mean estimate.
        
        Returns:
            Current mean of the window
        """
        return self.estimation
    
    def get_window_size(self) -> int:
        """
        Get the current window size.
        
        Returns:
            Number of elements in the window
        """
        return self.width
    
    def get_variance(self) -> float:
        """
        Get the current variance estimate.
        
        Returns:
            Current variance of the window
        """
        return self.variance


def detect_drift_with_adwin(
    data_stream: List[float],
    delta: float = 0.002,
    mint: int = 10
) -> Tuple[List[int], List[float]]:
    """
    Detect drift points in a data stream using ADWIN.
    
    Args:
        data_stream: List of observations
        delta: Confidence parameter
        mint: Minimum window size
        
    Returns:
        Tuple of (drift_indices, drift_values) where drift was detected
    """
    detector = ADWIN(delta=delta, mint=mint)
    drift_indices: List[int] = []
    drift_values: List[float] = []
    
    for i, value in enumerate(data_stream):
        if detector.add_element(value):
            drift_indices.append(i)
            drift_values.append(value)
    
    logger.info(
        "adwin_detection_complete",
        stream_length=len(data_stream),
        drifts_detected=len(drift_indices)
    )
    
    return drift_indices, drift_values
