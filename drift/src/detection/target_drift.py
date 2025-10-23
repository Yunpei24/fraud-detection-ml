"""
Target drift detection for fraud rate monitoring.

Detects changes in the target variable distribution (fraud rate).
"""
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ..config import get_logger, settings
from ..config.constants import TARGET_DRIFT_THRESHOLD, BASELINE_FRAUD_RATE
from ..utils import MetricsComputationException

logger = get_logger(__name__)


class TargetDriftDetector:
    """
    Detector for target variable drift (changes in fraud rate).
    
    Monitors:
    - Fraud rate changes
    - Label distribution shifts
    - Temporal patterns in fraud
    """
    
    def __init__(self, settings=None, baseline_fraud_rate: Optional[float] = None):
        """
        Initialize target drift detector.
        
        Args:
            settings: Settings object from config
            baseline_fraud_rate: Baseline fraud rate from training data
        """
        self.settings = settings
        self.threshold = getattr(settings, 'target_drift_threshold', 0.5) if settings else 0.5
        self.baseline_fraud_rate = baseline_fraud_rate or BASELINE_FRAUD_RATE
        self.logger = logger
    
    def compute_fraud_rate(self, labels: np.ndarray) -> float:
        """
        Compute fraud rate from labels.
        
        Args:
            labels: Array of binary labels (0=legitimate, 1=fraud)
            
        Returns:
            Fraud rate (proportion of fraudulent transactions)
        """
        try:
            if len(labels) == 0:
                return 0.0
            
            fraud_count = np.sum(labels == 1)
            total_count = len(labels)
            fraud_rate = fraud_count / total_count
            
            self.logger.debug(
                "Fraud rate computed",
                extra={
                    "fraud_count": int(fraud_count),
                    "total_count": total_count,
                    "fraud_rate": fraud_rate
                }
            )
            
            return float(fraud_rate)
            
        except Exception as e:
            self.logger.error(f"Fraud rate computation failed: {e}")
            raise MetricsComputationException(
                "Fraud rate computation failed",
                details={"error": str(e)}
            )
    
    def compare_with_baseline(
        self,
        current_labels: np.ndarray
    ) -> Dict:
        """
        Compare current fraud rate with baseline.
        
        Args:
            current_labels: Current production labels
            
        Returns:
            Dictionary with comparison results
        """
        try:
            current_fraud_rate = self.compute_fraud_rate(current_labels)
            
            # Absolute change
            absolute_change = current_fraud_rate - self.baseline_fraud_rate
            
            # Relative change
            if self.baseline_fraud_rate > 0:
                relative_change = absolute_change / self.baseline_fraud_rate
            else:
                relative_change = 0.0 if current_fraud_rate == 0 else float('inf')
            
            result = {
                "current_fraud_rate": float(current_fraud_rate),
                "baseline_fraud_rate": float(self.baseline_fraud_rate),
                "absolute_change": float(absolute_change),
                "relative_change": float(relative_change),
                "direction": "increase" if absolute_change > 0 else "decrease"
            }
            
            self.logger.info(
                "Fraud rate comparison completed",
                extra=result
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Baseline comparison failed: {e}")
            raise MetricsComputationException(
                "Baseline comparison failed",
                details={"error": str(e)}
            )
    
    def detect(
        self,
        baseline_labels: np.ndarray,
        current_labels: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Detect target drift by comparing current and baseline labels.
        
        Main entry point for target drift detection.
        
        Args:
            baseline_labels: Baseline training labels
            current_labels: Current production labels
            threshold: Relative change threshold (default from settings)
            
        Returns:
            Dictionary with drift detection results
            
        Raises:
            ValueError: If baseline_labels or current_labels are empty
        """
        if len(baseline_labels) == 0 or len(current_labels) == 0:
            raise ValueError("baseline_labels and current_labels cannot be empty")
        
        # Update baseline from provided data
        self.baseline_fraud_rate = self.compute_fraud_rate(baseline_labels)
        
        # Call detect_shift with new baseline
        return self.detect_shift(current_labels, threshold)
    
    def detect_shift(
        self,
        current_labels: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Detect if target distribution has shifted significantly.
        
        Args:
            current_labels: Current production labels
            threshold: Relative change threshold (default from settings)
            
        Returns:
            Dictionary with drift detection results
        """
        if threshold is None:
            threshold = getattr(self.settings, 'target_drift_threshold', 0.5) if self.settings else 0.5
        
        try:
            # Compare with baseline
            comparison = self.compare_with_baseline(current_labels)
            
            # Determine if drift detected
            drift_detected = abs(comparison["relative_change"]) > threshold
            
            # Severity assessment
            if abs(comparison["relative_change"]) > 2 * threshold:
                severity = "CRITICAL"
            elif abs(comparison["relative_change"]) > threshold:
                severity = "HIGH"
            elif abs(comparison["relative_change"]) > threshold * 0.5:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            
            result = {
                "drift_detected": drift_detected,
                "severity": severity,
                "threshold": threshold,
                **comparison
            }
            
            self.logger.info(
                "Target drift detection completed",
                extra={
                    "drift_detected": drift_detected,
                    "severity": severity,
                    "relative_change": comparison["relative_change"]
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Target drift detection failed: {e}")
            raise MetricsComputationException(
                "Target drift detection failed",
                details={"error": str(e)}
            )
    
    def chi_squared_test(
        self,
        current_labels: np.ndarray,
        baseline_labels: np.ndarray
    ) -> Dict:
        """
        Perform chi-squared test on label distributions.
        
        Args:
            current_labels: Current production labels
            baseline_labels: Baseline training labels
            
        Returns:
            Test results
        """
        try:
            # Create contingency table
            current_counts = pd.Series(current_labels).value_counts()
            baseline_counts = pd.Series(baseline_labels).value_counts()
            
            # Align indices
            all_labels = set(current_counts.index) | set(baseline_counts.index)
            current_aligned = current_counts.reindex(all_labels, fill_value=0)
            baseline_aligned = baseline_counts.reindex(all_labels, fill_value=0)
            
            contingency_table = pd.DataFrame({
                "current": current_aligned,
                "baseline": baseline_aligned
            }).T
            
            # Perform test
            statistic, p_value, dof, expected = chi2_contingency(contingency_table)
            drift_detected = p_value < 0.05
            
            result = {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": drift_detected
            }
            
            self.logger.debug(
                "Chi-squared test completed",
                extra=result
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Chi-squared test failed: {e}")
            raise MetricsComputationException(
                "Chi-squared test failed",
                details={"error": str(e)}
            )
    
    def analyze_temporal_pattern(
        self,
        labels: np.ndarray,
        timestamps: List[str],
        window_size: str = "1H"
    ) -> pd.DataFrame:
        """
        Analyze temporal patterns in fraud rate.
        
        Args:
            labels: Array of labels
            timestamps: List of timestamps
            window_size: Time window for aggregation (e.g., '1H', '1D')
            
        Returns:
            DataFrame with fraud rate over time
        """
        try:
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(timestamps),
                "label": labels
            })
            
            # Resample by time window
            df.set_index("timestamp", inplace=True)
            fraud_rate_timeline = df.resample(window_size).agg({
                "label": ["sum", "count", "mean"]
            })
            
            fraud_rate_timeline.columns = ["fraud_count", "total_count", "fraud_rate"]
            
            self.logger.info(
                f"Temporal analysis completed with {len(fraud_rate_timeline)} time windows"
            )
            
            return fraud_rate_timeline
            
        except Exception as e:
            self.logger.error(f"Temporal analysis failed: {e}")
            raise MetricsComputationException(
                "Temporal analysis failed",
                details={"error": str(e)}
            )
    
    def detect_anomalous_spikes(
        self,
        fraud_rate_timeline: pd.DataFrame,
        std_threshold: float = 3.0
    ) -> List[str]:
        """
        Detect anomalous spikes in fraud rate.
        
        Args:
            fraud_rate_timeline: DataFrame with fraud rate over time
            std_threshold: Number of standard deviations for anomaly detection
            
        Returns:
            List of timestamps with anomalous spikes
        """
        try:
            mean_fraud_rate = fraud_rate_timeline["fraud_rate"].mean()
            std_fraud_rate = fraud_rate_timeline["fraud_rate"].std()
            
            # Detect outliers
            z_scores = (fraud_rate_timeline["fraud_rate"] - mean_fraud_rate) / std_fraud_rate
            anomalies = fraud_rate_timeline[abs(z_scores) > std_threshold]
            
            anomaly_timestamps = [
                str(ts) for ts in anomalies.index
            ]
            
            self.logger.info(
                f"Detected {len(anomaly_timestamps)} anomalous fraud rate spikes"
            )
            
            return anomaly_timestamps
            
        except Exception as e:
            self.logger.error(f"Spike detection failed: {e}")
            return []
