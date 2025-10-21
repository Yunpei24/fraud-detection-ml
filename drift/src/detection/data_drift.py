"""
Data drift detection using statistical tests.

Detects changes in feature distributions between baseline (training) 
and production data using multiple statistical methods.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency

from ..config import get_logger, settings
from ..config.constants import (
    DATA_DRIFT_THRESHOLD,
    KS_TEST_ALPHA,
    PSI_NO_CHANGE,
    PSI_MODERATE_CHANGE,
    PSI_SIGNIFICANT_CHANGE
)
from ..utils import MetricsComputationException

logger = get_logger(__name__)


class DataDriftDetector:
    """
    Detector for data drift in feature distributions.
    
    Uses multiple statistical tests:
    - Kolmogorov-Smirnov (KS) test for continuous features
    - Chi-squared test for categorical features  
    - Population Stability Index (PSI) for overall drift
    """
    
    def __init__(self, baseline_data: Optional[pd.DataFrame] = None):
        """
        Initialize data drift detector.
        
        Args:
            baseline_data: Baseline feature distributions (from training/validation)
        """
        self.baseline_data = baseline_data
        self.baseline_stats = self._compute_baseline_stats() if baseline_data is not None else None
        self.logger = logger
    
    def _compute_baseline_stats(self) -> Dict[str, Dict]:
        """
        Compute baseline statistics for all features.
        
        Returns:
            Dictionary with feature statistics
        """
        stats_dict = {}
        
        for col in self.baseline_data.columns:
            if pd.api.types.is_numeric_dtype(self.baseline_data[col]):
                stats_dict[col] = {
                    "mean": float(self.baseline_data[col].mean()),
                    "std": float(self.baseline_data[col].std()),
                    "min": float(self.baseline_data[col].min()),
                    "max": float(self.baseline_data[col].max()),
                    "median": float(self.baseline_data[col].median()),
                    "q25": float(self.baseline_data[col].quantile(0.25)),
                    "q75": float(self.baseline_data[col].quantile(0.75)),
                    "type": "numeric"
                }
            else:
                value_counts = self.baseline_data[col].value_counts(normalize=True)
                stats_dict[col] = {
                    "distribution": value_counts.to_dict(),
                    "unique_values": self.baseline_data[col].nunique(),
                    "type": "categorical"
                }
        
        return stats_dict
    
    def ks_test(
        self,
        current_dist: np.ndarray,
        baseline_dist: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        Perform Kolmogorov-Smirnov test.
        
        Tests if two samples come from the same distribution.
        
        Args:
            current_dist: Current production data
            baseline_dist: Baseline training data
            
        Returns:
            (statistic, p_value, drift_detected)
        """
        try:
            statistic, p_value = ks_2samp(baseline_dist, current_dist)
            drift_detected = p_value < KS_TEST_ALPHA
            
            self.logger.debug(
                "KS test completed",
                extra={
                    "statistic": statistic,
                    "p_value": p_value,
                    "drift_detected": drift_detected
                }
            )
            
            return statistic, p_value, drift_detected
            
        except Exception as e:
            self.logger.error(f"KS test failed: {e}")
            raise MetricsComputationException(
                "KS test computation failed",
                details={"error": str(e)}
            )
    
    def chi_squared_test(
        self,
        current: pd.Series,
        baseline: pd.Series
    ) -> Tuple[float, float, bool]:
        """
        Perform chi-squared test for categorical features.
        
        Args:
            current: Current categorical data
            baseline: Baseline categorical data
            
        Returns:
            (statistic, p_value, drift_detected)
        """
        try:
            # Get value counts
            current_counts = current.value_counts()
            baseline_counts = baseline.value_counts()
            
            # Align indices
            all_categories = set(current_counts.index) | set(baseline_counts.index)
            current_aligned = current_counts.reindex(all_categories, fill_value=0)
            baseline_aligned = baseline_counts.reindex(all_categories, fill_value=0)
            
            # Create contingency table
            contingency_table = pd.DataFrame({
                "current": current_aligned,
                "baseline": baseline_aligned
            }).T
            
            # Perform test
            statistic, p_value, dof, expected = chi2_contingency(contingency_table)
            drift_detected = p_value < KS_TEST_ALPHA
            
            self.logger.debug(
                "Chi-squared test completed",
                extra={
                    "statistic": statistic,
                    "p_value": p_value,
                    "drift_detected": drift_detected
                }
            )
            
            return statistic, p_value, drift_detected
            
        except Exception as e:
            self.logger.error(f"Chi-squared test failed: {e}")
            raise MetricsComputationException(
                "Chi-squared test computation failed",
                details={"error": str(e)}
            )
    
    def compute_psi(
        self,
        current: np.ndarray,
        baseline: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Compute Population Stability Index (PSI).
        
        PSI measures the shift in population distribution:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.25: Moderate change
        - PSI >= 0.25: Significant change
        
        Args:
            current: Current production data
            baseline: Baseline training data
            bins: Number of bins for discretization
            
        Returns:
            PSI score
        """
        try:
            # Create bins based on baseline distribution
            breakpoints = np.percentile(
                baseline,
                np.linspace(0, 100, bins + 1)
            )
            breakpoints = np.unique(breakpoints)  # Remove duplicates
            
            # Bin both distributions
            baseline_binned = np.digitize(baseline, breakpoints[:-1])
            current_binned = np.digitize(current, breakpoints[:-1])
            
            # Calculate proportions with actual number of bins
            actual_bins = len(breakpoints)
            baseline_counts = np.bincount(baseline_binned, minlength=actual_bins)
            current_counts = np.bincount(current_binned, minlength=actual_bins)
            
            baseline_props = baseline_counts / len(baseline)
            current_props = current_counts / len(current)
            
            # Avoid division by zero
            baseline_props = np.where(baseline_props == 0, 0.0001, baseline_props)
            current_props = np.where(current_props == 0, 0.0001, current_props)
            
            # Calculate PSI
            psi = np.sum(
                (current_props - baseline_props) * np.log(current_props / baseline_props)
            )
            
            self.logger.debug(
                "PSI computed",
                extra={"psi": psi}
            )
            
            return float(psi)
            
        except Exception as e:
            self.logger.error(f"PSI computation failed: {e}")
            raise MetricsComputationException(
                "PSI computation failed",
                details={"error": str(e)}
            )
    
    def compute_drift_score(
        self,
        current_data: pd.DataFrame,
        method: str = "psi"
    ) -> Dict[str, float]:
        """
        Compute drift score for all features.
        
        Args:
            current_data: Current production data
            method: Method to use (psi, ks, or chi2)
            
        Returns:
            Dictionary of drift scores per feature
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not provided")
        
        drift_scores = {}
        
        try:
            for col in current_data.columns:
                if col not in self.baseline_data.columns:
                    self.logger.warning(f"Column {col} not in baseline data, skipping")
                    continue
                
                if method == "psi":
                    if pd.api.types.is_numeric_dtype(current_data[col]):
                        score = self.compute_psi(
                            current_data[col].values,
                            self.baseline_data[col].values
                        )
                        drift_scores[col] = score
                    
                elif method == "ks":
                    if pd.api.types.is_numeric_dtype(current_data[col]):
                        stat, p_val, _ = self.ks_test(
                            current_data[col].values,
                            self.baseline_data[col].values
                        )
                        drift_scores[col] = stat
                
                elif method == "chi2":
                    if not pd.api.types.is_numeric_dtype(current_data[col]):
                        stat, p_val, _ = self.chi_squared_test(
                            current_data[col],
                            self.baseline_data[col]
                        )
                        drift_scores[col] = stat
            
            self.logger.info(
                f"Computed drift scores for {len(drift_scores)} features using {method}"
            )
            
            return drift_scores
            
        except Exception as e:
            self.logger.error(f"Drift score computation failed: {e}")
            raise MetricsComputationException(
                "Drift score computation failed",
                details={"error": str(e), "method": method}
            )
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Detect data drift across all features.
        
        Args:
            current_data: Current production data
            threshold: PSI threshold (default from settings)
            
        Returns:
            Dictionary with drift detection results
        """
        threshold = threshold or settings.data_drift_threshold
        
        try:
            # Compute PSI for all features
            psi_scores = self.compute_drift_score(current_data, method="psi")
            
            # Determine drift status
            drifted_features = []
            moderate_drift_features = []
            
            for feature, score in psi_scores.items():
                if score >= PSI_SIGNIFICANT_CHANGE:
                    drifted_features.append(feature)
                elif score >= PSI_MODERATE_CHANGE:
                    moderate_drift_features.append(feature)
            
            # Overall drift detected if any feature has significant drift
            drift_detected = len(drifted_features) > 0
            
            # Average PSI across all features
            avg_psi = np.mean(list(psi_scores.values()))
            
            result = {
                "drift_detected": drift_detected,
                "avg_psi": float(avg_psi),
                "threshold": threshold,
                "num_features_drifted": len(drifted_features),
                "num_features_moderate_drift": len(moderate_drift_features),
                "drifted_features": drifted_features,
                "moderate_drift_features": moderate_drift_features,
                "psi_scores": psi_scores
            }
            
            self.logger.info(
                "Data drift detection completed",
                extra={
                    "drift_detected": drift_detected,
                    "avg_psi": avg_psi,
                    "drifted_features": len(drifted_features)
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Data drift detection failed: {e}")
            raise MetricsComputationException(
                "Data drift detection failed",
                details={"error": str(e)}
            )
    
    def get_feature_importance_drift(
        self,
        current_data: pd.DataFrame,
        feature_importance: Dict[str, float],
        top_n: int = 10
    ) -> Dict:
        """
        Compute drift for most important features only.
        
        Args:
            current_data: Current production data
            feature_importance: Feature importance scores
            top_n: Number of top features to monitor
            
        Returns:
            Drift scores for top features
        """
        # Get top N features
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        top_features = [f[0] for f in sorted_features]
        
        # Compute drift only for these features
        current_subset = current_data[top_features]
        drift_scores = self.compute_drift_score(current_subset, method="psi")
        
        return drift_scores
