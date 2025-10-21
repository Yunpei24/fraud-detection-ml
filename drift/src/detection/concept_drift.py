"""
Concept drift detection for model performance monitoring.

Detects changes in the relationship between features and target,
manifesting as model performance degradation.
"""
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from ..config import get_logger, settings
from ..config.constants import (
    CONCEPT_DRIFT_THRESHOLD,
    BASELINE_RECALL,
    BASELINE_PRECISION,
    BASELINE_FPR,
    BASELINE_F1_SCORE
)
from ..utils import MetricsComputationException

logger = get_logger(__name__)


class ConceptDriftDetector:
    """
    Detector for concept drift (model performance degradation).
    
    Monitors:
    - Recall (fraud detection rate)
    - False Positive Rate (false alarm rate)
    - Precision
    - F1 Score
    - ROC-AUC
    """
    
    def __init__(
        self,
        settings=None,
        baseline_recall: Optional[float] = None,
        baseline_precision: Optional[float] = None,
        baseline_fpr: Optional[float] = None
    ):
        """
        Initialize concept drift detector.
        
        Args:
            settings: Settings object from config
            baseline_recall: Baseline recall from validation
            baseline_precision: Baseline precision from validation
            baseline_fpr: Baseline FPR from validation
        """
        self.settings = settings
        self.recall_threshold = getattr(settings, 'drift_recall_threshold', 0.95) if settings else 0.95
        self.fpr_threshold = getattr(settings, 'drift_fpr_threshold', 0.02) if settings else 0.02
        self.baseline_recall = baseline_recall or BASELINE_RECALL
        self.baseline_precision = baseline_precision or BASELINE_PRECISION
        self.baseline_fpr = baseline_fpr or BASELINE_FPR
        self.baseline_f1 = BASELINE_F1_SCORE
        self.logger = logger
    
    def compute_recall(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute recall (sensitivity, true positive rate).
        
        Recall = TP / (TP + FN)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Recall score
        """
        try:
            recall = recall_score(y_true, y_pred, zero_division=0.0)
            
            self.logger.debug(
                "Recall computed",
                extra={"recall": recall}
            )
            
            return float(recall)
            
        except Exception as e:
            self.logger.error(f"Recall computation failed: {e}")
            raise MetricsComputationException(
                "Recall computation failed",
                details={"error": str(e)}
            )
    
    def compute_precision(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute precision.
        
        Precision = TP / (TP + FP)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Precision score
        """
        try:
            precision = precision_score(y_true, y_pred, zero_division=0.0)
            
            self.logger.debug(
                "Precision computed",
                extra={"precision": precision}
            )
            
            return float(precision)
            
        except Exception as e:
            self.logger.error(f"Precision computation failed: {e}")
            raise MetricsComputationException(
                "Precision computation failed",
                details={"error": str(e)}
            )
    
    def compute_fpr(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute False Positive Rate.
        
        FPR = FP / (FP + TN)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            False positive rate
        """
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            if (fp + tn) == 0:
                fpr = 0.0
            else:
                fpr = fp / (fp + tn)
            
            self.logger.debug(
                "FPR computed",
                extra={
                    "fpr": fpr,
                    "fp": int(fp),
                    "tn": int(tn)
                }
            )
            
            return float(fpr)
            
        except Exception as e:
            self.logger.error(f"FPR computation failed: {e}")
            raise MetricsComputationException(
                "FPR computation failed",
                details={"error": str(e)}
            )
    
    def compute_f1_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute F1 score (harmonic mean of precision and recall).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            F1 score
        """
        try:
            f1 = f1_score(y_true, y_pred, zero_division=0.0)
            
            self.logger.debug(
                "F1 score computed",
                extra={"f1_score": f1}
            )
            
            return float(f1)
            
        except Exception as e:
            self.logger.error(f"F1 score computation failed: {e}")
            raise MetricsComputationException(
                "F1 score computation failed",
                details={"error": str(e)}
            )
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Compute all performance metrics for predictions.
        
        Main entry point for metric computation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with all metrics (recall, precision, fpr, f1_score)
        """
        try:
            if len(y_true) == 0:
                raise ValueError("Cannot compute metrics on empty arrays")
            
            if len(y_true) != len(y_pred):
                raise ValueError("y_true and y_pred must have same length")
            
            recall = self.compute_recall(y_true, y_pred)
            precision = self.compute_precision(y_true, y_pred)
            fpr = self.compute_fpr(y_true, y_pred)
            f1 = self.compute_f1_score(y_true, y_pred)
            
            metrics = {
                'recall': float(recall),
                'precision': float(precision),
                'fpr': float(fpr),
                'f1_score': float(f1),
                'baseline_recall': float(self.baseline_recall),
                'baseline_precision': float(self.baseline_precision),
                'baseline_fpr': float(self.baseline_fpr)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics computation failed: {e}")
            raise MetricsComputationException(
                "Metrics computation failed",
                details={"error": str(e)}
            )
    
    def detect(
        self,
        baseline_y_true: np.ndarray,
        baseline_y_pred: np.ndarray,
        current_y_true: np.ndarray,
        current_y_pred: np.ndarray
    ) -> Dict:
        """
        Detect concept drift by comparing baseline and current performance.
        
        Main entry point for concept drift detection.
        
        Args:
            baseline_y_true: Baseline true labels
            baseline_y_pred: Baseline predicted labels
            current_y_true: Current true labels
            current_y_pred: Current predicted labels
            
        Returns:
            Dictionary with drift detection results
        """
        try:
            # Compute baseline metrics from provided baseline
            baseline_metrics = self.compute_metrics(baseline_y_true, baseline_y_pred)
            
            # Compute current metrics
            current_metrics = self.compute_metrics(current_y_true, current_y_pred)
            
            # Detect degradation by comparing baseline vs current
            # NOT by comparing with absolute thresholds
            recall_degraded = current_metrics['recall'] < (baseline_metrics['recall'] - 0.05)
            fpr_degraded = current_metrics['fpr'] > (baseline_metrics['fpr'] + 0.01)
            
            drift_detected = recall_degraded or fpr_degraded
            
            # Determine severity
            if recall_degraded and fpr_degraded:
                severity = "CRITICAL"
            elif recall_degraded or fpr_degraded:
                severity = "HIGH"
            else:
                severity = "LOW"
            
            result = {
                'drift_detected': drift_detected,
                'severity': severity,
                'metrics': {
                    **current_metrics,
                    'baseline_recall': baseline_metrics['recall'],
                    'baseline_precision': baseline_metrics['precision'],
                    'baseline_fpr': baseline_metrics['fpr'],
                    'baseline_f1_score': baseline_metrics['f1_score']
                },
                'recall_degraded': recall_degraded,
                'fpr_degraded': fpr_degraded
            }
            
            if drift_detected:
                if severity == "CRITICAL":
                    self.logger.error(
                        "CRITICAL concept drift detected",
                        extra={
                            "drift_detected": drift_detected,
                            "severity": severity,
                            "recall_degraded": recall_degraded,
                            "fpr_degraded": fpr_degraded
                        }
                    )
                else:
                    self.logger.warning(
                        "Concept drift detected",
                        extra={
                            "drift_detected": drift_detected,
                            "severity": severity,
                            "recall_degraded": recall_degraded,
                            "fpr_degraded": fpr_degraded
                        }
                    )
            else:
                self.logger.info(
                    "No concept drift detected",
                    extra={
                        "drift_detected": drift_detected,
                        "severity": severity
                    }
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Concept drift detection failed: {e}")
            raise MetricsComputationException(
                "Concept drift detection failed",
                details={"error": str(e)}
            )

    def compute_f1_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute F1 score (harmonic mean of precision and recall).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            F1 score
        """
        try:
            f1 = f1_score(y_true, y_pred, zero_division=0.0)
            
            self.logger.debug(
                "F1 score computed",
                extra={"f1_score": f1}
            )
            
            return float(f1)
            
        except Exception as e:
            self.logger.error(f"F1 score computation failed: {e}")
            raise MetricsComputationException(
                "F1 score computation failed",
                details={"error": str(e)}
            )
    
    def compute_roc_auc(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray
    ) -> float:
        """
        Compute ROC-AUC score.
        
        Args:
            y_true: True labels
            y_score: Predicted probabilities
            
        Returns:
            ROC-AUC score
        """
        try:
            if len(np.unique(y_true)) < 2:
                self.logger.warning("Only one class present in y_true, cannot compute ROC-AUC")
                return 0.0
            
            roc_auc = roc_auc_score(y_true, y_score)
            
            self.logger.debug(
                "ROC-AUC computed",
                extra={"roc_auc": roc_auc}
            )
            
            return float(roc_auc)
            
        except Exception as e:
            self.logger.error(f"ROC-AUC computation failed: {e}")
            return 0.0
    
    def compute_confusion_matrix_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Compute all confusion matrix derived metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with all metrics
        """
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Compute derived metrics
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics = {
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "recall": float(recall),
                "precision": float(precision),
                "fpr": float(fpr),
                "specificity": float(specificity),
                "f1_score": float(f1)
            }
            
            self.logger.debug(
                "Confusion matrix metrics computed",
                extra=metrics
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Confusion matrix metrics computation failed: {e}")
            raise MetricsComputationException(
                "Confusion matrix metrics computation failed",
                details={"error": str(e)}
            )
    
    def compare_with_baseline(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Compare current performance with baseline.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Compute current metrics
            current_recall = self.compute_recall(y_true, y_pred)
            current_precision = self.compute_precision(y_true, y_pred)
            current_fpr = self.compute_fpr(y_true, y_pred)
            current_f1 = self.compute_f1_score(y_true, y_pred)
            
            # Compute changes
            recall_change = current_recall - self.baseline_recall
            precision_change = current_precision - self.baseline_precision
            fpr_change = current_fpr - self.baseline_fpr
            f1_change = current_f1 - self.baseline_f1
            
            comparison = {
                "current_recall": float(current_recall),
                "baseline_recall": float(self.baseline_recall),
                "recall_change": float(recall_change),
                "current_precision": float(current_precision),
                "baseline_precision": float(self.baseline_precision),
                "precision_change": float(precision_change),
                "current_fpr": float(current_fpr),
                "baseline_fpr": float(self.baseline_fpr),
                "fpr_change": float(fpr_change),
                "current_f1": float(current_f1),
                "baseline_f1": float(self.baseline_f1),
                "f1_change": float(f1_change)
            }
            
            self.logger.info(
                "Baseline comparison completed",
                extra=comparison
            )
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Baseline comparison failed: {e}")
            raise MetricsComputationException(
                "Baseline comparison failed",
                details={"error": str(e)}
            )
    
    def detect_degradation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        recall_threshold: Optional[float] = None,
        fpr_threshold: Optional[float] = None
    ) -> Dict:
        """
        Detect if model performance has degraded.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            recall_threshold: Minimum acceptable recall
            fpr_threshold: Maximum acceptable FPR
            
        Returns:
            Dictionary with drift detection results
        """
        recall_threshold = recall_threshold or (self.baseline_recall - CONCEPT_DRIFT_THRESHOLD)
        fpr_threshold = fpr_threshold or (self.baseline_fpr + CONCEPT_DRIFT_THRESHOLD)
        
        try:
            # Compare with baseline
            comparison = self.compare_with_baseline(y_true, y_pred)
            
            # Check if performance degraded
            recall_degraded = comparison["current_recall"] < recall_threshold
            fpr_degraded = comparison["current_fpr"] > fpr_threshold
            f1_degraded = comparison["current_f1"] < (self.baseline_f1 - CONCEPT_DRIFT_THRESHOLD)
            
            drift_detected = recall_degraded or fpr_degraded or f1_degraded
            
            # Determine severity
            if recall_degraded and fpr_degraded:
                severity = "CRITICAL"
            elif recall_degraded or fpr_degraded:
                severity = "HIGH"
            elif f1_degraded:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            
            result = {
                "drift_detected": drift_detected,
                "severity": severity,
                "recall_degraded": recall_degraded,
                "fpr_degraded": fpr_degraded,
                "f1_degraded": f1_degraded,
                "recall_threshold": float(recall_threshold),
                "fpr_threshold": float(fpr_threshold),
                **comparison
            }
            
            self.logger.info(
                "Concept drift detection completed",
                extra={
                    "drift_detected": drift_detected,
                    "severity": severity,
                    "recall_change": comparison["recall_change"],
                    "fpr_change": comparison["fpr_change"]
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Concept drift detection failed: {e}")
            raise MetricsComputationException(
                "Concept drift detection failed",
                details={"error": str(e)}
            )
    
    def analyze_error_patterns(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Analyze patterns in prediction errors.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            features: Feature matrix (optional)
            
        Returns:
            Error pattern analysis
        """
        try:
            # Get confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Identify error indices
            false_positive_idx = np.where((y_true == 0) & (y_pred == 1))[0]
            false_negative_idx = np.where((y_true == 1) & (y_pred == 0))[0]
            
            analysis = {
                "false_positive_count": int(len(false_positive_idx)),
                "false_negative_count": int(len(false_negative_idx)),
                "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
                "error_rate": float((fp + fn) / len(y_true))
            }
            
            self.logger.info(
                "Error pattern analysis completed",
                extra=analysis
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error pattern analysis failed: {e}")
            return {}
