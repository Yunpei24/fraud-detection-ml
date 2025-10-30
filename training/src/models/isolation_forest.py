# training/src/models/isolation_forest.py
"""
Isolation Forest model for anomaly-based fraud detection.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
from sklearn.ensemble import IsolationForest

from src.config.logging_config import get_logger
from .hyperparameter_tuning import tune_isolation_forest_hyperparameters

logger = get_logger(__name__)


class IsolationForestModel:
    """
    Isolation Forest for unsupervised anomaly detection.
    
    Parameters:
    -----------
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning
    random_state : int
        Random seed for reproducibility
    contamination : float or 'auto'
        Expected proportion of outliers (fraud ratio). If 'auto', estimated from data
    params : Dict[str, Any]
        Additional IsolationForest hyperparameters
    """
    
    def __init__(
        self,
        tune_hyperparams: bool = False,
        random_state: int = 42,
        contamination: float = 'auto',  # Changed from 0.0017 to 'auto'
        params: Optional[Dict[str, Any]] = None,
    ):
        self.tune_hyperparams = tune_hyperparams
        self.random_state = random_state
        self.contamination = contamination
        self.params = params or {}
        self.optimal_threshold = None  # Store optimal threshold from tuning
        
        # Default parameters optimized for fraud detection
        default_params = {
            "n_estimators": 300,  # Increased from 200
            "max_samples": 0.8,   # Changed from 'auto' to 0.8
            "contamination": contamination,
            "max_features": 0.8,  # Changed from 1.0 to 0.8
            "random_state": random_state,
            "n_jobs": 1,  # Changed from -1 to avoid memory issues
        }
        
        # Merge user params
        default_params.update(self.params)
        
        self.model = IsolationForest(**default_params)
        self.is_fitted = False
        
        logger.info(f"IsolationForestModel initialized (contamination={contamination}, tune_hyperparams={tune_hyperparams})")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "IsolationForestModel":
        """
        Train the Isolation Forest. Labels y are ignored (unsupervised).
        Supports hyperparameter tuning if enabled and validation data provided.
        """
        if self.tune_hyperparams and X_val is not None and y_val is not None:
            logger.info("Performing hyperparameter tuning for Isolation Forest...")
            tuning_results = tune_isolation_forest_hyperparameters(
                X, y, X_val, y_val, random_state=self.random_state
            )
            
            # Update model with best parameters
            best_params = tuning_results['best_params']
            self.model = IsolationForest(**best_params)
            self.optimal_threshold = tuning_results.get('optimal_threshold')
            
            logger.info(f"Using tuned hyperparameters: {best_params}")
            if self.optimal_threshold is not None:
                logger.info(f"Using optimal threshold: {self.optimal_threshold:.4f}")
        
        logger.info(f"Training Isolation Forest on {X.shape[0]} samples...")
        self.model.fit(X)
        self.is_fitted = True
        
        # Log contamination estimate
        if hasattr(self.model, 'offset_'):
            logger.info(f"Isolation Forest training completed (offset={self.model.offset_:.4f})")
        else:
            logger.info("Isolation Forest training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels: 1 for inliers (normal), -1 for outliers (fraud).
        Convert to binary: 0 for normal, 1 for fraud.
        
        Uses optimal threshold from hyperparameter tuning if available.
        """
        if self.optimal_threshold is not None:
            # Use tuned threshold
            scores = self.model.decision_function(X)
            predictions = (scores <= self.optimal_threshold).astype(int)
        else:
            # Use default contamination-based prediction
            predictions = self.model.predict(X)
            # Map: -1 (outlier/fraud) -> 1, +1 (inlier/normal) -> 0
            predictions = (predictions == -1).astype(int)
        
        return predictions
    
    # def predict_proba(self, X: np.ndarray) -> np.ndarray:
    #     """
    #     Return anomaly scores as calibrated probabilities.
        
    #     Uses a more sophisticated calibration approach that considers
    #     the contamination parameter and provides better fraud probability estimates.
        
    #     Returns shape (n_samples, 2) for compatibility: [prob_normal, prob_fraud]
    #     """
    #     # Get raw anomaly scores (more negative = more anomalous)
    #     scores = self.model.decision_function(X)
        
    #     # Get contamination rate for calibration
    #     if self.contamination == 'auto':
    #         # Estimate contamination from training data distribution
    #         # This is a rough estimate - in practice you'd use validation data
    #         contamination_rate = 0.01  # Conservative estimate
    #     else:
    #         contamination_rate = self.contamination
        
    #     # Use optimal threshold if available from tuning
    #     if self.optimal_threshold is not None:
    #         threshold = self.optimal_threshold
    #     else:
    #         # Fall back to percentile-based threshold
    #         threshold = np.percentile(scores, (1 - contamination_rate) * 100)
        
    #     # Calibrate scores to probabilities using a sigmoid-like transformation
    #     # Center the scores around the threshold
    #     fraud_proba = 1 / (1 + np.exp((scores - threshold) / np.abs(np.std(scores))))
        
    #     # Ensure we have some fraud predictions even with low contamination
    #     min_fraud_prob = contamination_rate * 2  # Allow some flexibility
    #     fraud_proba = np.maximum(fraud_proba, min_fraud_prob)
        
    #     # Normalize to ensure probabilities sum to 1
    #     normal_proba = 1 - fraud_proba
        
    #     return np.column_stack([normal_proba, fraud_proba])
    
    def predict_proba(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return anomaly scores as calibrated probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : Optional[np.ndarray]
            Target labels (0=normal, 1=fraud). Used to dynamically estimate
            contamination rate as y.mean() when available.
        """
        # Get raw anomaly scores (more negative = more anomalous)
        scores = self.model.decision_function(X)

        # --- 🔹 Adaptative contamination rate ---
        if self.contamination == 'auto':
            # Dynamically infer contamination rate from labels if available
            contamination_rate = y.mean() if y is not None else 0.04
        else:
            contamination_rate = self.contamination

        # --- 🔹 Threshold selection ---
        if self.optimal_threshold is not None:
            threshold = self.optimal_threshold
        else:
            # Percentile-based threshold based on contamination rate
            threshold = np.percentile(scores, (1 - contamination_rate) * 100)

        # --- 🔹 Sigmoid calibration of scores ---
        # Shift scores around the threshold and scale by score dispersion
        fraud_proba = 1 / (1 + np.exp((scores - threshold) / np.abs(np.std(scores) + 1e-8)))

        # Ensure some minimum fraud probability for numerical stability
        min_fraud_prob = contamination_rate * 2  # allow some flexibility
        fraud_proba = np.maximum(fraud_proba, min_fraud_prob)

        # Normalize to ensure probabilities sum to 1
        normal_proba = 1 - fraud_proba

        return np.column_stack([normal_proba, fraud_proba])


    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw anomaly scores (more negative = more anomalous).
        """
        return self.model.decision_function(X)
