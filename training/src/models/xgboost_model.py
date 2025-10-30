# training/src/models/xgboost_model.py
"""
XGBoost model wrapper with SMOTE support for imbalanced fraud detection.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from src.config.logging_config import get_logger
from .hyperparameter_tuning import tune_xgboost_hyperparameters

logger = get_logger(__name__)


class XGBoostModel:
    """
    XGBoost classifier with optional SMOTE for handling class imbalance.
    
    Parameters:
    -----------
    use_smote : bool
        Whether to apply SMOTE oversampling during training
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning
    random_state : int
        Random seed for reproducibility
    params : Dict[str, Any]
        Additional XGBoost hyperparameters
    """
    
    def __init__(
        self,
        use_smote: bool = True,
        tune_hyperparams: bool = False,
        random_state: int = 42,
        n_estimators: int = 300,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.use_smote = use_smote
        self.tune_hyperparams = tune_hyperparams
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.params = params or {}
        
        # Default parameters optimized for fraud detection
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": n_estimators,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "scale_pos_weight": None,  # Will be set during fit
            "random_state": random_state,
            # Removed deprecated use_label_encoder parameter
        }
        
        # Merge user params
        default_params.update(self.params)
        
        self.model = xgb.XGBClassifier(**default_params)
        self.smote = SMOTE(random_state=random_state) if use_smote else None
        self.is_fitted = False
        self.optimal_threshold = 0.5  # Default threshold
        
        logger.info(f"XGBoostModel initialized (use_smote={use_smote}, tune_hyperparams={tune_hyperparams})")
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "XGBoostModel":
        """
        Train the XGBoost model with optional SMOTE resampling and hyperparameter tuning.
        """
        X_train, y_train = X, y
        
        # Calculate class imbalance ratio for scale_pos_weight
        neg_count = len(y_train[y_train == 0])
        pos_count = len(y_train[y_train == 1])
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        # Update scale_pos_weight in model
        self.model.set_params(scale_pos_weight=scale_pos_weight)
        
        if self.use_smote and self.smote is not None:
            logger.info(f"Applying SMOTE: original class distribution: {np.bincount(y_train.astype(int))}")
            X_train, y_train = self.smote.fit_resample(X, y)
            logger.info(f"After SMOTE: {np.bincount(y_train.astype(int))}")
        
        if self.tune_hyperparams and X_val is not None and y_val is not None:
            logger.info("Performing hyperparameter tuning...")
            tuning_results = tune_xgboost_hyperparameters(
                X_train, y_train, X_val, y_val, random_state=self.random_state
            )
            
            # Update model with best parameters
            best_params = tuning_results['best_params']
            self.model = xgb.XGBClassifier(**best_params)
            self.model.set_params(scale_pos_weight=scale_pos_weight)  # Keep the imbalance handling
            
            # Store optimal threshold
            self.optimal_threshold = tuning_results.get('optimal_threshold', 0.5)
            
            logger.info(f"Using tuned hyperparameters: {best_params}")
            logger.info(f"Optimal threshold: {self.optimal_threshold:.4f}")
        
        logger.info(f"Training XGBoost on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("XGBoost training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities. Returns shape (n_samples, 2).
        """
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances from the trained model.
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Return as dict for compatibility with tests
        return {f'feature_{i}': float(imp) for i, imp in enumerate(self.model.feature_importances_)}
