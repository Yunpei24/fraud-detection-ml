# training/src/models/neural_network.py
"""
Neural Network (MLP) model wrapper for fraud detection.
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import joblib

from src.config.logging_config import get_logger
from .hyperparameter_tuning import tune_neural_network_hyperparameters

logger = get_logger(__name__)


class NeuralNetworkModel:
    """
    Multi-Layer Perceptron classifier with optional SMOTE for handling class imbalance.
    
    Note: Input features should be scaled (0-1 or standardized) for best results.
    
    Parameters:
    -----------
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning
    random_state : int
        Random seed for reproducibility
    params : Dict[str, Any]
        Additional MLP hyperparameters
    """
    
    def __init__(
        self,
        tune_hyperparams: bool = False,
        random_state: int = 42,
        hidden_layers: tuple = (128, 64, 32),
        epochs: int = 300,
        batch_size: int = 256,
        dropout_rate: float = 0.0,
        early_stopping: bool = False,
        patience: int = 20,
        class_weight: Optional[Dict[int, float]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.tune_hyperparams = tune_hyperparams
        self.random_state = random_state
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.early_stopping = early_stopping
        self.patience = patience
        self.class_weight = class_weight
        self.params = params or {}
        
        # Default parameters optimized for fraud detection
        default_params = {
            "hidden_layer_sizes": hidden_layers,
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,  # L2 regularization
            "batch_size": batch_size,
            "learning_rate": "adaptive",
            "learning_rate_init": 0.001,
            "max_iter": epochs,
            "early_stopping": early_stopping,
            "validation_fraction": 0.1,
            "n_iter_no_change": patience,
            "random_state": random_state,
        }
        
        # Merge user params
        default_params.update(self.params)
        
        self.model = MLPClassifier(**default_params)
        # Neural networks benefit from SMOTE
        self.smote = SMOTE(random_state=random_state)
        self.is_fitted = False
        
        logger.info(f"NeuralNetworkModel initialized (hidden_layers={hidden_layers}, tune_hyperparams={tune_hyperparams})")
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "NeuralNetworkModel":
        """
        Train the neural network model with optional SMOTE resampling and hyperparameter tuning.
        """
        X_train, y_train = X, y
        
        # Apply SMOTE for imbalanced data
        logger.info(f"Applying SMOTE: original class distribution: {np.bincount(y_train.astype(int))}")
        X_train, y_train = self.smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE: {np.bincount(y_train.astype(int))}")
        
        if self.tune_hyperparams and X_val is not None and y_val is not None:
            logger.info("Performing hyperparameter tuning...")
            tuning_results = tune_neural_network_hyperparameters(
                X_train, y_train, X_val, y_val, random_state=self.random_state
            )
            
            # Update model with best parameters
            best_params = tuning_results['best_params']
            self.model = MLPClassifier(**best_params)
            
            logger.info(f"Using tuned hyperparameters: {best_params}")
        
        logger.info(f"Training Neural Network on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info(f"Neural Network training completed in {self.model.n_iter_} iterations")
        
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
    
    @classmethod
    def load(cls, filepath: str) -> "NeuralNetworkModel":
        """
        Load a saved Neural Network model.
        """
        model = joblib.load(filepath)
        model.is_fitted = True
        return model
    
    def save(self, filepath: str) -> None:
        """
        Save the Neural Network model.
        """
        joblib.dump(self, filepath)
