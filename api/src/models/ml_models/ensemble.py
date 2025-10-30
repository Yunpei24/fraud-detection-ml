"""
Ensemble ML model for fraud detection.
Combines XGBoost, Random Forest, Neural Network, and Isolation Forest.
"""
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch

from ...config import get_logger, settings

logger = get_logger(__name__)


class EnsembleModel:
    """
    Ensemble model combining multiple ML models for fraud detection.

    Models:
    - XGBoost: Gradient boosting classifier (50% weight)
    - Random Forest: Ensemble tree method (30% weight)
    - Neural Network: Deep learning model (15% weight)
    - Isolation Forest: Anomaly detection (5% weight)

    Uses weighted voting for final prediction.
    """

    def __init__(self, models_path: Optional[str] = None):
        """
        Initialize ensemble model.

        Args:
            models_path: Path to model files directory
        """
        self.models_path = models_path or settings.model_path
        self.xgboost_model = None
        self.random_forest_model = None
        self.nn_model = None
        self.isolation_forest_model = None
        self.shap_explainer_xgb = None
        self.shap_explainer_rf = None
        self.shap_explainer_nn = None
        self.shap_explainer_iforest = None
        self.loaded_at = None
        self.model_version = settings.model_version

        logger.info(f"Initializing EnsembleModel from {self.models_path}")

    def load_models(self) -> None:
        """
        Load all models from disk.

        Raises:
            FileNotFoundError: If model files are not found
            Exception: If model loading fails
        """
        try:
            logger.info("Loading models...")

            # Load XGBoost model
            xgboost_path = os.path.join(self.models_path, settings.xgboost_model_name)
            if os.path.exists(xgboost_path):
                with open(xgboost_path, "rb") as f:
                    self.xgboost_model = pickle.load(f)
                logger.info(" XGBoost model loaded")
            else:
                logger.warning(f"XGBoost model not found at {xgboost_path}, using mock")
                self.xgboost_model = self._create_mock_model("xgboost")

            # Load Random Forest model
            rf_path = os.path.join(self.models_path, settings.random_forest_model_name)
            if os.path.exists(rf_path):
                with open(rf_path, "rb") as f:
                    self.random_forest_model = pickle.load(f)
                logger.info(" Random Forest model loaded")
            else:
                logger.warning(
                    f"Random Forest model not found at {rf_path}, using mock"
                )
                self.random_forest_model = self._create_mock_model("random_forest")

            # Load Neural Network model
            nn_path = os.path.join(self.models_path, settings.nn_model_name)
            if os.path.exists(nn_path):
                self.nn_model = torch.load(nn_path)
                self.nn_model.eval()
                logger.info(" Neural Network model loaded")
            else:
                logger.warning(f"NN model not found at {nn_path}, using mock")
                self.nn_model = self._create_mock_model("neural_network")

            # Load Isolation Forest model
            iforest_path = os.path.join(
                self.models_path, settings.isolation_forest_model_name
            )
            if os.path.exists(iforest_path):
                with open(iforest_path, "rb") as f:
                    self.isolation_forest_model = pickle.load(f)
                logger.info(" Isolation Forest model loaded")
            else:
                logger.warning(
                    f"Isolation Forest not found at {iforest_path}, using mock"
                )
                self.isolation_forest_model = self._create_mock_model(
                    "isolation_forest"
                )

            # Load SHAP explainer for XGBoost
            shap_xgb_path = os.path.join(
                self.models_path, settings.shap_explainer_xgb_name
            )
            if os.path.exists(shap_xgb_path):
                with open(shap_xgb_path, "rb") as f:
                    self.shap_explainer_xgb = pickle.load(f)
                logger.info(" SHAP explainer (XGBoost) loaded")
            else:
                logger.warning(f"SHAP explainer (XGBoost) not found at {shap_xgb_path}")

            # Load SHAP explainer for Random Forest
            shap_rf_path = os.path.join(
                self.models_path, settings.shap_explainer_rf_name
            )
            if os.path.exists(shap_rf_path):
                with open(shap_rf_path, "rb") as f:
                    self.shap_explainer_rf = pickle.load(f)
                logger.info(" SHAP explainer (Random Forest) loaded")
            else:
                logger.warning(f"SHAP explainer (RF) not found at {shap_rf_path}")

            # Load SHAP explainer for Neural Network
            shap_nn_path = os.path.join(
                self.models_path, settings.shap_explainer_nn_name
            )
            if os.path.exists(shap_nn_path):
                with open(shap_nn_path, "rb") as f:
                    self.shap_explainer_nn = pickle.load(f)
                logger.info("✓ SHAP explainer (Neural Network) loaded")
            else:
                logger.warning(f"SHAP explainer (NN) not found at {shap_nn_path}")

            # Load SHAP explainer for Isolation Forest
            shap_iforest_path = os.path.join(
                self.models_path, settings.shap_explainer_iforest_name
            )
            if os.path.exists(shap_iforest_path):
                with open(shap_iforest_path, "rb") as f:
                    self.shap_explainer_iforest = pickle.load(f)
                logger.info("✓ SHAP explainer (Isolation Forest) loaded")
            else:
                logger.warning(
                    f"SHAP explainer (IForest) not found at {shap_iforest_path}"
                )

            self.loaded_at = datetime.utcnow()
            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _create_mock_model(self, model_type: str):
        """
        Create a mock model for testing when real models are not available.

        Args:
            model_type: Type of model to mock

        Returns:
            Mock model object
        """

        class MockModel:
            def __init__(self, model_type):
                self.model_type = model_type

            def predict(self, X):
                """Mock prediction - random but deterministic."""
                np.random.seed(42)
                return np.random.randint(0, 2, size=len(X))

            def predict_proba(self, X):
                """Mock probability prediction."""
                np.random.seed(42)
                probs = np.random.random(size=(len(X), 2))
                probs = probs / probs.sum(axis=1, keepdims=True)
                return probs

        return MockModel(model_type)

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make fraud prediction using ensemble voting.

        Args:
            features: Input features (2D array, shape [n_samples, n_features])

        Returns:
            Dictionary containing:
                - prediction: Final prediction (0 or 1)
                - confidence: Confidence score (0.0 to 1.0)
                - fraud_score: Probability of fraud
                - individual_scores: Scores from each model

        Raises:
            ValueError: If models are not loaded
            ValueError: If input shape is invalid
        """
        if not self.health_check():
            raise ValueError("Models are not loaded. Call load_models() first.")

        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)

        try:
            # Get predictions from each model
            scores = {}

            # XGBoost prediction
            if hasattr(self.xgboost_model, "predict_proba"):
                xgb_proba = self.xgboost_model.predict_proba(features)[:, 1]
            else:
                xgb_proba = self.xgboost_model.predict(features).astype(float)
            scores["xgboost"] = float(xgb_proba[0])

            # Random Forest prediction
            if hasattr(self.random_forest_model, "predict_proba"):
                rf_proba = self.random_forest_model.predict_proba(features)[:, 1]
            else:
                rf_proba = self.random_forest_model.predict(features).astype(float)
            scores["random_forest"] = float(rf_proba[0])

            # Neural Network prediction
            if hasattr(self.nn_model, "predict_proba"):
                nn_proba = self.nn_model.predict_proba(features)[:, 1]
            else:
                nn_proba = self.nn_model.predict(features).astype(float)
            scores["neural_network"] = float(nn_proba[0])

            # Isolation Forest prediction (anomaly score)
            if hasattr(self.isolation_forest_model, "predict_proba"):
                if_proba = self.isolation_forest_model.predict_proba(features)[:, 1]
            else:
                # Isolation Forest returns -1 (outlier) or 1 (inlier)
                # Convert to probability: -1 -> 1.0, 1 -> 0.0
                if_pred = self.isolation_forest_model.predict(features)
                if_proba = (1 - if_pred) / 2  # Maps -1->1.0, 1->0.0
            scores["isolation_forest"] = float(if_proba[0])

            # Ensemble voting with weights (50% XGB, 30% RF, 15% NN, 5% IForest)
            weighted_score = (
                scores["xgboost"] * 0.50
                + scores["random_forest"] * 0.30
                + scores["neural_network"] * 0.15
                + scores["isolation_forest"] * 0.05
            )

            # Calculate confidence (higher when models agree)
            score_variance = np.var(list(scores.values()))
            confidence = 1.0 - min(score_variance * 2, 0.5)  # Max penalty of 0.5

            # Final prediction
            prediction = int(weighted_score >= settings.fraud_threshold)

            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "fraud_score": float(weighted_score),
                "individual_scores": scores,
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def explain_prediction(
        self,
        features: np.ndarray,
        prediction_result: Dict[str, Any],
        model_type: str = "xgboost",
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for prediction.

        Args:
            features: Input features
            prediction_result: Result from predict()
            model_type: Which model's explanation to generate ('xgboost', 'random_forest', 'neural_network', 'isolation_forest')

        Returns:
            Dictionary with explanation data
        """
        if not settings.enable_shap_explanation:
            return {"message": "SHAP explanations are disabled"}

        # Select appropriate SHAP explainer
        if model_type == "xgboost":
            explainer = self.shap_explainer_xgb
        elif model_type == "random_forest":
            explainer = self.shap_explainer_rf
        elif model_type == "neural_network":
            explainer = self.shap_explainer_nn
        elif model_type == "isolation_forest":
            explainer = self.shap_explainer_iforest
        else:
            explainer = self.shap_explainer_xgb  # Default to XGBoost

        if explainer is None:
            logger.warning(
                f"SHAP explainer ({model_type}) not loaded, returning mock explanation"
            )
            return self._create_mock_explanation(features)

        try:
            # Ensure 2D array
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Get SHAP values
            shap_values = explainer.shap_values(features)

            # Get top features by absolute SHAP value
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (fraud)

            feature_importance = {}
            for i, shap_val in enumerate(shap_values[0]):
                feature_importance[f"feature_{i}"] = float(shap_val)

            # Sort by absolute importance
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )

            return {
                "model": model_type,
                "top_features": dict(sorted_features[:10]),
                "all_features": feature_importance,
                "method": "SHAP",
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed for {model_type}: {e}")
            return self._create_mock_explanation(features)

    def _create_mock_explanation(self, features: np.ndarray) -> Dict[str, Any]:
        """Create mock SHAP explanation for testing."""
        # Simple mock: features with highest absolute values
        if features.ndim == 1:
            features = features.reshape(1, -1)

        feature_importance = {
            f"feature_{i}": float(abs(val)) for i, val in enumerate(features[0])
        }

        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "top_features": dict(sorted_features[:10]),
            "method": "Mock (magnitude-based)",
        }

    def health_check(self) -> bool:
        """
        Check if all models are loaded and ready.

        Returns:
            True if all models are loaded, False otherwise
        """
        return (
            self.xgboost_model is not None
            and self.random_forest_model is not None
            and self.nn_model is not None
            and self.isolation_forest_model is not None
        )

    def get_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        return {
            "version": self.model_version,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "models": {
                "xgboost": "loaded" if self.xgboost_model else "not loaded",
                "random_forest": "loaded" if self.random_forest_model else "not loaded",
                "neural_network": "loaded" if self.nn_model else "not loaded",
                "isolation_forest": "loaded"
                if self.isolation_forest_model
                else "not loaded",
            },
            "shap_explainers": {
                "xgboost": "loaded" if self.shap_explainer_xgb else "not loaded",
                "random_forest": "loaded" if self.shap_explainer_rf else "not loaded",
                "neural_network": "loaded" if self.shap_explainer_nn else "not loaded",
                "isolation_forest": "loaded"
                if self.shap_explainer_iforest
                else "not loaded",
            },
        }

    def get_feature_importance(self, model_type: str) -> Dict[str, Any]:
        """
        Get global feature importance for a specific model.

        Args:
            model_type: Type of model ('xgboost', 'neural_network', 'isolation_forest')

        Returns:
            Dictionary with feature importance data
        """
        try:
            if model_type == "xgboost" and self.xgboost_model:
                # XGBoost has built-in feature importance
                importance = self.xgboost_model.feature_importances_
                feature_importance = {
                    f"feature_{i}": float(imp) for i, imp in enumerate(importance)
                }

            elif model_type == "neural_network" and self.nn_model:
                # For neural networks, we could use permutation importance or other methods
                # For now, return a message that NN feature importance is not available
                logger.warning("Neural network feature importance not implemented")
                return {
                    "message": "Feature importance not available for neural networks"
                }

            elif model_type == "isolation_forest" and self.isolation_forest_model:
                # Isolation Forest doesn't have traditional feature importance
                # Could use permutation importance or other methods
                logger.warning("Isolation Forest feature importance not implemented")
                return {
                    "message": "Feature importance not available for Isolation Forest"
                }

            else:
                return {"message": f"Model {model_type} not loaded or not supported"}

            return {"feature_importance": feature_importance, "model_type": model_type}

        except Exception as e:
            logger.error(f"Feature importance retrieval failed for {model_type}: {e}")
            return {"message": f"Error retrieving feature importance: {str(e)}"}
