"""
Prediction service for fraud detection.
"""

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fraud_detection_common.feature_engineering import build_feature_frame
from fraud_detection_common.preprocessor import DataPreprocessor

from ..config import get_logger, settings
from ..models import EnsembleModel
from ..utils import InvalidInputException, PredictionFailedException, validate_features

logger = get_logger(__name__)


# Exception classes
class PredictionError(Exception):
    """Base exception for prediction service errors."""

    pass


class ModelLoadError(PredictionError):
    """Exception raised when model loading fails."""

    pass


# Model classes
@dataclass
class PredictionResult:
    """Result of a single prediction."""

    prediction_id: str
    timestamp: datetime
    model_version: str
    fraud_probability: float
    is_fraud: bool
    confidence_score: float
    processing_time_ms: float


@dataclass
class TrafficRoutingConfig:
    """Configuration for traffic routing between champion and canary models."""

    canary_enabled: bool = False
    canary_traffic_pct: int = 0
    champion_traffic_pct: int = 100
    canary_model_uris: List[str] = None
    champion_model_uris: List[str] = None
    ensemble_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.canary_model_uris is None:
            self.canary_model_uris = []
        if self.champion_model_uris is None:
            self.champion_model_uris = []
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "xgboost": 0.50,
                "random_forest": 0.30,
                "neural_network": 0.15,
                "isolation_forest": 0.05,
            }


class TrafficRouter:
    """Handles traffic routing between champion and canary models."""

    def __init__(self, config_path: str = None):
        """
        Initialize traffic router.

        Args:
            config_path: Path to traffic routing configuration file
        """
        self.config_path = config_path or settings.traffic_routing_config
        self.config = self._load_config()
        self.logger = logger

    def _load_config(self) -> TrafficRoutingConfig:
        """Load traffic routing configuration from file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, "r") as f:
                    data = json.load(f)
                return TrafficRoutingConfig(**data)
            else:
                # Default configuration - no canary
                return TrafficRoutingConfig()
        except Exception as e:
            self.logger.warning(f"Failed to load traffic routing config: {e}")
            return TrafficRoutingConfig()

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        self.logger.info(
            f"Traffic routing config reloaded: canary_enabled={self.config.canary_enabled}, canary_pct={self.config.canary_traffic_pct}%"
        )

    def should_use_canary(self) -> bool:
        """
        Determine if the current request should use canary models.

        Returns:
            True if canary models should be used, False for champion
        """
        if not self.config.canary_enabled:
            return False

        # Use random routing based on traffic percentage
        return random.random() * 100 < self.config.canary_traffic_pct

    def get_model_info(self) -> Dict[str, Any]:
        """Get current traffic routing information."""
        return {
            "canary_enabled": self.config.canary_enabled,
            "canary_traffic_pct": self.config.canary_traffic_pct,
            "champion_traffic_pct": self.config.champion_traffic_pct,
            "canary_model_uris": self.config.canary_model_uris,
            "champion_model_uris": self.config.champion_model_uris,
            "ensemble_weights": self.config.ensemble_weights,
        }


class PredictionService:
    """Service for making fraud predictions."""

    def __init__(self, model: Any, traffic_router: Optional[Any] = None):
        """
        Initialize prediction service.

        Args:
            model: Ensemble model instance (champion model)
            traffic_router: Traffic router for canary deployment
        """
        self.model = model  # Champion model
        self.canary_model = None  # Will be loaded if canary is enabled
        self.traffic_router = traffic_router or TrafficRouter()
        self.logger = logger

        # Try to load canary model if canary is enabled
        self._load_canary_model_if_needed()

    def _load_canary_model_if_needed(self) -> None:
        """Load canary model if canary deployment is enabled."""
        if self.traffic_router.canary_percentage > 0:
            try:
                # Create canary model instance with canary model path
                canary_path = self.traffic_router.canary_model_path
                self.canary_model = EnsembleModel(models_path=canary_path)
                self.canary_model.load_models()
                self.logger.info(f"Canary model loaded successfully from {canary_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load canary model: {e}")
                self.canary_model = None

    async def predict_single(
        self,
        transaction_id: str,
        features: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make prediction for a single transaction.

        Args:
            transaction_id: Unique transaction identifier
            features: Transaction features
            metadata: Optional metadata

        Returns:
            Prediction result dictionary

        Raises:
            PredictionFailedException: If prediction fails
        """
        start_time = time.time()

        try:
            # Validate features
            validate_features(features)

            # Determine which model to use (champion or canary)
            use_canary = self.traffic_router.should_use_canary()
            selected_model = (
                self.canary_model if (use_canary and self.canary_model) else self.model
            )
            model_type = "canary" if use_canary else "champion"

            # Log traffic routing decision
            self.logger.info(
                f"Traffic routing decision: {model_type} model selected",
                extra={
                    "transaction_id": transaction_id,
                    "model_type": model_type,
                    "canary_percentage": self.traffic_router.canary_percentage,
                },
            )

            # Prepare features
            features_array = self._prepare_features(features)

            # Make prediction
            prediction_result = selected_model.predict(features_array)

            # Get explanation if enabled
            explanation = None
            if settings.enable_shap_explanation:
                try:
                    explanation = self.model.explain_prediction(
                        features_array, prediction_result
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to generate explanation: {e}")

            # Calculate processing time
            processing_time = time.time() - start_time

            # Build response
            result = {
                "transaction_id": transaction_id,
                "prediction": int(prediction_result["prediction"]),
                "confidence": float(prediction_result["confidence"]),
                "fraud_score": float(prediction_result["fraud_score"]),
                "risk_level": self._get_risk_level(prediction_result["fraud_score"]),
                "processing_time": processing_time,
                "model_version": settings.model_version,
                "model_type": model_type,  # "champion" or "canary"
                "timestamp": time.time(),
            }

            if explanation:
                result["explanation"] = explanation

            if metadata:
                result["metadata"] = metadata

            # Log prediction
            self.logger.info(
                f"Prediction made",
                extra={
                    "transaction_id": transaction_id,
                    "prediction": result["prediction"],
                    "fraud_score": result["fraud_score"],
                    "processing_time": processing_time,
                },
            )

            return result

        except InvalidInputException:
            raise
        except Exception as e:
            self.logger.error(
                f"Prediction failed for transaction {transaction_id}: {e}",
                exc_info=True,
            )
            raise PredictionFailedException(
                f"Failed to make prediction: {str(e)}",
                details={"transaction_id": transaction_id, "error": str(e)},
            )

    async def predict_batch(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make predictions for multiple transactions.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Batch prediction results
        """
        start_time = time.time()
        predictions = []
        errors = []

        for transaction in transactions:
            transaction_id = transaction.get("transaction_id")
            features = transaction.get("features")
            metadata = transaction.get("metadata")

            try:
                result = await self.predict_single(
                    transaction_id=transaction_id, features=features, metadata=metadata
                )
                predictions.append(result)
            except Exception as e:
                errors.append({"transaction_id": transaction_id, "error": str(e)})
                self.logger.error(
                    f"Failed to process transaction {transaction_id}: {e}"
                )

        # Calculate statistics
        total_time = time.time() - start_time
        fraud_count = sum(1 for p in predictions if p["prediction"] == 1)

        return {
            "total_transactions": len(transactions),
            "successful_predictions": len(predictions),
            "failed_predictions": len(errors),
            "fraud_detected": fraud_count,
            "fraud_rate": fraud_count / len(predictions) if predictions else 0,
            "predictions": predictions,
            "errors": errors if errors else None,
            "processing_time": total_time,
            "avg_processing_time": (
                total_time / len(transactions) if transactions else 0
            ),
        }

    def _prepare_features(self, features: List[float]) -> np.ndarray:
        """
        Prepare features for model input.
        This now includes preprocessing and feature engineering.

        Args:
            features: Raw features list from the request

        Returns:
            Numpy array ready for model
        """
        # 1. Create a DataFrame from the raw features
        # The column names must match what the preprocessor and feature engineer expect.
        # Based on the dataset description: time, v1-v28, amount
        columns = ["time"] + [f"v{i}" for i in range(1, 29)] + ["amount"]
        # The 'Class' column is not in the input, but we add it for schema consistency
        # with the training data, then drop it.
        input_df = pd.DataFrame([features], columns=columns)

        # 2. Preprocess the data
        # We need a pre-fitted scaler. This should be loaded with the model.
        # For now, we assume the model object has the preprocessor.
        if not hasattr(self.model, "preprocessor") or not isinstance(
            self.model.preprocessor, DataPreprocessor
        ):
            raise PredictionFailedException("Model is missing a valid preprocessor.")

        preprocessed_df = self.model.preprocessor.transform(input_df)

        # 3. Engineer features
        # The feature engineering steps must be the same as in training.
        engineered_df = build_feature_frame(preprocessed_df)

        # 4. Ensure final feature set matches model expectation
        # The model was trained on a specific set of columns after all transformations.
        # We must align the columns of the engineered_df with the model's expected features.

        # For now, we convert to numpy array. A more robust solution would be to
        # align columns with a stored list of feature names from the training phase.
        features_array = engineered_df.to_numpy(dtype=np.float32)

        # Reshape to (1, n_features) for single prediction
        features_array = features_array.reshape(1, -1)

        return features_array

    def _get_risk_level(self, fraud_score: float) -> str:
        """
        Get risk level from fraud score.

        Args:
            fraud_score: Fraud score (0-1)

        Returns:
            Risk level string
        """
        if fraud_score >= 0.9:
            return "CRITICAL"
        elif fraud_score >= 0.7:
            return "HIGH"
        elif fraud_score >= 0.5:
            return "MEDIUM"
        elif fraud_score >= 0.3:
            return "LOW"
        else:
            return "MINIMAL"

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Model information dictionary
        """
        return self.model.get_info()

    def check_model_health(self) -> bool:
        """
        Check if model is healthy.

        Returns:
            True if model is healthy
        """
        return self.model.health_check()

    async def explain_prediction_shap(
        self,
        transaction_id: str,
        features: List[float],
        model_type: str = "ensemble",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a prediction.

        Args:
            transaction_id: Transaction identifier
            features: Feature values
            model_type: Model to explain ('xgboost', 'neural_network', 'isolation_forest', 'ensemble')
            metadata: Additional metadata

        Returns:
            SHAP explanation dictionary
        """
        try:
            # First make a prediction to get prediction_result
            prediction_result = await self.predict_single(
                transaction_id=transaction_id, features=features, metadata=metadata
            )

            # Prepare features
            features_array = self._prepare_features(features)

            # Generate SHAP explanation
            explanation = self.model.explain_prediction(
                features=features_array,
                prediction_result=prediction_result,
                model_type=model_type,
            )

            # Format the response
            top_features = []
            if "top_features" in explanation:
                for i, (feature_name, importance) in enumerate(
                    explanation["top_features"].items()
                ):
                    top_features.append(
                        {
                            "feature_name": feature_name,
                            "importance_score": float(importance),
                            "rank": i + 1,
                        }
                    )

            # Calculate shap values summary from all_features if available
            all_features = explanation.get("all_features", {})
            positive_contributors = sum(1 for v in all_features.values() if v > 0)
            negative_contributors = sum(1 for v in all_features.values() if v < 0)

            result = {
                "transaction_id": transaction_id,
                "model_type": model_type,
                "method": "SHAP",
                "top_features": top_features,
                "base_value": 0.0,  # SHAP base value (would need to extract from explainer)
                "prediction_value": float(prediction_result.get("fraud_score", 0.0)),
                "shap_values_summary": {
                    "positive_contributors": positive_contributors,
                    "negative_contributors": negative_contributors,
                    "total_features": len(features),
                },
            }

            self.logger.info(
                f"Generated SHAP explanation for transaction {transaction_id}"
            )
            return result

        except Exception as e:
            self.logger.error(f"SHAP explanation failed for {transaction_id}: {e}")
            raise PredictionFailedException(
                error_code="E800",
                message="SHAP explanation generation failed",
                details={"transaction_id": transaction_id, "error": str(e)},
            )

    async def get_feature_importance(self, model_type: str) -> Dict[str, Any]:
        """
        Get global feature importance for a model.

        Args:
            model_type: Model type ('xgboost', 'neural_network', 'isolation_forest')

        Returns:
            Feature importance dictionary
        """
        try:
            # Get feature importance from model
            importance_dict = self.model.get_feature_importance(model_type)

            # Format the response
            feature_importances = []
            if "feature_importance" in importance_dict:
                # Sort by importance (descending)
                sorted_features = sorted(
                    importance_dict["feature_importance"].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )

                for i, (feature_name, importance) in enumerate(sorted_features):
                    feature_importances.append(
                        {
                            "feature_name": feature_name,
                            "importance_score": float(importance),
                            "rank": i + 1,
                        }
                    )

            result = {
                "model_type": model_type,
                "method": "feature_importance",
                "feature_importances": feature_importances,
                "total_features": len(feature_importances),
            }

            self.logger.info(f"Retrieved feature importance for model {model_type}")
            return result

        except Exception as e:
            self.logger.error(
                f"Feature importance retrieval failed for {model_type}: {e}"
            )
            raise PredictionFailedException(
                error_code="E802",
                message="Feature importance retrieval failed",
                details={"model_type": model_type, "error": str(e)},
            )
