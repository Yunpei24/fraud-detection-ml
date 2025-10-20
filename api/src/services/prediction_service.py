"""
Prediction service for fraud detection.
"""
import time
from typing import Dict, Any, List, Optional
import numpy as np

from ..models import EnsembleModel
from ..utils import (
    validate_features,
    PredictionFailedException,
    InvalidInputException
)
from ..config import get_logger, settings

logger = get_logger(__name__)


class PredictionService:
    """Service for making fraud predictions."""
    
    def __init__(self, model: EnsembleModel):
        """
        Initialize prediction service.
        
        Args:
            model: Loaded ensemble model
        """
        self.model = model
        self.logger = logger
    
    async def predict_single(
        self,
        transaction_id: str,
        features: List[float],
        metadata: Optional[Dict[str, Any]] = None
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
            
            # Prepare features
            features_array = self._prepare_features(features)
            
            # Make prediction
            prediction_result = self.model.predict(features_array)
            
            # Get explanation if enabled
            explanation = None
            if settings.enable_shap_explanation:
                try:
                    explanation = self.model.explain_prediction(features_array, prediction_result)
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
                "timestamp": time.time()
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
                    "processing_time": processing_time
                }
            )
            
            return result
            
        except InvalidInputException:
            raise
        except Exception as e:
            self.logger.error(
                f"Prediction failed for transaction {transaction_id}: {e}",
                exc_info=True
            )
            raise PredictionFailedException(
                f"Failed to make prediction: {str(e)}",
                details={
                    "transaction_id": transaction_id,
                    "error": str(e)
                }
            )
    
    async def predict_batch(
        self,
        transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
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
                    transaction_id=transaction_id,
                    features=features,
                    metadata=metadata
                )
                predictions.append(result)
            except Exception as e:
                errors.append({
                    "transaction_id": transaction_id,
                    "error": str(e)
                })
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
            "avg_processing_time": total_time / len(transactions) if transactions else 0
        }
    
    def _prepare_features(self, features: List[float]) -> np.ndarray:
        """
        Prepare features for model input.
        
        Args:
            features: Raw features list
            
        Returns:
            Numpy array ready for model
        """
        # Convert to numpy array
        features_array = np.array(features, dtype=np.float32)
        
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
