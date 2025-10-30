"""
Validation utilities for the Fraud Detection API.
"""
from typing import List, Dict, Any
import numpy as np

from ..config import constants
from .exceptions import InvalidInputException


def validate_features(features: List[float]) -> bool:
    """Validate 30 transaction features (Time, v1-v28, amount)."""
    if not features:
        raise InvalidInputException("Features list cannot be empty", details={"field": "features"})
    
    if len(features) != constants.EXPECTED_FEATURES:
        raise InvalidInputException(
            f"Expected {constants.EXPECTED_FEATURES} features, got {len(features)}",
            details={
                "field": "features",
                "expected": constants.EXPECTED_FEATURES,
                "received": len(features)
            }
        )
    
    for i, feature in enumerate(features):
        if not isinstance(feature, (int, float)):
            raise InvalidInputException(
                f"Feature at index {i} is not numeric",
                details={"field": "features", "index": i, "type": type(feature).__name__}
            )
        
        if np.isnan(feature) or np.isinf(feature):
            raise InvalidInputException(
                f"Feature at index {i} is NaN or Inf",
                details={"field": "features", "index": i, "value": str(feature)}
            )
    
    return True


def validate_transaction_request(data: Dict[str, Any]) -> bool:
    """
    Validate transaction request data.
    
    Args:
        data: Request data dictionary
        
    Returns:
        True if valid
        
    Raises:
        InvalidInputException: If request is invalid
    """
    # Check required fields
    required_fields = ["transaction_id", "features"]
    for field in required_fields:
        if field not in data:
            raise InvalidInputException(
                f"Missing required field: {field}",
                details={"field": field}
            )
    
    # Validate transaction_id
    transaction_id = data.get("transaction_id", "")
    if not transaction_id or not isinstance(transaction_id, str):
        raise InvalidInputException(
            "transaction_id must be a non-empty string",
            details={"field": "transaction_id"}
        )
    
    if len(transaction_id) > 100:
        raise InvalidInputException(
            "transaction_id too long (max 100 characters)",
            details={
                "field": "transaction_id",
                "length": len(transaction_id)
            }
        )
    
    # Validate features
    features = data.get("features", [])
    validate_features(features)
    
    return True


def validate_prediction_output(output: Dict[str, Any]) -> bool:
    """
    Validate prediction output data.
    
    Args:
        output: Prediction output dictionary
        
    Returns:
        True if valid
        
    Raises:
        InvalidInputException: If output is invalid
    """
    # Check required fields
    required_fields = ["prediction", "confidence", "fraud_score"]
    for field in required_fields:
        if field not in output:
            raise InvalidInputException(
                f"Missing required field in prediction output: {field}",
                details={"field": field}
            )
    
    # Validate prediction
    prediction = output.get("prediction")
    if prediction not in [0, 1]:
        raise InvalidInputException(
            "prediction must be 0 or 1",
            details={
                "field": "prediction",
                "value": prediction
            }
        )
    
    # Validate confidence
    confidence = output.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise InvalidInputException(
            "confidence must be between 0 and 1",
            details={
                "field": "confidence",
                "value": confidence
            }
        )
    
    # Validate fraud_score
    fraud_score = output.get("fraud_score")
    if not isinstance(fraud_score, (int, float)) or not 0 <= fraud_score <= 1:
        raise InvalidInputException(
            "fraud_score must be between 0 and 1",
            details={
                "field": "fraud_score",
                "value": fraud_score
            }
        )
    
    return True


def validate_batch_request(transactions: List[Dict[str, Any]]) -> bool:
    """
    Validate batch prediction request.
    
    Args:
        transactions: List of transaction dictionaries
        
    Returns:
        True if valid
        
    Raises:
        InvalidInputException: If batch request is invalid
    """
    if not transactions:
        raise InvalidInputException(
            "Batch request cannot be empty",
            details={"field": "transactions"}
        )
    
    if len(transactions) > constants.MAX_BATCH_SIZE:
        raise InvalidInputException(
            f"Batch size exceeds maximum of {constants.MAX_BATCH_SIZE}",
            details={
                "field": "transactions",
                "size": len(transactions),
                "max": constants.MAX_BATCH_SIZE
            }
        )
    
    # Validate each transaction
    for i, transaction in enumerate(transactions):
        try:
            validate_transaction_request(transaction)
        except InvalidInputException as e:
            raise InvalidInputException(
                f"Invalid transaction at index {i}: {e.message}",
                details={
                    "index": i,
                    "transaction_id": transaction.get("transaction_id"),
                    "original_error": e.details
                }
            )
    
    return True
