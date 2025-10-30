"""
Helper utilities for the Fraud Detection API.
"""
import json
import time
from datetime import datetime
from typing import Any, Dict, List


def format_response(
    data: Any, status_code: int = 200, message: str = None
) -> Dict[str, Any]:
    """
    Format API response with consistent structure.

    Args:
        data: Response data
        status_code: HTTP status code
        message: Optional message

    Returns:
        Formatted response dictionary
    """
    response = {
        "data": data,
        "status_code": status_code,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if message:
        response["message"] = message

    return response


def calculate_processing_time(start_time: float) -> float:
    """
    Calculate processing time in seconds.

    Args:
        start_time: Start time (from time.time())

    Returns:
        Processing time in seconds
    """
    return round(time.time() - start_time, 4)


def sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize input data by removing/masking sensitive fields.

    Args:
        data: Input data dictionary

    Returns:
        Sanitized data dictionary
    """
    # Fields to remove completely
    sensitive_fields = ["password", "api_key", "token", "secret"]

    # Fields to mask (show only last 4 characters)
    maskable_fields = ["card_number", "account_number"]

    sanitized = data.copy()

    for field in sensitive_fields:
        if field in sanitized:
            sanitized.pop(field)

    for field in maskable_fields:
        if field in sanitized and isinstance(sanitized[field], str):
            value = sanitized[field]
            if len(value) > 4:
                sanitized[field] = "*" * (len(value) - 4) + value[-4:]

    return sanitized


def extract_features_metadata(features: List[float]) -> Dict[str, Any]:
    """
    Extract metadata from feature values.

    Args:
        features: List of feature values

    Returns:
        Metadata dictionary
    """
    return {
        "feature_count": len(features),
        "min_value": min(features),
        "max_value": max(features),
        "mean_value": sum(features) / len(features) if features else 0,
        "has_negative": any(f < 0 for f in features),
        "has_zero": any(f == 0 for f in features),
    }


def format_explanation(explanation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format SHAP explanation for human readability.

    Args:
        explanation: Raw explanation dictionary

    Returns:
        Formatted explanation
    """
    if not explanation or "top_features" not in explanation:
        return {"message": "No explanation available"}

    top_features = explanation.get("top_features", [])

    formatted = {
        "summary": f"Top {len(top_features)} influential features",
        "features": [],
    }

    for feature in top_features:
        formatted["features"].append(
            {
                "index": feature.get("feature"),
                "impact": feature.get("shap_value"),
                "direction": "increases"
                if feature.get("shap_value", 0) > 0
                else "decreases",
                "strength": abs(feature.get("shap_value", 0)),
            }
        )

    return formatted


def batch_predictions_to_csv(predictions: List[Dict[str, Any]]) -> str:
    """
    Convert batch predictions to CSV format.

    Args:
        predictions: List of prediction dictionaries

    Returns:
        CSV string
    """
    if not predictions:
        return "transaction_id,prediction,confidence,fraud_score\n"

    # Header
    csv_lines = ["transaction_id,prediction,confidence,fraud_score"]

    # Data rows
    for pred in predictions:
        line = f"{pred.get('transaction_id', '')},"
        line += f"{pred.get('prediction', '')},"
        line += f"{pred.get('confidence', '')},"
        line += f"{pred.get('fraud_score', '')}"
        csv_lines.append(line)

    return "\n".join(csv_lines)


def generate_transaction_id(prefix: str = "TXN") -> str:
    """
    Generate a unique transaction ID.

    Args:
        prefix: Prefix for the transaction ID

    Returns:
        Generated transaction ID
    """
    timestamp = int(time.time() * 1000)  # Milliseconds
    return f"{prefix}-{timestamp}"


def merge_metadata(
    request_metadata: Dict[str, Any], prediction_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge request and prediction metadata.

    Args:
        request_metadata: Metadata from the request
        prediction_metadata: Metadata from the prediction

    Returns:
        Merged metadata dictionary
    """
    merged = {}

    # Add request metadata
    if request_metadata:
        merged["request"] = request_metadata

    # Add prediction metadata
    if prediction_metadata:
        merged["prediction"] = prediction_metadata

    # Add timestamp
    merged["processed_at"] = datetime.utcnow().isoformat()

    return merged


def validate_json_structure(data: str) -> bool:
    """
    Validate if string is valid JSON.

    Args:
        data: JSON string

    Returns:
        True if valid JSON, False otherwise
    """
    try:
        json.loads(data)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def truncate_string(text: str, max_length: int = 100) -> str:
    """
    Truncate string to maximum length.

    Args:
        text: Input string
        max_length: Maximum length

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def get_fraud_risk_level(fraud_score: float) -> str:
    """
    Get human-readable fraud risk level.

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
