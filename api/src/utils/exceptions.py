"""
Custom exceptions for the Fraud Detection API.
"""
from typing import Any, Dict, Optional


class FraudDetectionException(Exception):
    """Base exception for fraud detection API."""

    def __init__(
        self,
        message: str,
        error_code: str = "E000",
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize exception.

        Args:
            message: Error message
            error_code: Error code
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ModelNotLoadedException(FraudDetectionException):
    """Exception raised when ML models are not loaded."""

    def __init__(self, message: str = "ML models are not loaded"):
        super().__init__(message=message, error_code="E002")


class InvalidInputException(FraudDetectionException):
    """Exception raised for invalid input data."""

    def __init__(
        self,
        message: str = "Invalid input data",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, error_code="E001", details=details)


class PredictionFailedException(FraudDetectionException):
    """Exception raised when prediction fails."""

    def __init__(
        self,
        message: str = "Prediction failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, error_code="E003", details=details)


class DatabaseException(FraudDetectionException):
    """Exception raised for database errors."""

    def __init__(
        self,
        message: str = "Database operation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, error_code="E004", details=details)


class CacheException(FraudDetectionException):
    """Exception raised for cache errors."""

    def __init__(
        self,
        message: str = "Cache operation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, error_code="E005", details=details)


class TimeoutException(FraudDetectionException):
    """Exception raised when operation times out."""

    def __init__(
        self,
        message: str = "Operation timed out",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, error_code="E006", details=details)


class RateLimitExceededException(FraudDetectionException):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, error_code="E007", details=details)


class UnauthorizedException(FraudDetectionException):
    """Exception raised for unauthorized access."""

    def __init__(
        self,
        message: str = "Unauthorized access",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, error_code="E008", details=details)
