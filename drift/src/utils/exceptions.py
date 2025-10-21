"""
Custom exceptions for drift detection component.
"""
from typing import Optional, Dict, Any


class DriftDetectionException(Exception):
    """Base exception for drift detection."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "D000",
        details: Optional[Dict[str, Any]] = None
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


class InsufficientDataException(DriftDetectionException):
    """Exception raised when insufficient data for drift detection."""
    
    def __init__(
        self,
        message: str = "Insufficient data for drift detection",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="D001",
            details=details
        )


class DatabaseException(DriftDetectionException):
    """Exception raised for database errors."""
    
    def __init__(
        self,
        message: str = "Database operation failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="D002",
            details=details
        )


class AlertException(DriftDetectionException):
    """Exception raised when alert sending fails."""
    
    def __init__(
        self,
        message: str = "Failed to send alert",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="D003",
            details=details
        )


class RetrainingTriggerException(DriftDetectionException):
    """Exception raised when retraining trigger fails."""
    
    def __init__(
        self,
        message: str = "Failed to trigger retraining",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="D004",
            details=details
        )


class MetricsComputationException(DriftDetectionException):
    """Exception raised when metrics computation fails."""
    
    def __init__(
        self,
        message: str = "Failed to compute drift metrics",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="D005",
            details=details
        )


class StorageException(DriftDetectionException):
    """Exception raised for storage errors."""
    
    def __init__(
        self,
        message: str = "Storage operation failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="D006",
            details=details
        )
