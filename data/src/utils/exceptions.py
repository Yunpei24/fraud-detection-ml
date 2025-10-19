"""
Custom exception classes for the data module.
"""


class DataModuleException(Exception):
    """Base exception for all data module errors."""
    pass


class IngestionException(DataModuleException):
    """Raised when data ingestion fails."""
    pass


class ValidationException(DataModuleException):
    """Raised when data validation fails."""
    pass


class TransformationException(DataModuleException):
    """Raised when data transformation fails."""
    pass


class StorageException(DataModuleException):
    """Raised when storage operations fail."""
    pass


class ConfigurationException(DataModuleException):
    """Raised when configuration is invalid."""
    pass


class DataQualityException(DataModuleException):
    """Raised when data quality checks fail."""
    pass
