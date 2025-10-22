# training/src/utils/exceptions.py

class ConfigError(Exception):
    """Configuration-related errors (missing/invalid keys, paths, etc.)."""


class DataLoadError(Exception):
    """Errors while loading data from local/remote sources."""


class PreprocessError(Exception):
    """Errors during data cleaning / schema validation / splitting."""


class FeatureError(Exception):
    """Errors building features or applying scalers/encoders."""


class TrainingError(Exception):
    """Failures during model training or inference."""


class TuningError(Exception):
    """Failures during hyperparameter tuning (Optuna, search spaces)."""


class ValidationError(Exception):
    """Model does not meet business constraints (recall, FPR, etc.)."""


class RegistryError(Exception):
    """Issues with MLflow model registry actions."""
