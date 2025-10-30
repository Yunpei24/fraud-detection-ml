"""
Configuration module for Fraud Detection ML training pipeline.
Migrated to use centralized configuration for consistency.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

# Add project root to path to import centralized config
# project_root = Path(__file__).parent.parent.parent.parent
# sys.path.insert(0, str(project_root))

from config import get_settings
# try:
#     from config import get_settings
# except ImportError:
#     # Fallback for when running from training directory
#     import sys
#     sys.path.insert(0, str(project_root / "config"))
#     from config import get_settings

# Get centralized settings
settings = get_settings()

# ------------------------------------------------------------
# 1. Data paths
# ------------------------------------------------------------
@dataclass
class DataPaths:
    RAW_DATA: Path = Path(settings.training.train_data_path).parent / "creditcard.csv"
    PROCESSED_DATA: Path = Path("data/processed/train.parquet")
    MODELS_DIR: Path = Path(settings.training.model_output_dir)
    ARTIFACTS_DIR: Path = Path("training/artifacts/")
    SCALER_PATH: Path = Path("training/artifacts/scaler.pkl")
    LOG_DIR: Path = Path("training/logs/")


# ------------------------------------------------------------
# 2. Model hyperparameters
# ------------------------------------------------------------
@dataclass
class ModelParameters:
    xgboost_params: dict = None
    random_forest_params: dict = None
    mlp_params: dict = None
    isolation_forest_params: dict = None

    def __post_init__(self):
        self.xgboost_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        self.random_forest_params = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.mlp_params = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 200,
            "random_state": 42,
        }
        self.isolation_forest_params = {
            "n_estimators": 300,
            "contamination": 0.0017,  # fraud ratio ~0.17%
            "max_samples": "auto",
            "random_state": 42,
        }


# ------------------------------------------------------------
# 3. Training parameters
# ------------------------------------------------------------
@dataclass
class TrainingParameters:
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    use_smote: bool = True
    target_col: str = "Class"
    threshold: float = settings.training.eval_threshold
    min_recall: float = 0.95
    max_fpr: float = 0.02


# ------------------------------------------------------------
# 4. MLflow configuration
# ------------------------------------------------------------
@dataclass
class MLflowConfig:
    tracking_uri: str = settings.mlflow.tracking_uri
    experiment_name: str = settings.mlflow.experiment_name
    registered_model_name: str = settings.mlflow.model_name
    register_model: bool = True


# ------------------------------------------------------------
# 5. Central configuration class
# ------------------------------------------------------------
@dataclass
class TrainingConfig:
    """
    Master configuration object to access all sub-configs.
    Migrated to use centralized settings.
    """
    data: DataPaths = DataPaths()
    model: ModelParameters = ModelParameters()
    training: TrainingParameters = TrainingParameters()
    mlflow: MLflowConfig = MLflowConfig()

    # Additional centralized settings
    batch_size: int = settings.training.batch_size
    epochs: int = settings.training.epochs
    validation_split: float = settings.training.validation_split
    early_stopping_patience: int = settings.training.early_stopping_patience
    cv_folds: int = settings.training.cv_folds
    enable_feature_selection: bool = settings.training.enable_feature_selection
    enable_scaling: bool = settings.training.enable_scaling
    prometheus_port: int = settings.training.prometheus_port


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    cfg = TrainingConfig()
    print("Raw data path:", cfg.data.RAW_DATA)
    print("XGBoost params:", cfg.model.xgboost_params)
    print("MLflow experiment:", cfg.mlflow.experiment_name)
