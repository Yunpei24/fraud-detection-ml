"""
Configuration module for Fraud Detection ML training pipeline.
Centralized settings for data paths, model hyperparameters,
training parameters, and MLflow tracking.
"""

from dataclasses import dataclass
from pathlib import Path


# ------------------------------------------------------------
# 1. Data paths
# ------------------------------------------------------------
@dataclass
class DataPaths:
    RAW_DATA: Path = Path("data/raw/creditcard.csv")
    PROCESSED_DATA: Path = Path("data/processed/train.parquet")
    MODELS_DIR: Path = Path("models/")
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
    threshold: float = 0.5
    min_recall: float = 0.95
    max_fpr: float = 0.02


# ------------------------------------------------------------
# 4. MLflow configuration
# ------------------------------------------------------------
@dataclass
class MLflowConfig:
    tracking_uri: str = "file:mlruns"
    experiment_name: str = "fraud_detection_training"
    registered_model_name: str = "fraud_detector"
    register_model: bool = True


# ------------------------------------------------------------
# 5. Central configuration class
# ------------------------------------------------------------
@dataclass
class TrainingConfig:
    """
    Master configuration object to access all sub-configs.
    """
    data: DataPaths = DataPaths()
    model: ModelParameters = ModelParameters()
    training: TrainingParameters = TrainingParameters()
    mlflow: MLflowConfig = MLflowConfig()


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    cfg = TrainingConfig()
    print("Raw data path:", cfg.data.RAW_DATA)
    print("XGBoost params:", cfg.model.xgboost_params)
    print("MLflow experiment:", cfg.mlflow.experiment_name)
