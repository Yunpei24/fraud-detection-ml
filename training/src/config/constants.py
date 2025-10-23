"""
Global constants for the Fraud Detection ML Training Module.
These values remain static throughout the training lifecycle.
"""

from typing import List

# ------------------------------------------------------------
# 1. General Configuration
# ------------------------------------------------------------
TRAIN_TEST_SPLIT: float = 0.8
RANDOM_STATE: int = 42
MIN_RECALL: float = 0.95
MAX_FPR: float = 0.02

# Fraud ratio in the dataset (approx. 0.17%)
FRAUD_RATIO: float = 0.0017

# Default SMOTE sampling strategy
SMOTE_SAMPLING_STRATEGY: float = 0.5  # resample minority class to 50% of majority

# Default threshold for binary classification
DEFAULT_THRESHOLD: float = 0.5


# ------------------------------------------------------------
# 2. Dataset Column Names
# ------------------------------------------------------------
TARGET_COLUMN: str = "Class"
AMOUNT_COLUMN: str = "Amount"
TIME_COLUMN: str = "Time"

# All 28 PCA-transformed feature names (V1–V28)
PCA_FEATURES: List[str] = [
    f"V{i}" for i in range(1, 29)
]

# Full feature set used for training
FEATURE_COLUMNS: List[str] = PCA_FEATURES + [TIME_COLUMN, AMOUNT_COLUMN]

# ------------------------------------------------------------
# 3. Data Validation
# ------------------------------------------------------------
EXPECTED_COLUMNS: List[str] = FEATURE_COLUMNS + [TARGET_COLUMN]

# Minimum number of records required for a valid training dataset
MIN_SAMPLE_SIZE: int = 10_000

# ------------------------------------------------------------
# 4. Evaluation Constraints
# ------------------------------------------------------------
# For model validation before promotion
EVAL_MIN_F1: float = 0.70
EVAL_MIN_PRECISION: float = 0.60
EVAL_MIN_RECALL: float = 0.95
EVAL_MAX_FPR: float = 0.02

# ------------------------------------------------------------
# 5. Logging and Artifacts
# ------------------------------------------------------------
LOG_FILE_NAME: str = "training_log.txt"
MODEL_DIR_NAME: str = "models"
ARTIFACTS_DIR_NAME: str = "training/artifacts"
PLOTS_DIR_NAME: str = "training/plots"
REPORTS_DIR_NAME: str = "training/reports"

# ------------------------------------------------------------
# 6. Miscellaneous
# ------------------------------------------------------------
VERSION: str = "1.0.0"
PROJECT_NAME: str = "Credit Card Fraud Detection"
AUTHOR: str = "Akinrinade Adeleke"
