# training/src/models/hyperparameter_tuning.py
"""
Hypedef tune_xgboost_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_vdef tune_random_forest_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    use_random_search: bool = True,
    n_iter: int = 8,
    cv_folds: int = 2,  # Reduced from 3
    random_state: int = 42
) -> Dict[str, Any]:rray,
    use_random_search: bool = True,
    n_iter: int = 10,
    cv_folds: int = 2,  # Reduced from 3
    random_state: int = 42
) -> Dict[str, Any]: tuning utilities for fraud detection models.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import make_scorer, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def create_custom_scorer():
    """
    Create a custom scorer that balances AUC, recall, and precision for fraud detection.
    """

    def fraud_score(y_true, y_pred_proba):
        """
        Custom scoring function that considers:
        - AUC (primary metric)
        - Recall for fraud detection (important)
        - Precision to avoid false positives
        """
        try:
            if len(y_pred_proba.shape) > 1:
                y_pred_proba = y_pred_proba[:, 1]

            # Use AUC as primary metric for efficiency
            auc = roc_auc_score(y_true, y_pred_proba)
            return auc

        except Exception as e:
            logger.warning(f"Error in fraud_score calculation: {e}")
            # Fallback to AUC only
            try:
                auc = roc_auc_score(
                    y_true,
                    y_pred_proba[:, 1] if len(y_pred_proba.shape) > 1 else y_pred_proba,
                )
                return auc
            except:
                return 0.5  # Neutral score

    # Use response_method instead of deprecated needs_proba for sklearn >= 1.4
    return make_scorer(fraud_score, response_method="predict_proba")


def tune_xgboost_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    use_random_search: bool = True,
    n_iter: int = 3,  # Reduced from 5
    cv_folds: int = 2,
    random_state: int = 24,
) -> Dict[str, Any]:
    """
    Tune XGBoost hyperparameters using grid or random search.

    Returns optimized parameters.
    """
    logger.info("🔍 Tuning XGBoost hyperparameters...")

    # Sample a subset of data for hyperparameter tuning to reduce memory usage
    sample_size = min(50000, len(X_train))  # Limit to 50k samples max
    if len(X_train) > sample_size:
        logger.info(
            f"Sampling {sample_size} examples from {len(X_train)} for hyperparameter tuning"
        )
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(X_train), size=sample_size, replace=False)
        X_train_sample = X_train[sample_indices]
        y_train_sample = y_train[sample_indices]
    else:
        X_train_sample = X_train
        y_train_sample = y_train

    # Base model with current good defaults
    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
        scale_pos_weight=len(y_train_sample[y_train_sample == 0])
        / len(y_train_sample[y_train_sample == 1]),  # Handle imbalance
        # Removed deprecated use_label_encoder parameter
        n_jobs=1,  # Use single thread to reduce memory usage
    )

    if use_random_search:
        # Reduced parameter space for memory efficiency
        param_dist = {
            "n_estimators": [100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.7, 0.8],
            "min_child_weight": [1, 3],
            "gamma": [0.1, 0.2],
            "reg_alpha": [0.01, 0.1],
            "reg_lambda": [0.1, 1],
        }

        search = RandomizedSearchCV(
            base_model,
            param_dist,
            n_iter=n_iter,
            scoring=create_custom_scorer(),
            cv=cv_folds,
            random_state=random_state,
            n_jobs=1,  # Sequential processing to reduce memory usage
            verbose=1,
        )
    else:
        # Grid search (smaller parameter space for faster execution)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
            "min_child_weight": [1, 3],
        }

        search = GridSearchCV(
            base_model,
            param_grid,
            scoring=create_custom_scorer(),
            cv=cv_folds,
            n_jobs=1,  # Sequential processing to reduce memory usage
            verbose=1,
        )

    # Fit search on sampled data
    search.fit(X_train_sample, y_train_sample)

    # Get best parameters
    best_params = search.best_params_
    best_score = search.best_score_

    logger.info(f" XGBoost tuning complete. Best score: {best_score:.4f}")
    logger.info(f"   Best params: {best_params}")

    # Evaluate on validation set
    best_model = search.best_estimator_
    val_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)

    precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores, nan=0)
    best_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_threshold_idx]

    logger.info(f"   Validation AUC: {val_auc:.4f}")
    logger.info(f"   Optimal threshold: {optimal_threshold:.4f}")

    return {
        "best_params": best_params,
        "best_cv_score": best_score,
        "val_auc": val_auc,
        "optimal_threshold": optimal_threshold,
        "best_model": best_model,
    }


def tune_random_forest_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    use_random_search: bool = True,
    n_iter: int = 3,  # Reduced from 5
    cv_folds: int = 2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Tune Random Forest hyperparameters.
    """
    logger.info("🔍 Tuning Random Forest hyperparameters...")

    # Sample a subset of data for hyperparameter tuning to reduce memory usage
    sample_size = min(50000, len(X_train))  # Limit to 50k samples max
    if len(X_train) > sample_size:
        logger.info(
            f"Sampling {sample_size} examples from {len(X_train)} for hyperparameter tuning"
        )
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(X_train), size=sample_size, replace=False)
        X_train_sample = X_train[sample_indices]
        y_train_sample = y_train[sample_indices]
    else:
        X_train_sample = X_train
        y_train_sample = y_train

    base_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=1,  # Use single core to reduce memory usage
        class_weight="balanced",
    )

    if use_random_search:
        # Reduced parameter space for memory efficiency
        param_dist = {
            "n_estimators": [100, 200],  # Reduced options
            "max_depth": [10, 15],  # Reduced options
            "min_samples_split": [2, 5],  # Reduced options
            "min_samples_leaf": [1, 2],  # Reduced options
            "max_features": ["sqrt"],  # Single option
            "bootstrap": [True],  # Single option
        }

        search = RandomizedSearchCV(
            base_model,
            param_dist,
            n_iter=n_iter,
            scoring=create_custom_scorer(),
            cv=cv_folds,
            random_state=random_state,
            n_jobs=1,  # Sequential processing to reduce memory usage
            verbose=1,
        )
    else:
        # Simplified grid search
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 15],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
        }

        search = GridSearchCV(
            base_model,
            param_grid,
            scoring=create_custom_scorer(),
            cv=cv_folds,
            n_jobs=1,  # Sequential processing to reduce memory usage
            verbose=1,
        )

    # Fit search on sampled data
    search.fit(X_train_sample, y_train_sample)

    best_params = search.best_params_
    best_score = search.best_score_

    logger.info(f" Random Forest tuning complete. Best score: {best_score:.4f}")
    logger.info(f"   Best params: {best_params}")

    # Validation evaluation
    best_model = search.best_estimator_
    val_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)

    logger.info(f"   Validation AUC: {val_auc:.4f}")

    return {
        "best_params": best_params,
        "best_cv_score": best_score,
        "val_auc": val_auc,
        "best_model": best_model,
    }


def tune_neural_network_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cv_folds: int = 2,
    random_state: int = 24,
) -> Dict[str, Any]:
    """
    Tune Neural Network hyperparameters using grid search.
    """
    logger.info(" Tuning Neural Network hyperparameters...")

    # Sample a smaller subset for neural network tuning (they're very memory intensive)
    sample_size = min(25000, len(X_train))  # Limit to 25k samples for NN
    if len(X_train) > sample_size:
        logger.info(
            f"Sampling {sample_size} examples from {len(X_train)} for neural network hyperparameter tuning"
        )
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(X_train), size=sample_size, replace=False)
        X_train_sample = X_train[sample_indices]
        y_train_sample = y_train[sample_indices]
    else:
        X_train_sample = X_train
        y_train_sample = y_train

    base_model = MLPClassifier(random_state=random_state)

    # Neural networks are expensive, so use smaller parameter space
    param_grid = {
        "hidden_layer_sizes": [(64,), (64, 32)],  # Reduced options
        "activation": ["relu"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.001],
        "learning_rate": ["constant"],
        "learning_rate_init": [0.001],
        "batch_size": [64, 128],  # Smaller batches
        "max_iter": [200],  # Limit iterations
    }

    search = GridSearchCV(
        base_model,
        param_grid,
        scoring=create_custom_scorer(),
        cv=cv_folds,
        n_jobs=1,  # Sequential processing to reduce memory usage
        verbose=1,
    )

    # Fit on sampled data
    search.fit(X_train_sample, y_train_sample)

    best_params = search.best_params_
    best_score = search.best_score_

    logger.info(f" Neural Network tuning complete. Best score: {best_score:.4f}")
    logger.info(f"   Best params: {best_params}")

    # Validation evaluation
    best_model = search.best_estimator_
    val_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)

    logger.info(f"   Validation AUC: {val_auc:.4f}")

    return {
        "best_params": best_params,
        "best_cv_score": best_score,
        "val_auc": val_auc,
        "best_model": best_model,
    }


def tune_isolation_forest_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Tune Isolation Forest hyperparameters using validation-based approach.

    For unsupervised anomaly detection, we evaluate based on validation set performance
    with different contamination rates and decision thresholds.
    """
    logger.info("🔍 Tuning Isolation Forest hyperparameters...")

    # Get actual fraud rate from training data for better contamination estimation
    fraud_rate = np.mean(y_train)
    logger.info(f"Training data fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")

    # Test different contamination rates around the actual fraud rate
    contamination_candidates = [
        max(0.001, fraud_rate * 0.5),  # Half the fraud rate
        fraud_rate,  # Actual fraud rate
        min(0.05, fraud_rate * 2),  # Double the fraud rate (max 5%)
        min(0.1, fraud_rate * 3),  # Triple (max 10%)
    ]
    contamination_candidates = list(set(contamination_candidates))  # Remove duplicates

    best_score = -np.inf
    best_params = None
    best_model = None
    best_threshold = None

    # Parameter combinations to test
    param_combinations = [
        {"n_estimators": 200, "max_samples": 0.8, "max_features": 0.8},
        {"n_estimators": 300, "max_samples": 0.6, "max_features": 0.9},
        {"n_estimators": 400, "max_samples": 0.7, "max_features": 0.7},
    ]

    for params in param_combinations:
        for contamination in contamination_candidates:
            # Create model
            model_params = {
                **params,
                "contamination": contamination,
                "random_state": random_state,
                "n_jobs": 1,
            }

            model = IsolationForest(**model_params)

            # Fit on training data
            model.fit(X_train)

            # Get anomaly scores on validation set
            val_scores = model.decision_function(X_val)

            # Test different thresholds for optimal fraud detection
            # Use percentiles of scores as threshold candidates
            score_percentiles = np.percentiles(val_scores, [5, 10, 20, 30, 50])

            for threshold in score_percentiles:
                # Predict fraud (anomaly score <= threshold)
                val_predictions = (val_scores <= threshold).astype(int)

                # Calculate metrics
                tp = np.sum((val_predictions == 1) & (y_val == 1))
                fp = np.sum((val_predictions == 1) & (y_val == 0))
                fn = np.sum((val_predictions == 0) & (y_val == 1))
                tn = np.sum((val_predictions == 0) & (y_val == 0))

                if tp + fn > 0:  # Avoid division by zero
                    recall = tp / (tp + fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                    # Score: prioritize recall but penalize high FPR
                    score = recall * 0.7 + precision * 0.2 - fpr * 0.1

                    if score > best_score:
                        best_score = score
                        best_params = model_params
                        best_model = model
                        best_threshold = threshold

    logger.info(f" Isolation Forest tuning complete. Best score: {best_score:.4f}")
    logger.info(f"   Best params: {best_params}")
    logger.info(f"   Best threshold: {best_threshold:.4f}")

    return {
        "best_params": best_params,
        "best_cv_score": best_score,
        "best_model": best_model,
        "optimal_threshold": best_threshold,
    }
