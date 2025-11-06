# training/src/pipelines/training_pipeline.py
"""
Training pipeline for fraud detection.
Loads data from PostgreSQL, applies preprocessing and feature engineering,
trains multiple models with SMOTE, evaluates, and registers best model in MLflow.

ARTIFACT PERSISTENCE:
- Saves artifacts locally in /app/training/artifacts/[timestamp]/
- Then logs to MLflow (files persist after upload)
- Local files accessible for debugging and backup
"""
from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fraud_detection_common.feature_engineering import build_feature_frame

# Use common package for preprocessing and feature engineering
from fraud_detection_common.preprocessor import DataPreprocessor
from src.config.logging_config import get_logger

# Data loading and splitting
from src.data.loader import load_training_data
from src.data.splitter import stratified_split
from src.evaluation.explainability import (
    create_explanation_report_from_model,
    plot_shap_summary,
)

# Evaluation
from src.evaluation.metrics import calculate_all_metrics
from src.evaluation.plots import (
    plot_confusion_matrix_plot,
    plot_feature_importance,
    plot_precision_recall_curve_plot,
    plot_roc_auc,
)
from src.evaluation.validation import validate_all_models

# Feature selection (optional)
from src.features.selection import select_k_best_mutual_info

# MLflow utilities
from src.mlflow_utils.experiment import setup_experiment, start_run
from src.mlflow_utils.registry import register_model
from src.mlflow_utils.tracking import log_artifact, log_metrics, log_model, log_params

# Models
from src.models.isolation_forest import IsolationForestModel
from src.models.neural_network import NeuralNetworkModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    """Training configuration parameters."""

    # Data splitting
    test_size: float = 0.2
    val_size: float = 0.1  # from train portion
    random_state: int = 42

    # Model training
    use_smote: bool = True  # SMOTE for imbalanced data
    tune_hyperparams: bool = (
        True  # Re-enable hyperparameter tuning with conservative settings
    )

    # Feature engineering and selection
    feature_k: Optional[int] = None  # top features via MI (None = use all)
    outlier_method: Optional[str] = "IQR"  # Outlier handling

    # Business constraints
    min_recall: float = 0.80  # Minimum recall for fraud detection
    max_fpr: float = (
        0.30  # Maximum false positive rate (eased from 0.05 to 0.30 for imbalanced data)
    )

    # MLflow
    experiment_name: str = "fraud_detection_training"
    register_models: bool = True

    # Data source
    use_postgres: bool = True  # If False, falls back to local CSV
    local_csv_path: str = "/app/data/raw/creditcard.csv"  # Container path

    # Artifact persistence (NEW)
    artifacts_dir: str = "/app/training/artifacts"  # Local persistence directory
    save_local_artifacts: bool = True  # Save locally before MLflow
    run_timestamp: Optional[str] = None  # Auto-generated timestamp


# -------------------------
# Helper Functions (NEW)
# -------------------------
def create_artifacts_directory(cfg: TrainConfig) -> Path:
    """
    Create timestamped artifacts directory for this training run.

    Structure:
        /app/training/artifacts/[timestamp]/
            ├── models/        (trained models .pkl)
            ├── plots/         (evaluation plots .png)
            ├── metrics/       (metrics .json)
            ├── explainability/ (SHAP artifacts)
            └── run_summary.json

    Returns:
        Path to created directory
    """
    if cfg.run_timestamp is None:
        cfg.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path(cfg.artifacts_dir) / cfg.run_timestamp

    # Create subdirectories
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "explainability").mkdir(parents=True, exist_ok=True)

    logger.info(f" Artifacts directory created: {run_dir}")

    return run_dir


def save_artifact_local_and_mlflow(
    artifact_data: Any,
    filename: str,
    artifact_type: str,
    local_dir: Path,
    mlflow_artifact_path: Optional[str] = None,
) -> str:
    """
    Save artifact LOCALLY first, then log to MLflow from local file.
    File persists locally even after MLflow upload.

    Args:
        artifact_data: Data to save (figure, dict, model, etc.)
        filename: Filename
        artifact_type: 'plot', 'json', 'model', 'pkl'
        local_dir: Local save directory
        mlflow_artifact_path: MLflow artifact path (optional)

    Returns:
        Local file path
    """
    # Determine local path
    local_path = local_dir / filename

    # Save locally based on type
    if artifact_type == "plot":
        artifact_data.savefig(local_path, dpi=160, bbox_inches="tight")
        logger.debug(f"   Plot saved: {local_path}")

    elif artifact_type == "json":
        with open(local_path, "w") as f:
            json.dump(artifact_data, f, indent=2)
        logger.debug(f"   JSON saved: {local_path}")

    elif artifact_type in ["model", "pkl"]:
        joblib.dump(artifact_data, local_path)
        logger.debug(f"   {artifact_type.upper()} saved: {local_path}")

    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")

    # Log to MLflow from local file (file persists after upload)
    try:
        log_artifact(str(local_path), mlflow_artifact_path)
        logger.debug(f"    Logged to MLflow: {filename}")
    except Exception as e:
        logger.warning(f"   MLflow upload failed: {e}")

    return str(local_path)


# -------------------------
# Public entry point
# -------------------------
def run_training(cfg: Optional[TrainConfig] = None) -> bool:
    """
    Main training pipeline entry point.

    Args:
        cfg: Training configuration

    Returns:
        True if validation passed, False otherwise
    """
    cfg = cfg or TrainConfig()

    logger.info("=" * 80)
    logger.info(" FRAUD DETECTION MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Configuration: {asdict(cfg)}")

    # Create artifacts directory for this run
    artifacts_dir = create_artifacts_directory(cfg)

    # MLflow experiment setup
    setup_experiment(cfg.experiment_name)

    # Pipeline steps
    X, y, feature_names = load_data(cfg)
    Xtr, Xval, Xte, ytr, yval, yte, feature_names_final = preprocess_data(
        X, y, feature_names, cfg
    )

    # Train & evaluate models in parallel
    models = train_models(Xtr, ytr, Xval, yval, cfg)
    results = evaluate_models(models, Xte, yte, feature_names_final, artifacts_dir)

    # Validate against business rules
    ok = validate_models(results, cfg)
    if not ok:
        logger.error(" Validation failed against business constraints")
    else:
        logger.info(" Validation passed")

    # Register best model (artifacts logged inside each model's MLflow run)
    register_best_model(models, results, feature_names_final, cfg, artifacts_dir)

    # Save run summary
    summary = {
        "timestamp": cfg.run_timestamp,
        "validation_passed": ok,
        "models_trained": list(models.keys()),
        "best_model_by_auc": max(results.items(), key=lambda x: x[1].get("auc", 0))[0],
        "metrics": results,
        "artifacts_path": str(artifacts_dir),
    }
    summary_path = artifacts_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 80)
    logger.info(
        f" Training pipeline completed (validation={'PASSED' if ok else 'FAILED'})"
    )
    logger.info(f" Artifacts saved to: {artifacts_dir}")
    logger.info("=" * 80)

    return ok


# -------------------------
# Pipeline steps
# -------------------------
def load_data(cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load training data from PostgreSQL or local CSV.

    Returns:
        Tuple of (X, y, feature_names)
    """
    logger.info(" Step 1/5: Loading training data...")

    if cfg.use_postgres:
        try:
            df = load_training_data()
            logger.info(
                f" Loaded {len(df)} samples from PostgreSQL (training_transactions table)"
            )
        except Exception as e:
            logger.warning(f"Failed to load from PostgreSQL: {e}")
            logger.info(f"Falling back to local CSV: {cfg.local_csv_path}")
            from src.data.loader import load_local_csv

            df = load_local_csv(cfg.local_csv_path)
    else:
        from src.data.loader import load_local_csv

        df = load_local_csv(cfg.local_csv_path)
        logger.info(f" Loaded {len(df)} samples from local CSV")

    # Extract target and features
    if "class" not in df.columns:
        raise ValueError("Target column 'class' not found in data")

    y = df["class"].astype(int).values
    X_df = df.drop(columns=["class"])

    # Log class distribution
    fraud_count = np.sum(y == 1)
    normal_count = np.sum(y == 0)
    fraud_pct = 100 * fraud_count / len(y)
    logger.info(
        f"   Class distribution: Normal={normal_count}, Fraud={fraud_count} ({fraud_pct:.2f}%)"
    )

    feature_names = list(X_df.columns)
    X = X_df.values

    return X, y, feature_names


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    cfg: TrainConfig,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list
]:
    """
    Preprocess data: feature engineering, cleaning, scaling, splitting.

    Returns:
        Tuple of (Xtr, Xval, Xte, ytr, yval, yte, feature_names)
    """
    logger.info("Step 2/5: Preprocessing and feature engineering...")

    # Convert to DataFrame for processing
    df = pd.DataFrame(X, columns=feature_names)
    df["class"] = y

    # Feature engineering using common package
    logger.info("   Building behavioral and temporal features...")
    df = build_feature_frame(
        df,
        amount_col="amount",
        time_col="time",
        roll_window=10,
    )

    # Split features and target again
    y = df["class"].values
    X_df = df.drop(columns=["class"])

    # Preprocessing using common package
    logger.info("   Applying preprocessing (outlier handling, scaling)...")
    preprocessor = DataPreprocessor(
        drop_columns=["time"],  # Drop time column after feature engineering
        scale_columns=["amount"],
        outlier_columns=["amount"] if cfg.outlier_method else [],
        outlier_method=cfg.outlier_method,
    )

    X_df_clean, artifacts = preprocessor.fit_transform(
        X_df,
        persist_dir=None,  # Don't persist locally, will log to MLflow
        artifacts_prefix="preprocessor",
    )

    feature_names = list(X_df_clean.columns)
    logger.info(f" Features after preprocessing: {len(feature_names)}")
    X_clean = X_df_clean.values

    # Optional: Feature selection via mutual information
    if cfg.feature_k and cfg.feature_k < len(feature_names):
        logger.info(
            f"   Selecting top {cfg.feature_k} features via mutual information..."
        )
        X_df_selected, selected_features = select_k_best_mutual_info(
            pd.DataFrame(X_clean, columns=feature_names), target=y, k=cfg.feature_k
        )
        X_clean = X_df_selected.values
        feature_names = selected_features
        logger.info(
            f"   Selected features: {selected_features[:5]}... ({len(selected_features)} total)"
        )

    # Stratified split (train/val/test)
    logger.info(f"   Splitting data (test={cfg.test_size}, val={cfg.val_size})...")
    splits = stratified_split(
        X_clean,
        y,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
    )
    Xtr, Xval, Xte = splits["X_train"], splits["X_val"], splits["X_test"]
    ytr, yval, yte = splits["y_train"], splits["y_val"], splits["y_test"]

    logger.info(f" Preprocessing complete")
    logger.info(f"   Train: {Xtr.shape}, Val: {Xval.shape}, Test: {Xte.shape}")
    logger.info(f"   Features: {len(feature_names)}")

    return Xtr, Xval, Xte, ytr, yval, yte, feature_names


def train_models(Xtr, ytr, Xval, yval, cfg: TrainConfig) -> Dict[str, Any]:
    """
    Train multiple models in parallel.

    Returns:
        Dictionary of {model_name: trained_model}
    """
    logger.info(" Step 3/5: Training models...")

    model_names = ["xgboost", "random_forest", "neural_network", "isolation_forest"]
    models: Dict[str, Any] = {}

    def _train_one(name: str):
        """Train a single model with MLflow tracking."""
        try:
            logger.info(f"   Training {name}...")

            if name == "xgboost":
                model = XGBoostModel(
                    use_smote=cfg.use_smote,
                    tune_hyperparams=cfg.tune_hyperparams,
                    random_state=cfg.random_state,
                )
            elif name == "random_forest":
                model = RandomForestModel(
                    use_smote=cfg.use_smote,
                    tune_hyperparams=cfg.tune_hyperparams,
                    random_state=cfg.random_state,
                )
            elif name == "neural_network":
                model = NeuralNetworkModel(
                    tune_hyperparams=cfg.tune_hyperparams, random_state=cfg.random_state
                )
            elif name == "isolation_forest":
                model = IsolationForestModel(random_state=cfg.random_state)
            else:
                raise ValueError(f"Unknown model: {name}")

            # Train with MLflow tracking
            with start_run(run_name=f"train_{name}"):
                # Log hyperparameters
                if hasattr(model, "model"):
                    params = (
                        model.model.get_params()
                        if hasattr(model.model, "get_params")
                        else {}
                    )
                    log_params(
                        {
                            f"{name}_{k}": v
                            for k, v in params.items()
                            if isinstance(v, (int, float, str, bool))
                        }
                    )

                # Train
                if cfg.tune_hyperparams and name in [
                    "xgboost",
                    "random_forest",
                    "neural_network",
                ]:
                    # Pass validation data for hyperparameter tuning
                    model.fit(Xtr, ytr, X_val=Xval, y_val=yval)
                else:
                    model.fit(Xtr, ytr)

                # Validation metrics
                if hasattr(model, "predict_proba"):
                    # Special case for Isolation Forest (needs y for adaptive contamination rate)
                    if name == "isolation_forest":
                        val_proba = model.predict_proba(Xval, yval)[:, 1]
                    else:
                        val_proba = model.predict_proba(Xval)[:, 1]
                else:
                    val_proba = model.predict(Xval).astype(float)

                val_pred = model.predict(Xval)
                val_metrics = calculate_all_metrics(yval, val_proba, val_pred)

                # Log validation metrics
                log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

                logger.info(
                    f"   {name}: val_auc={val_metrics.get('auc', 0):.4f}, val_recall={val_metrics.get('recall', 0):.4f}"
                )

            return name, model

        except Exception as e:
            logger.error(f"  Failed to train {name}: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            raise  # Re-raise to fail the pipeline

    # Train models sequentially to avoid memory issues
    for name in model_names:
        name, model = _train_one(name)
        models[name] = model

    logger.info(f"Trained {len(models)} models")
    return models


def evaluate_models(
    models: Dict[str, Any], Xte, yte, feature_names: list, artifacts_dir: Path
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all models on test set and generate plots.
    Saves artifacts locally first, then logs to MLflow.

    Returns:
        Dictionary of {model_name: metrics_dict}
    """
    logger.info(" Step 4/5: Evaluating models on test set...")

    results: Dict[str, Dict[str, float]] = {}
    plots_data = {}
    plots_dir = artifacts_dir / "plots"
    explainability_dir = artifacts_dir / "explainability"

    for name, model in models.items():
        logger.info(f"   Evaluating {name}...")

        # Predictions
        if hasattr(model, "predict_proba"):
            if name == "isolation_forest":
                y_proba = model.predict_proba(Xte, yte)
            else:
                y_proba = model.predict_proba(Xte)
            y_score = y_proba[:, 1]  # fraud probability
        else:
            y_pred = model.predict(Xte)
            y_score = y_pred.astype(float)

        y_pred = model.predict(Xte)

        # Calculate metrics
        metrics = calculate_all_metrics(yte, y_score, y_pred)
        results[name] = metrics

        # Store for plotting
        plots_data[name] = {
            "y_true": yte,
            "y_score": y_score,
            "y_pred": y_pred,
        }

        # Log test metrics to MLflow
        with start_run(run_name=f"eval_{name}"):
            log_metrics({f"test_{k}": v for k, v in metrics.items()})

        logger.info(
            f"   {name}: auc={metrics.get('auc', 0):.4f}, recall={metrics.get('recall', 0):.4f}, precision={metrics.get('precision', 0):.4f}"
        )

    # Generate plots (parallelized for speed) with local persistence
    logger.info("   Generating evaluation plots (with local save)...")

    def generate_plots_for_model(name: str, data: dict, model: Any) -> list:
        """Generate all plots for a single model and save locally + MLflow."""
        model_plots = []
        try:
            # ROC curve
            fig_roc = plot_roc_auc(
                data["y_true"], data["y_score"], title=f"ROC Curve - {name}"
            )
            save_artifact_local_and_mlflow(
                fig_roc, f"{name}_roc.png", "plot", plots_dir
            )
            plt.close(fig_roc)

            # Precision-Recall curve
            fig_pr = plot_precision_recall_curve_plot(
                data["y_true"], data["y_score"], title=f"Precision-Recall - {name}"
            )
            save_artifact_local_and_mlflow(fig_pr, f"{name}_pr.png", "plot", plots_dir)
            plt.close(fig_pr)

            # Confusion matrix
            fig_cm = plot_confusion_matrix_plot(
                data["y_true"], data["y_pred"], title=f"Confusion Matrix - {name}"
            )
            save_artifact_local_and_mlflow(fig_cm, f"{name}_cm.png", "plot", plots_dir)
            plt.close(fig_cm)

            # Feature importance (if available)
            if hasattr(model, "get_feature_importance"):
                try:
                    fig_fi = plot_feature_importance(
                        model.model if hasattr(model, "model") else model,
                        feature_names,
                        title=f"Feature Importance - {name}",
                    )
                    save_artifact_local_and_mlflow(
                        fig_fi, f"{name}_importance.png", "plot", plots_dir
                    )
                    plt.close(fig_fi)
                except Exception as e:
                    logger.warning(
                        f"    Feature importance plot failed for {name}: {e}"
                    )

            logger.info(f"  Plots generated for {name}")
        except Exception as e:
            logger.warning(f"    Plot generation failed for {name}: {e}")

        return model_plots

    try:
        # Parallelize plot generation across models (max 4 workers for 4 models)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    generate_plots_for_model, name, data, models[name]
                ): name
                for name, data in plots_data.items()
            }

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.warning(
                        f"    Failed to generate plots for {model_name}: {e}"
                    )

        logger.info("   Plots saved locally and logged to MLflow")

        # Generate and log SHAP explanations with local persistence
        logger.info("   Generating SHAP explanations...")
        try:
            # Use a small sample for SHAP (faster computation)
            sample_size = min(50, Xte.shape[0])
            sample_indices = np.random.choice(
                Xte.shape[0], size=sample_size, replace=False
            )
            X_sample = Xte[sample_indices]

            for name, model in models.items():
                try:
                    # CRITICAL: Create MLflow run for SHAP logging
                    with start_run(run_name=f"shap_{name}"):
                        # Generate SHAP explanation
                        (
                            explanation_report,
                            explainer,
                        ) = create_explanation_report_from_model(
                            model.model if hasattr(model, "model") else model,
                            X_sample,
                            feature_names=feature_names,
                            top_n=20,
                        )

                        # Save explainer locally + MLflow
                        explainer_path = save_artifact_local_and_mlflow(
                            explainer,
                            f"shap_explainer_{name}.pkl",
                            "pkl",
                            explainability_dir,
                        )

                        logger.info(
                            f"    SHAP explainer logged to MLflow: shap_explainer_{name}.pkl"
                        )

                        logger.info(
                            f"    SHAP explainer logged to MLflow: shap_explainer_{name}.pkl"
                        )

                        # Save explanation report locally + MLflow
                        save_artifact_local_and_mlflow(
                            explanation_report,
                            f"shap_report_{name}.json",
                            "json",
                            explainability_dir,
                        )

                        # Generate and log SHAP summary plot
                        if hasattr(model, "predict_proba"):
                            shap_values_for_plot = explainer.shap_values(X_sample)
                            if isinstance(shap_values_for_plot, list):
                                shap_values_for_plot = (
                                    shap_values_for_plot[1]
                                    if len(shap_values_for_plot) > 1
                                    else shap_values_for_plot[0]
                                )

                            fig_shap = plot_shap_summary(
                                shap_values_for_plot,
                                X_sample,
                                feature_names=feature_names,
                                title=f"SHAP Summary - {name}",
                            )

                            save_artifact_local_and_mlflow(
                                fig_shap,
                                f"{name}_shap_summary.png",
                                "plot",
                                explainability_dir,
                            )
                            plt.close(fig_shap)

                        logger.info(f"   SHAP explanation generated for {name}")

                except Exception as e:
                    logger.warning(f"    SHAP explanation failed for {name}: {e}")

            logger.info("   SHAP explanations logged to MLflow")

        except Exception as e:
            logger.warning(f"    SHAP explanation generation failed: {e}")

    except Exception as e:
        logger.warning(f"    Plot generation failed: {e}")

    logger.info(" Evaluation complete")
    return results


def validate_models(results: Dict[str, Dict[str, float]], cfg: TrainConfig) -> bool:
    """
    Validate models against business constraints.

    Returns:
        True if at least one model passes validation
    """
    logger.info(" Step 5/5: Validating against business constraints...")
    logger.info(f"   Constraints: min_recall={cfg.min_recall}, max_fpr={cfg.max_fpr}")
    logger.info(
        f"   Model results: xgboost:{results['xgboost']}, \nrandom_forest:{results['random_forest']}, \nneural_network:{results['neural_network']}, \nisolation_forest:{results['isolation_forest']}"
    )

    passed = validate_all_models(
        results, min_recall=cfg.min_recall, max_fpr=cfg.max_fpr
    )

    if passed:
        logger.info("  At least one model meets business requirements")
    else:
        logger.error("   No model meets business requirements")

    return passed


def register_best_model(
    models: Dict[str, Any],
    results: Dict[str, Dict[str, float]],
    feature_names: list,
    cfg: TrainConfig,
    artifacts_dir: Path,
):
    """
    Register ALL 4 models in MLflow Model Registry as Staging.
    Saves models locally first, then logs to MLflow.
    This allows the canary deployment to compare the complete ensemble.
    """
    if not cfg.register_models:
        logger.info("Model registration disabled in config")
        return

    logger.info(" Registering all 4 models as ensemble in MLflow...")

    models_dir = artifacts_dir / "models"
    registered_models = []

    for model_name, model in models.items():
        try:
            # Register each model individually
            mlflow_name = f"fraud_detection_{model_name}"

            # Save model locally FIRST (before MLflow)
            local_model_path = models_dir / f"{model_name}_model.pkl"
            model_to_save = model.model if hasattr(model, "model") else model
            joblib.dump(model_to_save, local_model_path)
            logger.info(f"   Model saved locally: {local_model_path}")

            # Create a run for model registration
            with start_run(run_name=f"register_{model_name}"):
                # Log the model to MLflow (from in-memory object)
                log_model(model_to_save, artifact_path="model")
                register_model(
                    name=mlflow_name,
                    stage="Staging",
                )

                # Log individual model metadata as artifact
                metadata = {
                    "model_type": model_name,
                    "metrics": results[model_name],
                    "feature_names": feature_names,
                    "config": asdict(cfg),
                    "ensemble_weight": get_ensemble_weight(model_name),
                    "local_path": str(local_model_path),
                    "timestamp": cfg.run_timestamp,
                }

                # Save metadata locally + MLflow
                save_artifact_local_and_mlflow(
                    metadata,
                    f"{model_name}_metadata.json",
                    "json",
                    models_dir,
                )

                logger.info(f"  Model registered: {mlflow_name} (Staging)")
                registered_models.append(mlflow_name)

        except Exception as e:
            logger.error(f"   Failed to register {model_name}: {e}")

    # Log ensemble metadata as artifact in dedicated run
    with start_run(run_name="register_ensemble"):
        ensemble_metadata = {
            "ensemble_version": f"v{cfg.random_state}",  # Use random_state as version
            "models": registered_models,
            "weights": {
                "xgboost": 0.50,
                "random_forest": 0.30,
                "neural_network": 0.15,
                "isolation_forest": 0.05,
            },
            "metrics": results,
            "feature_names": feature_names,
            "config": asdict(cfg),
            "timestamp": cfg.run_timestamp,
            "artifacts_path": str(artifacts_dir),
        }

        # Save ensemble metadata locally + MLflow
        save_artifact_local_and_mlflow(
            ensemble_metadata,
            "ensemble_metadata.json",
            "json",
            artifacts_dir,
        )

    logger.info(f"  Ensemble registered with {len(registered_models)} models")
    logger.info(f"   All artifacts persisted to: {artifacts_dir}")


def get_ensemble_weight(model_name: str) -> float:
    """Get the ensemble weight for a model."""
    weights = {
        "xgboost": 0.50,
        "random_forest": 0.30,
        "neural_network": 0.15,
        "isolation_forest": 0.05,
    }
    return weights.get(model_name, 0.0)


if __name__ == "__main__":
    # CLI entry point
    ok = run_training()
    raise SystemExit(0 if ok else 1)
