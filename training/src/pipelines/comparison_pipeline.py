# training/src/pipelines/comparison_pipeline.py
"""
Model comparison pipeline for canary deployment strategy.
Compares champion (Production) vs challenger (Staging) models from MLflow Registry.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass

import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from src.evaluation.metrics import calculate_all_metrics
from src.config.logging_config import get_logger
from src.data.loader import load_training_data
from src.data.splitter import stratified_split

logger = get_logger(__name__)


@dataclass
class ComparisonResult:
    """Results from model comparison."""
    champion_metrics: Dict[str, float]
    challenger_metrics: Dict[str, float]
    champion_latency_ms: float
    challenger_latency_ms: float
    decision: str  # "promote_challenger" or "keep_champion"
    reason: str


def load_model_from_registry(
    model_name: str,
    stage: str = "Production"
) -> Tuple[Any, str]:
    """
    Load a model from MLflow Model Registry.
    
    Args:
        model_name: Name of the registered model
        stage: Model stage (Production, Staging, etc.)
    
    Returns:
        Tuple of (model, version)
    """
    client = MlflowClient()
    
    try:
        # Get latest version for the specified stage
        versions = client.get_latest_versions(model_name, stages=[stage])
        
        if not versions:
            raise ValueError(f"No model found in {stage} stage for '{model_name}'")
        
        latest_version = versions[0]
        model_uri = f"models:/{model_name}/{stage}"
        
        logger.info(f"Loading model: {model_name} v{latest_version.version} ({stage})")
        model = mlflow.pyfunc.load_model(model_uri)
        
        return model, latest_version.version
        
    except Exception as e:
        logger.error(f"Failed to load model from registry: {e}")
        raise


def compare_ensembles(
    champion_models: Dict[str, Any],
    challenger_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compare two model ensembles on the same test set.
    
    Args:
        champion_models: Dict of champion model instances
        challenger_models: Dict of challenger model instances
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary with metrics for both ensembles
    """
    logger.info("Comparing champion vs challenger ensembles...")
    
    def _evaluate_ensemble(ensemble_dict: Dict[str, Any], name: str):
        """Evaluate ensemble using weighted voting."""
        logger.info(f"   Evaluating {name} ensemble...")
        
        # Ensemble weights
        weights = {
            "fraud_detection_xgboost": 0.50,
            "fraud_detection_random_forest": 0.30,
            "fraud_detection_neural_network": 0.15,
            "fraud_detection_isolation_forest": 0.05,
        }
        
        # Get predictions from each model
        ensemble_scores = []
        
        for model_name, model in ensemble_dict.items():
            try:
                # Get predictions
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)
                    scores = proba[:, 1] if proba.ndim > 1 else proba
                else:
                    pred = model.predict(X_test)
                    scores = pred.astype(float)
                
                ensemble_scores.append(scores * weights.get(model_name, 0.0))
                
            except Exception as e:
                logger.warning(f"   Failed to get predictions from {model_name}: {e}")
                # Use zero contribution for failed model
                ensemble_scores.append(np.zeros(len(X_test)))
        
        # Weighted ensemble prediction
        final_scores = np.sum(ensemble_scores, axis=0)
        final_predictions = (final_scores > 0.5).astype(int)
        
        # Calculate metrics
        metrics = calculate_all_metrics(y_test, final_scores, final_predictions)
        
        logger.info(f"   {name}: AUC={metrics.get('auc', 0):.4f}, "
                   f"Recall={metrics.get('recall', 0):.4f}, "
                   f"Precision={metrics.get('precision', 0):.4f}, "
                   f"F1={metrics.get('f1', 0):.4f}")
        
        return metrics
    
    champion_metrics = _evaluate_ensemble(champion_models, "Champion")
    challenger_metrics = _evaluate_ensemble(challenger_models, "Challenger")
    
    return {
        "champion": champion_metrics,
        "challenger": challenger_metrics
    }


def measure_ensemble_latency(ensemble_models: Dict[str, Any], X: np.ndarray, n_trials: int = 10) -> float:
    """
    Measure average prediction latency for an ensemble.
    
    Args:
        ensemble_models: Dict of model instances
        X: Test data (uses first 100 samples)
        n_trials: Number of trials to average
    
    Returns:
        Average latency in milliseconds
    """
    # Use a small subset for latency measurement
    X_sample = X[:100] if len(X) > 100 else X
    
    latencies = []
    for _ in range(n_trials):
        start = time.perf_counter()
        
        # Get predictions from all models in ensemble
        ensemble_scores = []
        weights = {
            "fraud_detection_xgboost": 0.50,
            "fraud_detection_random_forest": 0.30,
            "fraud_detection_neural_network": 0.15,
            "fraud_detection_isolation_forest": 0.05,
        }
        
        for model_name, model in ensemble_models.items():
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_sample)
                    scores = proba[:, 1] if proba.ndim > 1 else proba
                else:
                    pred = model.predict(X_sample)
                    scores = pred.astype(float)
                
                ensemble_scores.append(scores * weights.get(model_name, 0.0))
            except Exception:
                ensemble_scores.append(np.zeros(len(X_sample)))
        
        # Final ensemble prediction
        final_scores = np.sum(ensemble_scores, axis=0)
        
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    avg_latency = np.mean(latencies)
    logger.info(f"   Ensemble average latency: {avg_latency:.2f}ms (n={n_trials})")
    
    return float(avg_latency)


def compare_models(
    champion: Any,
    challenger: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compare two models on the same test set.
    
    Args:
        champion: Current production model
        challenger: New candidate model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary with metrics for both models
    """
    logger.info("Comparing champion vs challenger models...")
    
    def _evaluate(model, name: str):
        logger.info(f"   Evaluating {name}...")
        
        # Get predictions
        try:
            # Try predict_proba first
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                y_score = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            else:
                # Fallback to predict
                y_pred = model.predict(X_test)
                y_score = y_pred.astype(float)
        except Exception as e:
            logger.warning(f"   Failed to get probabilities for {name}: {e}")
            y_pred = model.predict(X_test)
            y_score = y_pred.astype(float)
        
        y_pred = (y_score > 0.5).astype(int)
        
        # Calculate metrics
        metrics = calculate_all_metrics(y_test, y_score, y_pred)
        
        logger.info(f"   {name}: AUC={metrics.get('auc', 0):.4f}, "
                   f"Recall={metrics.get('recall', 0):.4f}, "
                   f"Precision={metrics.get('precision', 0):.4f}, "
                   f"F1={metrics.get('f1', 0):.4f}")
        
        return metrics
    
    champion_metrics = _evaluate(champion, "Champion")
    challenger_metrics = _evaluate(challenger, "Challenger")
    
    return {
        "champion": champion_metrics,
        "challenger": challenger_metrics
    }


def statistical_test(
    champion_metrics: Dict[str, float],
    challenger_metrics: Dict[str, float],
    *,
    min_improvement_pct: float = 1.0,  # 1% minimum improvement
    primary_metric: str = "f1",
) -> bool:
    """
    Determine if challenger is significantly better than champion.
    
    Args:
        champion_metrics: Champion model metrics
        challenger_metrics: Challenger model metrics
        min_improvement_pct: Minimum percentage improvement required
        primary_metric: Primary metric to compare (default: F1)
    
    Returns:
        True if challenger is significantly better
    """
    champ_score = champion_metrics.get(primary_metric, 0.0)
    chal_score = challenger_metrics.get(primary_metric, 0.0)
    
    if champ_score == 0:
        return chal_score > 0
    
    improvement_pct = 100 * (chal_score - champ_score) / champ_score
    
    logger.info(f"Statistical test: {primary_metric} improvement = {improvement_pct:.2f}%")
    
    return improvement_pct >= min_improvement_pct


def decide_deployment(
    comparison_results: Dict[str, Dict[str, float]],
    champion_latency: float,
    challenger_latency: float,
    *,
    min_recall: float = 0.95,
    max_fpr: float = 0.02,
    max_latency_ms: float = 100.0,
    min_improvement_pct: float = 1.0,
) -> ComparisonResult:
    """
    Make deployment decision based on metrics and business constraints.
    
    Args:
        comparison_results: Metrics for champion and challenger
        champion_latency: Champion model latency (ms)
        challenger_latency: Challenger model latency (ms)
        min_recall: Minimum required recall
        max_fpr: Maximum acceptable false positive rate
        max_latency_ms: Maximum acceptable latency (ms)
        min_improvement_pct: Minimum improvement required for promotion
    
    Returns:
        ComparisonResult with decision and reasoning
    """
    champ = comparison_results["champion"]
    chal = comparison_results["challenger"]
    
    logger.info("=" * 80)
    logger.info("DEPLOYMENT DECISION")
    logger.info("=" * 80)
    
    # Check business constraints for challenger
    chal_recall = chal.get("recall", 0.0)
    chal_fpr = chal.get("fpr", 1.0)
    
    # Business constraint checks
    meets_recall = chal_recall >= min_recall
    meets_fpr = chal_fpr <= max_fpr
    meets_latency = challenger_latency <= max_latency_ms
    
    logger.info(f"Challenger business constraints:")
    logger.info(f"   Recall: {chal_recall:.4f} ({'✅' if meets_recall else '❌'} >= {min_recall})")
    logger.info(f"   FPR: {chal_fpr:.4f} ({'✅' if meets_fpr else '❌'} <= {max_fpr})")
    logger.info(f"   Latency: {challenger_latency:.2f}ms ({'✅' if meets_latency else '❌'} <= {max_latency_ms}ms)")
    
    # If challenger doesn't meet constraints, keep champion
    if not (meets_recall and meets_fpr and meets_latency):
        reason = "Challenger failed business constraints"
        logger.info(f" Decision: KEEP CHAMPION ({reason})")
        return ComparisonResult(
            champion_metrics=champ,
            challenger_metrics=chal,
            champion_latency_ms=champion_latency,
            challenger_latency_ms=challenger_latency,
            decision="keep_champion",
            reason=reason
        )
    
    # Check if challenger is significantly better
    is_better = statistical_test(champ, chal, min_improvement_pct=min_improvement_pct)
    
    if is_better:
        reason = f"Challenger improved F1 by >={min_improvement_pct}% and meets constraints"
        logger.info(f" Decision: PROMOTE CHALLENGER ({reason})")
        decision = "promote_challenger"
    else:
        reason = f"Challenger improvement <{min_improvement_pct}% or not significant"
        logger.info(f" Decision: KEEP CHAMPION ({reason})")
        decision = "keep_champion"
    
    logger.info("=" * 80)
    
    return ComparisonResult(
        champion_metrics=champ,
        challenger_metrics=chal,
        champion_latency_ms=champion_latency,
        challenger_latency_ms=challenger_latency,
        decision=decision,
        reason=reason
    )


def run_comparison(
    model_names: List[str] = ["fraud_detection_xgboost", "fraud_detection_random_forest", "fraud_detection_neural_network", "fraud_detection_isolation_forest"],
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    **kwargs
) -> ComparisonResult:
    """
    Main entry point for ensemble model comparison pipeline.
    
    Args:
        model_names: List of model names in MLflow Registry to compare
        X_test: Test features (if None, will load from data)
        y_test: Test labels (if None, will load from data)
        **kwargs: Additional arguments for decide_deployment
    
    Returns:
        ComparisonResult with ensemble deployment decision
    """
    logger.info("=" * 80)
    logger.info("ENSEMBLE MODEL COMPARISON PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Comparing {len(model_names)} models: {model_names}")
    
    # Load champion ensemble (Production models)
    champion_models = {}
    champion_missing = False
    for model_name in model_names:
        try:
            model, version = load_model_from_registry(model_name, stage="Production")
            champion_models[model_name] = model
            logger.info(f" Champion {model_name} loaded: v{version} (Production)")
        except Exception as e:
            logger.warning(f"  No champion {model_name} found in Production: {e}")
            champion_missing = True
    
    # If no champion models exist (first deployment), automatically promote challenger
    if champion_missing or not champion_models:
        logger.info("🎯 No champion models found - this appears to be first deployment")
        logger.info(" Automatically promoting challenger ensemble for first deployment")
        return ComparisonResult(
            champion_metrics={},  # No champion metrics
            challenger_metrics={},  # Will be evaluated below if needed
            champion_latency_ms=0.0,
            challenger_latency_ms=0.0,
            decision="promote_challenger",
            reason="First deployment - no champion models exist"
        )
    
    # Load challenger ensemble (Staging models)
    challenger_models = {}
    for model_name in model_names:
        try:
            model, version = load_model_from_registry(model_name, stage="Staging")
            challenger_models[model_name] = model
            logger.info(f" Challenger {model_name} loaded: v{version} (Staging)")
        except Exception as e:
            logger.error(f"Failed to load challenger {model_name}: {e}")
            raise
    
    # Load test data if not provided
    if X_test is None or y_test is None:
        logger.info("Loading test data...")
        
        df = load_training_data()
        y = df["class"].values
        X = df.drop(columns=["class"]).values
        
        _, _, X_test, _, _, y_test = stratified_split(X, y, test_size=0.2, val_size=0.0)
        logger.info(f" Test data loaded: {X_test.shape}")
    
    # Compare ensemble metrics
    comparison_results = compare_ensembles(champion_models, challenger_models, X_test, y_test)
    
    # Measure ensemble latency
    logger.info("Measuring ensemble prediction latency...")
    champion_latency = measure_ensemble_latency(champion_models, X_test)
    challenger_latency = measure_ensemble_latency(challenger_models, X_test)
    
    # Make ensemble deployment decision
    result = decide_deployment(
        comparison_results,
        champion_latency,
        challenger_latency,
        **kwargs
    )
    
    logger.info("=" * 80)
    logger.info(f"ENSEMBLE FINAL DECISION: {result.decision.upper()}")
    logger.info(f"REASON: {result.reason}")
    logger.info("=" * 80)
    
    return result


if __name__ == "__main__":
    # CLI entry point
    result = run_comparison()
    
    # Exit with code based on decision
    if result.decision == "promote_challenger":
        raise SystemExit(0)  # Success - promote
    else:
        raise SystemExit(1)  # Keep champion - no deployment
