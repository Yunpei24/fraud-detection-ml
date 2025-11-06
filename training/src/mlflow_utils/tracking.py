# training/src/mlflow_utils/tracking.py
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

try:
    import mlflow.sklearn  # type: ignore

    import mlflow  # type: ignore

    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False

# xgboost is optional
try:
    import mlflow.xgboost  # type: ignore
    import xgboost as xgb  # type: ignore

    _HAS_XGB = True
except Exception:
    xgb = None  # type: ignore
    _HAS_XGB = False

logger = logging.getLogger(__name__)


def log_params(params: Dict[str, Any]) -> None:
    if not (MLFLOW_AVAILABLE and mlflow.active_run()):
        return
    try:
        mlflow.log_params({k: _stringify(v) for k, v in params.items()})
    except Exception:
        pass


def log_metrics(metrics: Dict[str, float]) -> None:
    if not (MLFLOW_AVAILABLE and mlflow.active_run()):
        return
    try:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if _is_number(v)})
    except Exception:
        pass


def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    if not (MLFLOW_AVAILABLE and mlflow.active_run()):
        logger.warning(
            "MLflow not available or no active run - skipping artifact logging"
        )
        return
    try:
        logger.debug(f"Logging artifact: {local_path} → {artifact_path}")
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logger.debug(f" Artifact logged: {local_path}")
    except Exception as e:
        logger.error(f" Failed to log artifact {local_path}: {e}")
        raise


def log_model(
    model: Any, model_type: str | None = None, artifact_path: str = "model"
) -> None:
    """
    Logs the underlying estimator to MLflow if available. Works with:
      - XGBClassifier (uses mlflow.xgboost when available)
      - scikit-learn classifiers (RandomForestClassifier, MLPClassifier)
      - Anomaly detectors (IsolationForest) - logged as generic sklearn model
      - Your wrappers that expose `.model` as the underlying estimator
    """
    if not (MLFLOW_AVAILABLE and mlflow.active_run()):
        logger.warning("MLflow not available or no active run - skipping model logging")
        return

    est = _unwrap_model(model)
    model_class = est.__class__.__name__

    try:
        if _HAS_XGB and xgb is not None and isinstance(est, xgb.XGBClassifier):
            # Case 1: XGBoost classifier
            logger.info(f"Logging XGBoost model to artifact_path='{artifact_path}'")
            mlflow.xgboost.log_model(est, artifact_path=artifact_path)
            logger.info(f" XGBoost model logged successfully")
        else:
            # Case 2: All sklearn models (classifiers + anomaly detectors)
            # This includes: RandomForestClassifier, MLPClassifier, IsolationForest
            logger.info(
                f"Logging sklearn model ({model_class}) to artifact_path='{artifact_path}'"
            )
            mlflow.sklearn.log_model(est, artifact_path=artifact_path)
            logger.info(f" Sklearn model ({model_class}) logged successfully")
    except Exception as e:
        logger.error(f" Failed to log model with mlflow: {e}")
        logger.error(f"   Model type: {model_class}")
        logger.error(
            f"   Model attributes: {dir(est)[:10]}..."
        )  # Show first 10 attributes

        # Last resort: dump to temp dir and log as artifacts
        try:
            logger.info("Attempting fallback: logging model as pickle artifact...")
            with tempfile.TemporaryDirectory() as td:
                dump_dir = Path(td) / "model_dump"
                dump_dir.mkdir(parents=True, exist_ok=True)
                # Try joblib if available
                try:
                    from joblib import dump  # type: ignore

                    dump(est, dump_dir / "model.joblib")
                    logger.info(f"Model dumped to joblib format")
                except Exception:
                    # Fall back to pickle
                    import pickle

                    with open(dump_dir / "model.pkl", "wb") as f:
                        pickle.dump(est, f)
                    logger.info(f"Model dumped to pickle format")

                mlflow.log_artifacts(str(dump_dir), artifact_path=artifact_path)
                logger.info(f" Model artifacts logged successfully (fallback)")
        except Exception as e2:
            logger.error(f" Fallback also failed: {e2}")
            raise RuntimeError(f"Failed to log model: {e}") from e2


# -------- helpers --------
def _unwrap_model(m: Any) -> Any:
    """
    Unwraps our simple wrappers that store underlying estimator as `.model`.
    """
    if hasattr(m, "get_native_model"):
        try:
            return m.get_native_model()
        except Exception:
            pass
    if hasattr(m, "model"):
        try:
            return m.model
        except Exception:
            pass
    return m


def _stringify(v: Any) -> str:
    return str(v)


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False
