# training/src/mlflow_utils/tracking.py
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

try:
    import mlflow  # type: ignore
    import mlflow.sklearn  # type: ignore
    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False

# xgboost is optional
try:
    import xgboost as xgb  # type: ignore
    import mlflow.xgboost  # type: ignore
    _HAS_XGB = True
except Exception:
    xgb = None  # type: ignore
    _HAS_XGB = False


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
        return
    try:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
    except Exception:
        pass


def log_model(model: Any, model_type: str | None = None, artifact_path: str = "model") -> None:
    """
    Logs the underlying estimator to MLflow if available. Works with:
      - XGBClassifier (uses mlflow.xgboost when available)
      - scikit-learn estimators (uses mlflow.sklearn)
      - Your wrappers that expose `.model` as the underlying estimator
    """
    if not (MLFLOW_AVAILABLE and mlflow.active_run()):
        return

    est = _unwrap_model(model)

    try:
        if _HAS_XGB and xgb is not None and isinstance(est, xgb.XGBClassifier):
            mlflow.xgboost.log_model(est, artifact_path=artifact_path)
        else:
            mlflow.sklearn.log_model(est, artifact_path=artifact_path)
    except Exception:
        # Last resort: dump to temp dir and log as artifacts
        try:
            with tempfile.TemporaryDirectory() as td:
                dump_dir = Path(td) / "model_dump"
                dump_dir.mkdir(parents=True, exist_ok=True)
                # Try joblib if available
                try:
                    from joblib import dump  # type: ignore
                    dump(est, dump_dir / "model.joblib")
                except Exception:
                    # Fall back to pickle
                    import pickle
                    with open(dump_dir / "model.pkl", "wb") as f:
                        pickle.dump(est, f)
                mlflow.log_artifacts(str(dump_dir), artifact_path=artifact_path)
        except Exception:
            pass


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
