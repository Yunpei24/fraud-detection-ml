# training/src/pipelines/training_pipeline.py
from __future__ import annotations

from typing import Dict, Any, Tuple
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from training.src.features.engineer import build_basic_features
from training.src.features.scaling import fit_scaler, transform_with_scaler
from training.src.features.selection import select_kbest

from training.src.evaluation.metrics import (
    compute_classification_metrics,
    compute_auc_metrics,
)
from training.src.evaluation.validation import validate_columns
from training.src.evaluation.explainability import shap_top_features

from training.src.models import (
    build_xgb, build_rf, build_mlp, build_isoforest
)

from training.src.mlflow_utils.tracking import start_mlflow_run, log_run_params, log_run_metrics
from training.src.mlflow_utils.experiment import set_experiment


def _build_model(kind: str, params: Dict[str, Any]):
    k = kind.lower()
    if k == "xgboost":
        return build_xgb(params)
    if k == "random_forest":
        return build_rf(params)
    if k == "mlp":
        return build_mlp(params)
    if k == "isolation_forest":
        return build_isoforest(params)
    raise ValueError(f"Unknown model kind: {kind}")


def run_training(cfg_path: str = "configs/model_xgb.yaml") -> Tuple[Any, Dict[str, float]]:
    cfg = yaml.safe_load(open(cfg_path))

    # experiment setup
    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "file:mlruns")
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    set_experiment(cfg.get("mlflow", {}).get("experiment", "fraud_training"))

    # load processed data
    df = pd.read_parquet("data/processed/train.parquet")
    validate_columns(df, required=["Class"])
    y = df["Class"].astype(int).values
    X = df.drop(columns=["Class"])

    # basic engineering and scaling
    X_eng, feats = build_basic_features(X.copy())
    scaler = fit_scaler(X_eng, kind="standard")
    X_sc = transform_with_scaler(X_eng, scaler)
    X_sel, selected = select_kbest(X_sc, y, k=min(20, X_sc.shape[1]))

    # split - stratify only for supervised
    mtype = cfg["model"]["type"].lower()
    strat = y if mtype != "isolation_forest" else None
    Xtr, Xte, ytr, yte = train_test_split(
        X_sel, y, test_size=cfg["train"]["test_size"], random_state=cfg["train"]["random_state"], stratify=strat
    )

    # optional SMOTE (only supervised)
    if cfg["train"].get("smote", False) and mtype in {"xgboost", "random_forest", "mlp"}:
        Xtr, ytr = SMOTE().fit_resample(Xtr, ytr)

    model = _build_model(cfg["model"]["type"], cfg["model"].get("params", {}))

    with start_mlflow_run(name=f"train_{cfg['model']['type']}"):
        # log config
        flat = {
            "model_type": cfg["model"]["type"],
            **{f"param_{k}": v for k, v in cfg["model"].get("params", {}).items()},
            "smote": bool(cfg["train"].get("smote", False)),
        }
        log_run_params(flat)

        # fit
        if mtype == "isolation_forest":
            model.fit(Xtr)
            proba = model.predict_proba(Xte)[:, 1]
            ypred = model.predict(Xte, threshold=0.0)
        else:
            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:, 1]
            thr = cfg["model"].get("threshold", 0.5)
            ypred = (proba >= thr).astype(int)

        # metrics
        cls = compute_classification_metrics(yte, ypred)
        aucs = compute_auc_metrics(yte, proba)
        metrics = {**cls, **aucs}
        log_run_metrics(metrics)

        # optional explainability for tree-based models
        if mtype in {"xgboost", "random_forest"}:
            try:
                top = shap_top_features(model, Xtr, feature_names=selected, top_n=10)
                # store as artifact text
                import mlflow
                mlflow.log_text("\n".join([f"{k},{v:.6f}" for k, v in top]), "explainability_top_features.csv")
            except Exception as e:
                print(f"SHAP skipped - {e}")

        print("metrics:", {k: round(v, 4) for k, v in metrics.items()})
        return model, metrics
