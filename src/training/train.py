# src/training/train.py
import os, yaml
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix
)

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import mlflow.sklearn
import mlflow.xgboost
from joblib import dump  # <-- needed for non-XGB models

PROCESSED_PATH = "data/processed/train.parquet"

def load_xy(path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    df = pd.read_parquet(path)
    y = df["Class"].astype(int).values
    X = df.drop(columns=["Class"])
    return X.values, y, list(X.columns)

def build_model(cfg):
    mtype = cfg["model"]["type"].lower()
    params = cfg["model"].get("params", {})

    if mtype == "xgboost":
        return XGBClassifier(**params, eval_metric="logloss")
    if mtype == "random_forest":
        return RandomForestClassifier(**params)
    if mtype == "mlp":
        return MLPClassifier(**params)
    if mtype == "isolation_forest":
        return IsolationForest(**params)

    raise ValueError(f"Unknown model type: {mtype}")

def predict_proba_or_score(model, mtype, X):
    if mtype == "isolation_forest":
        return model.decision_function(X)
    else:
        return model.predict_proba(X)[:, 1]

def binarize_predictions(mtype, scores, threshold, use_quantile=False, q=0.02):
    if mtype == "isolation_forest":
        if use_quantile:
            cut = np.quantile(scores, q)
            return (scores <= cut).astype(int)
        else:
            return (scores < 0).astype(int)
    else:
        return (scores >= threshold).astype(int)

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mtype = cfg["model"]["type"].lower()
    model_name = cfg["model"]["name"]

    X, y, feats = load_xy(PROCESSED_PATH)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=cfg["train"]["test_size"],
        random_state=cfg["train"]["random_state"],
        stratify=y if mtype != "isolation_forest" else None
    )

    if cfg["train"].get("smote", False) and mtype in {"xgboost","random_forest","mlp"}:
        Xtr, ytr = SMOTE().fit_resample(Xtr, ytr)

    model = build_model(cfg)

    with mlflow.start_run() as run:
        # Fit
        if mtype == "isolation_forest":
            model.fit(Xtr)
        else:
            model.fit(Xtr, ytr)

        # Scores
        scores = predict_proba_or_score(model, mtype, Xte)

        # Thresholding
        if mtype == "isolation_forest":
            ap = cfg["model"].get("anomaly_policy", {})
            use_q = ap.get("use_quantile", False)
            q = ap.get("quantile", 0.02)
            ypred = binarize_predictions(mtype, scores, threshold=None, use_quantile=use_q, q=q)
            ranks = pd.Series(scores).rank(method="average") / len(scores)
            proba_for_auc = 1.0 - ranks.values
        else:
            thr = cfg["model"]["threshold"]
            ypred = binarize_predictions(mtype, scores, threshold=thr)
            proba_for_auc = scores

        # Metrics
        try:
            auc = roc_auc_score(yte, proba_for_auc)
        except Exception:
            auc = float("nan")
        try:
            pr_auc = average_precision_score(yte, proba_for_auc)
        except Exception:
            pr_auc = float("nan")

        prec, rec, f1, _ = precision_recall_fscore_support(
            yte, ypred, average="binary", zero_division=0
        )
        cm = confusion_matrix(yte, ypred)

        # Log params/metrics/artifacts to MLflow
        mlflow.log_param("model_type", mtype)
        for k, v in cfg["model"].get("params", {}).items():
            mlflow.log_param(k, v)
        if mtype != "isolation_forest":
            mlflow.log_param("threshold", cfg["model"]["threshold"])

        mlflow.log_metric("auc", float(auc) if np.isfinite(auc) else -1.0)
        mlflow.log_metric("pr_auc", float(pr_auc) if np.isfinite(pr_auc) else -1.0)
        mlflow.log_metric("precision", float(prec))
        mlflow.log_metric("recall", float(rec))
        mlflow.log_metric("f1", float(f1))

        mlflow.log_text("\n".join(feats), "features.txt")
        mlflow.log_text("\n".join([",".join(map(str, r)) for r in cm]), "confusion_matrix.csv")

        if mtype == "xgboost":
            mlflow.xgboost.log_model(
                model, "model",
                registered_model_name=cfg["mlflow"]["register_name"] if cfg["mlflow"].get("register", False) else None
            )
        else:
            mlflow.sklearn.log_model(
                model, "model",
                registered_model_name=cfg["mlflow"]["register_name"] if cfg["mlflow"].get("register", False) else None
            )

        # --- NEW: persist a local artifact for DVC tracking ---
        out_dir = Path("models") / mtype
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "features.txt").write_text("\n".join(feats))

        if mtype == "xgboost":
            model.save_model(out_dir / "model.json")
        else:
            dump(model, out_dir / "model.joblib")

        print(f"[{model_name}] run_id={run.info.run_id} AUC={auc:.4f} PR_AUC={pr_auc:.4f} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    main(a.config)
