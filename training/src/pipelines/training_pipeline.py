# training/src/pipelines/training_pipeline.py
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# data & features
from training.src.data.loader import load_local_creditcard
from training.src.data.preprocessor import DataPreprocessor
from training.src.data.splitter import stratified_split
from training.src.features.engineer import build_feature_frame
from training.src.features.scaling import StandardScalerWrapper, MinMaxScalerWrapper
from training.src.features.selection import select_k_best_mutual_info

# models
from training.src.models.xgboost_model import XGBoostModel
from training.src.models.neural_network import NeuralNetworkModel
from training.src.models.isolation_forest import IsolationForestModel
from training.src.models.random_forest import RandomForestModel

# evaluation
from training.src.evaluation.metrics import calculate_all_metrics
from training.src.evaluation.validation import validate_all_models
from training.src.evaluation.plots import (
    plot_roc_auc,
    plot_precision_recall_curve_plot,
    plot_confusion_matrix_plot,
    plot_feature_importance,
    save_plots,
)

# mlflow utils (optional)
from training.src.mlflow_utils.experiment import setup_experiment, start_run, end_run
from training.src.mlflow_utils.tracking import log_params, log_metrics, log_model, log_artifact
from training.src.mlflow_utils.registry import register_model

ARTIFACTS_DIR = Path("training/artifacts")
MODEL_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
EXPLAIN_DIR = ARTIFACTS_DIR / "explain"


@dataclass
class TrainConfig:
    test_size: float = 0.2
    val_size: float = 0.1          # from train portion
    random_state: int = 42
    use_smote: bool = True         # applied inside models that support it
    feature_k: int = 25            # top features via MI (optional)
    scale_for_trees: bool = False  # trees usually don’t need scaling
    scale_for_nn: bool = True

    # business constraints
    min_recall: float = 0.95
    max_fpr: float = 0.02

    # MLflow
    experiment_name: str = "fraud_training"
    register_models: bool = True


# -------------------------
# Public entry point
# -------------------------
def run_training(cfg: TrainConfig | None = None) -> bool:
    cfg = cfg or TrainConfig()
    _ensure_dirs()

    # MLflow experiment setup (no-op if MLflow not configured)
    setup_experiment(cfg.experiment_name)

    X, y, feature_names = load_data()
    Xtr, Xval, Xte, ytr, yval, yte = preprocess_data(X, y, feature_names, cfg)

    # train & evaluate in parallel
    models = train_models(Xtr, ytr, Xval, yval, cfg)
    results = evaluate_models(models, Xte, yte)

    # validate against business rules
    ok = validate_models(results, cfg)
    if not ok:
        print("[training] Validation failed. See metrics below.")
    else:
        print("[training] Validation passed.")

    # register & persist
    register_models(models, results, cfg)
    save_artifacts(models, results)

    return ok


# -------------------------
# Steps
# -------------------------
def load_data() -> Tuple[np.ndarray, np.ndarray, list]:
    df = load_local_creditcard("data/raw/creditcard.csv")
    # feature build (idempotent; safe if columns missing)
    df = build_feature_frame(df, ts_col=None, country_col=None)

    y = df["Class"].astype(int).values
    X = df.drop(columns=["Class"]).values
    feats = [c for c in df.columns if c != "Class"]
    return X, y, feats


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    cfg: TrainConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # basic cleaning (no-ops for creditcard.csv but kept for extensibility)
    pre = DataPreprocessor()
    X_df = pre.fix_data_types(pre.handle_missing_values(pre.to_frame(X, feature_names)))
    pre.check_data_quality(X_df)

    # optional feature selection (top-K mutual information)
    if cfg.feature_k and cfg.feature_k < X_df.shape[1]:
        X_df, _ = select_k_best_mutual_info(X_df, target=y, k=cfg.feature_k)

    # split (stratified)
    Xtr, Xval, Xte, ytr, yval, yte = stratified_split(
        X_df.values, y, test_size=cfg.test_size, val_size=cfg.val_size, random_state=cfg.random_state
    )

    # scaling: trees (off by default), NN (on)
    if cfg.scale_for_trees:
        tree_scaler = StandardScalerWrapper().fit(Xtr)
        Xtr = tree_scaler.transform(Xtr)
        Xval = tree_scaler.transform(Xval)
        Xte = tree_scaler.transform(Xte)
    if cfg.scale_for_nn:
        nn_scaler = MinMaxScalerWrapper().fit(Xtr)
        Xtr = nn_scaler.transform(Xtr)
        Xval = nn_scaler.transform(Xval)
        Xte = nn_scaler.transform(Xte)

    return Xtr, Xval, Xte, ytr, yval, yte


def _train_one(name: str, Xtr, ytr, Xval, yval, cfg: TrainConfig):
    if name == "xgboost":
        model = XGBoostModel(use_smote=cfg.use_smote, random_state=cfg.random_state)
    elif name == "random_forest":
        model = RandomForestModel(use_smote=cfg.use_smote, random_state=cfg.random_state)
    elif name == "nn":
        model = NeuralNetworkModel(random_state=cfg.random_state)
    elif name == "isolation_forest":
        model = IsolationForestModel(random_state=cfg.random_state)
    else:
        raise ValueError(f"Unknown model key: {name}")

    run_name = f"train_{name}"
    with start_run(run_name):
        model.train(Xtr, ytr, Xval, yval) if name != "isolation_forest" else model.train(Xtr)
        log_params({"model": name, **model.get_params()})
        return name, model


def train_models(Xtr, ytr, Xval, yval, cfg: TrainConfig) -> Dict[str, object]:
    names = ["xgboost", "random_forest", "nn", "isolation_forest"]

    models: Dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=len(names)) as ex:
        futures = [ex.submit(_train_one, n, Xtr, ytr, Xval, yval, cfg) for n in names]
        for f in as_completed(futures):
            name, model = f.result()
            models[name] = model
            print(f"[training] finished {name}")
    return models


def evaluate_models(models: Dict[str, object], Xte, yte) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        # probabilities / scores
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xte)
        elif hasattr(model, "decision_function"):
            # anomaly score; larger is more normal → map to “anomaly probability”
            scores = model.decision_function(Xte)
            ranks = _to_rank_prob(scores)  # 0..1 where higher means more anomalous
            proba = ranks
        else:
            # crude fallback
            proba = model.predict(Xte)

        # threshold (each wrapper provides default threshold; IF uses 0 or quantile internally)
        yhat = model.predict(Xte)

        m = calculate_all_metrics(yte, np.asarray(proba), np.asarray(yhat))
        results[name] = m
        print(f"[eval] {name}: AUC={m['auc']:.4f} PR_AUC={m['pr_auc']:.4f} R={m['recall']:.3f} FPR={m['fpr']:.3f}")

        # log to MLflow
        log_metrics({f"{name}_{k}": v for k, v in m.items()})
        # plots
        try:
            figs = []
            figs.append(plot_roc_auc(yte, proba, title=f"ROC AUC - {name}"))
            figs.append(plot_precision_recall_curve_plot(yte, proba, title=f"PR Curve - {name}"))
            figs.append(plot_confusion_matrix_plot(yte, yhat, title=f"Confusion - {name}"))
            if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
                figs.append(plot_feature_importance(model.model if hasattr(model, "model") else model, [f"f{i}" for i in range(Xte.shape[1])], title=f"Feature Importance - {name}"))
            save_plots(str(PLOTS_DIR / name), *figs)
            for i in range(len(figs)):
                log_artifact(str(PLOTS_DIR / name / f"plot_{i+1:02d}.png"), artifact_path=f"plots/{name}")
        except Exception as e:
            print(f"[eval] plot logging failed for {name}: {e}")

    return results


def validate_models(results: Dict[str, Dict[str, float]], cfg: TrainConfig) -> bool:
    return validate_all_models(results, min_recall=cfg.min_recall, max_fpr=cfg.max_fpr)


def register_models(models: Dict[str, object], results: Dict[str, Dict[str, float]], cfg: TrainConfig):
    if not cfg.register_models:
        return
    # register the best by PR_AUC (or AUC fallback)
    def score(m):
        r = results[m]
        return (r.get("pr_auc") or 0.0, r.get("auc") or 0.0)

    best_name = max(models.keys(), key=score)
    model = models[best_name]
    try:
        log_model(model, best_name)  # logs under current run
        register_model(model, name=f"fraud_{best_name}", stage="staging")
        print(f"[registry] registered {best_name} → staging")
    except Exception as e:
        print(f"[registry] registration skipped: {e}")


def save_artifacts(models: Dict[str, object], results: Dict[str, Dict[str, float]]):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        out = MODEL_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
    # save metrics json
    import json
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))


# -------------------------
# CLI
# -------------------------
def _ensure_dirs():
    for p in [ARTIFACTS_DIR, MODEL_DIR, PLOTS_DIR, EXPLAIN_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)


def _to_rank_prob(scores: np.ndarray) -> np.ndarray:
    # map arbitrary scores to [0,1] with higher = more anomalous
    import pandas as pd
    ranks = pd.Series(scores).rank(method="average") / len(scores)
    return 1.0 - ranks.values  # lower score ⇒ more anomalous ⇒ higher prob


if __name__ == "__main__":
    # allow: python -m training.src.pipelines.training_pipeline
    ok = run_training()
    raise SystemExit(0 if ok else 1)
