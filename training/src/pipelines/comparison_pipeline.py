# training/src/pipelines/comparison_pipeline.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from training.src.models import build_xgb, build_rf, build_mlp, build_isoforest
from training.src.features.engineer import build_basic_features
from training.src.features.scaling import fit_scaler, transform_with_scaler
from training.src.features.selection import select_kbest
from training.src.evaluation.metrics import compute_classification_metrics, compute_auc_metrics


def _supervised(model_key: str) -> bool:
    return model_key in {"xgboost", "random_forest", "mlp"}


def _factory(kind: str, params: Dict[str, Any]):
    if kind == "xgboost":
        return build_xgb(params)
    if kind == "random_forest":
        return build_rf(params)
    if kind == "mlp":
        return build_mlp(params)
    if kind == "isolation_forest":
        return build_isoforest(params)
    raise ValueError(kind)


def run_comparison(configs: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    configs example:
    {
      "xgboost": {"params": {...}, "threshold": 0.35},
      "random_forest": {"params": {...}, "threshold": 0.5},
      "mlp": {"params": {...}, "threshold": 0.5},
      "isolation_forest": {"params": {...}}
    }
    """
    df = pd.read_parquet("data/processed/train.parquet")
    y = df["Class"].astype(int).values
    X = df.drop(columns=["Class"])

    X_eng, feats = build_basic_features(X.copy())
    scaler = fit_scaler(X_eng, kind="standard")
    X_sc = transform_with_scaler(X_eng, scaler)
    X_sel, selected = select_kbest(X_sc, y, k=min(20, X_sc.shape[1]))

    Xtr, Xte, ytr, yte = train_test_split(X_sel, y, test_size=0.2, random_state=42, stratify=y)

    rows: List[Dict[str, Any]] = []

    for kind, spec in configs.items():
        params = spec.get("params", {})
        model = _factory(kind, params)

        Xtr_fit, ytr_fit = Xtr, ytr
        if _supervised(kind) and spec.get("smote", False):
            Xtr_fit, ytr_fit = SMOTE().fit_resample(Xtr, ytr)

        if kind == "isolation_forest":
            model.fit(Xtr_fit)
            proba = model.predict_proba(Xte)[:, 1]
            ypred = model.predict(Xte, threshold=0.0)
        else:
            model.fit(Xtr_fit, ytr_fit)
            proba = model.predict_proba(Xte)[:, 1]
            thr = float(spec.get("threshold", 0.5))
            ypred = (proba >= thr).astype(int)

        cls = compute_classification_metrics(yte, ypred)
        aucs = compute_auc_metrics(yte, proba)

        rows.append({"model": kind, **cls, **aucs})

    report = pd.DataFrame(rows).sort_values(["pr_auc", "recall", "f1"], ascending=False)
    print(report)
    return report
