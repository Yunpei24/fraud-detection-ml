# training/src/models/xgboost_model.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
from joblib import dump, load
from xgboost import XGBClassifier


@dataclass
class XGBoostModel:
    params: Dict = field(default_factory=lambda: {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    })
    threshold: float = 0.35
    _clf: Optional[XGBClassifier] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
    ) -> "XGBoostModel":
        clf = XGBClassifier(**self.params)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            clf.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            clf.fit(X_train, y_train)

        self._clf = clf
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        return self._clf.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        thr = self.threshold if threshold is None else float(threshold)
        proba = self.predict_proba(X)
        return (proba >= thr).astype(int)

    def save_model(self, path: str) -> None:
        self._ensure_fitted()
        # Save as native XGBoost JSON for portability
        self._clf.save_model(path)

    def save_sklearn_joblib(self, path: str) -> None:
        self._ensure_fitted()
        dump({"model": self._clf, "threshold": self.threshold}, path)

    @staticmethod
    def load_sklearn_joblib(path: str) -> "XGBoostModel":
        bundle = load(path)
        obj = XGBoostModel()
        obj._clf = bundle["model"]
        obj.threshold = bundle.get("threshold", obj.threshold)
        return obj

    def _ensure_fitted(self) -> None:
        if self._clf is None:
            raise RuntimeError("XGBoostModel is not trained yet")
