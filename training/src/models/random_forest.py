# training/src/models/random_forest.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier


@dataclass
class RandomForestModel:
    params: Dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": None,  # or "balanced_subsample"
        "random_state": 42,
        "n_jobs": -1,
    })
    threshold: float = 0.50
    _clf: Optional[RandomForestClassifier] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RandomForestModel":
        clf = RandomForestClassifier(**self.params)
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

    def feature_importances(self) -> np.ndarray:
        self._ensure_fitted()
        return getattr(self._clf, "feature_importances_", None)

    def save_model(self, path: str) -> None:
        self._ensure_fitted()
        dump({"model": self._clf, "threshold": self.threshold}, path)

    @staticmethod
    def load_model(path: str) -> "RandomForestModel":
        bundle = load(path)
        obj = RandomForestModel()
        obj._clf = bundle["model"]
        obj.threshold = bundle.get("threshold", obj.threshold)
        return obj

    def _ensure_fitted(self) -> None:
        if self._clf is None:
            raise RuntimeError("RandomForestModel is not trained yet")
