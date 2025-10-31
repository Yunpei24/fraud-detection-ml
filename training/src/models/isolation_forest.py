# training/src/models/isolation_forest.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from joblib import dump, load
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestModel:
    params: Dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_samples": "auto",
        "contamination": "auto",  # or a float like 0.002 if you want to fix it
        "random_state": 42,
        "n_jobs": -1,
    })
    # For unsupervised scoring we will either
    # - use sign of decision_function (default)
    # - or a quantile cut if you want a fixed fraud rate externally
    use_quantile: bool = False
    quantile: float = 0.02  # lowest scores are anomalies
    _clf: Optional[IsolationForest] = None

    def train(self, X_train: np.ndarray) -> "IsolationForestModel":
        clf = IsolationForest(**self.params)
        clf.fit(X_train)
        self._clf = clf
        return self

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Higher scores mean more normal. Negative scores are more anomalous.
        """
        self._ensure_fitted()
        return self._clf.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns 1 for fraud and 0 for normal to match supervised label convention.
        Default policy: score < 0 is fraud.
        If use_quantile is True, mark the lowest quantile as fraud.
        """
        scores = self.decision_scores(X)
        if self.use_quantile:
            cut = np.quantile(scores, self.quantile)
            return (scores <= cut).astype(int)
        return (scores < 0.0).astype(int)

    def save_model(self, path: str) -> None:
        self._ensure_fitted()
        dump(
            {
                "model": self._clf,
                "use_quantile": self.use_quantile,
                "quantile": self.quantile,
            },
            path,
        )

    @staticmethod
    def load_model(path: str) -> "IsolationForestModel":
        bundle = load(path)
        obj = IsolationForestModel()
        obj._clf = bundle["model"]
        obj.use_quantile = bundle.get("use_quantile", obj.use_quantile)
        obj.quantile = bundle.get("quantile", obj.quantile)
        return obj

    def _ensure_fitted(self) -> None:
        if self._clf is None:
            raise RuntimeError("IsolationForestModel is not trained yet")
