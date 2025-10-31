# training/src/models/neural_network.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
from joblib import dump, load
from sklearn.neural_network import MLPClassifier


@dataclass
class NeuralNetworkModel:
    params: Dict = field(default_factory=lambda: {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": 256,
        "learning_rate": "adaptive",
        "max_iter": 50,
        "random_state": 42,
        "early_stopping": True,
        "n_iter_no_change": 10,
        "validation_fraction": 0.1,
    })
    threshold: float = 0.50
    _clf: Optional[MLPClassifier] = None

    def build_model(self, input_dim: int) -> "NeuralNetworkModel":
        # kept for API symmetry, MLPClassifier does not need input_dim at init
        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
    ) -> "NeuralNetworkModel":
        if epochs is not None:
            self.params["max_iter"] = int(epochs)

        clf = MLPClassifier(**self.params)
        # Early stopping inside sklearn uses internal split if early_stopping=True
        clf.fit(X_train, y_train)
        self._clf = clf
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        proba = self._clf.predict_proba(X)[:, 1]
        return proba

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        thr = self.threshold if threshold is None else float(threshold)
        proba = self.predict_proba(X)
        return (proba >= thr).astype(int)

    def save_model(self, path: str) -> None:
        self._ensure_fitted()
        dump({"model": self._clf, "threshold": self.threshold}, path)

    @staticmethod
    def load_model(path: str) -> "NeuralNetworkModel":
        bundle = load(path)
        obj = NeuralNetworkModel()
        obj._clf = bundle["model"]
        obj.threshold = bundle.get("threshold", obj.threshold)
        return obj

    def _ensure_fitted(self) -> None:
        if self._clf is None:
            raise RuntimeError("NeuralNetworkModel is not trained yet")
