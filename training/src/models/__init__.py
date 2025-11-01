# training/src/models/__init__.py
"""
Model classes for fraud detection.
All models follow a consistent interface: fit(), predict(), predict_proba().
"""
from __future__ import annotations

__all__ = [
    "XGBoostModel",
    "RandomForestModel",
    "NeuralNetworkModel",
    "IsolationForestModel",
]
