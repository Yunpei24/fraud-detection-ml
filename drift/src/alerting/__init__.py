"""
Alerting module initialization.
"""

from .alert_manager import AlertManager
from .rules import (
    HIGH_DATA_DRIFT,
    HIGH_TARGET_DRIFT,
    MODEL_DEGRADATION,
    ANOMALY_DETECTED
)

__all__ = [
    "AlertManager",
    "HIGH_DATA_DRIFT",
    "HIGH_TARGET_DRIFT",
    "MODEL_DEGRADATION",
    "ANOMALY_DETECTED",
]
