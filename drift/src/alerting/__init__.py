"""
Alerting module initialization.
"""

from .alert_manager import AlertManager
from .rules import (ANOMALY_DETECTED, HIGH_DATA_DRIFT, HIGH_TARGET_DRIFT,
                    MODEL_DEGRADATION)

__all__ = [
    "AlertManager",
    "HIGH_DATA_DRIFT",
    "HIGH_TARGET_DRIFT",
    "MODEL_DEGRADATION",
    "ANOMALY_DETECTED",
]
