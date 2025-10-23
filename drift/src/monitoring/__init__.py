"""
Monitoring module initialization.
"""

from .metrics import (
    drift_score_gauge,
    fraud_rate_gauge,
    model_recall_gauge,
    alert_counter,
    setup_prometheus_metrics
)
from .health import check_monitoring_status

__all__ = [
    "drift_score_gauge",
    "fraud_rate_gauge",
    "model_recall_gauge",
    "alert_counter",
    "setup_prometheus_metrics",
    "check_monitoring_status",
]
