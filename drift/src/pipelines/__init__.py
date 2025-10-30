"""
Pipelines module initialization.
"""

from .daily_analysis import (aggregate_daily_metrics, generate_daily_report,
                             identify_trends, recommend_actions,
                             run_daily_analysis)
from .hourly_monitoring import (call_api_drift_detection, check_thresholds,
                                run_hourly_monitoring, trigger_alerts,
                                update_dashboards)

__all__ = [
    "call_api_drift_detection",
    "check_thresholds",
    "trigger_alerts",
    "update_dashboards",
    "run_hourly_monitoring",
    "aggregate_daily_metrics",
    "generate_daily_report",
    "identify_trends",
    "recommend_actions",
    "run_daily_analysis",
]
