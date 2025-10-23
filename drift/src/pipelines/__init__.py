"""
Pipelines module initialization.
"""

from .hourly_monitoring import (
    fetch_recent_predictions,
    compute_all_drifts,
    check_thresholds,
    trigger_alerts,
    update_dashboards,
    run_hourly_monitoring
)
from .daily_analysis import (
    aggregate_daily_metrics,
    generate_daily_report,
    identify_trends,
    recommend_actions,
    run_daily_analysis
)

__all__ = [
    "fetch_recent_predictions",
    "compute_all_drifts",
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
