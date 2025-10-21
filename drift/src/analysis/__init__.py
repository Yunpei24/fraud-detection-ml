"""
Analysis module for drift detection visualization and reporting.
"""

from .visualization import (
    plot_drift_timeline,
    plot_feature_distributions,
    plot_confusion_matrix_timeline,
    save_drift_report
)
from .reporting import (
    generate_drift_report,
    generate_alert_summary,
    export_to_html
)
from .comparison import (
    compare_with_training,
    identify_new_patterns,
    extract_insights
)

__all__ = [
    "plot_drift_timeline",
    "plot_feature_distributions",
    "plot_confusion_matrix_timeline",
    "save_drift_report",
    "generate_drift_report",
    "generate_alert_summary",
    "export_to_html",
    "compare_with_training",
    "identify_new_patterns",
    "extract_insights",
]
