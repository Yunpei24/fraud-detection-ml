# training/src/evaluation/__init__.py

from .explainability import (create_explanation_report,
                             create_explanation_report_from_model,
                             generate_shap_values, plot_shap_summary,
                             save_explainer)
from .metrics import (calculate_all_metrics, calculate_auc_roc,
                      calculate_f2_score, calculate_precision_recall,
                      confusion_matrix_dict)
from .plots import (plot_confusion_matrix_plot, plot_feature_importance,
                    plot_precision_recall_curve_plot, plot_roc_auc, save_plots)
from .validation import (cross_validation, validate_all_models, validate_fpr,
                         validate_recall)

__all__ = [
    # metrics
    "calculate_auc_roc",
    "calculate_precision_recall",
    "calculate_f2_score",
    "confusion_matrix_dict",
    "calculate_all_metrics",
    # validation
    "validate_recall",
    "validate_fpr",
    "cross_validation",
    "validate_all_models",
    # plots
    "plot_roc_auc",
    "plot_confusion_matrix_plot",
    "plot_feature_importance",
    "plot_precision_recall_curve_plot",
    "save_plots",
    # explainability
    "generate_shap_values",
    "plot_shap_summary",
    "save_explainer",
    "create_explanation_report",
    "create_explanation_report_from_model",
]
