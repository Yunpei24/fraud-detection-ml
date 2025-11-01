"""
Alert rules for drift detection.

This module defines alert rules and thresholds for different types of drift.
"""

from enum import Enum
from typing import Any, Dict


class AlertType(Enum):
    """Alert type enumeration."""

    DATA_DRIFT = "data_drift"
    TARGET_DRIFT = "target_drift"
    CONCEPT_DRIFT = "concept_drift"
    ANOMALY_DETECTED = "anomaly_detected"
    MODEL_PERFORMANCE = "model_performance"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Alert rule definitions
HIGH_DATA_DRIFT = {
    "name": "HIGH_DATA_DRIFT",
    "type": AlertType.DATA_DRIFT,
    "condition": lambda drift_score: drift_score > 0.5,
    "severity": AlertSeverity.ERROR,
    "message_template": "High data drift detected: PSI score = {drift_score:.3f}",
    "description": "Triggered when average PSI score exceeds 0.5",
    "thresholds": {"critical": 0.7, "high": 0.5, "medium": 0.3, "low": 0.1},
}

HIGH_TARGET_DRIFT = {
    "name": "HIGH_TARGET_DRIFT",
    "type": AlertType.TARGET_DRIFT,
    "condition": lambda relative_change: abs(relative_change) > 0.5,
    "severity": AlertSeverity.ERROR,
    "message_template": "Significant target drift: fraud rate changed by {relative_change:.1%}",
    "description": "Triggered when fraud rate changes by more than 50%",
    "thresholds": {
        "critical": 1.0,  # 100% change
        "high": 0.5,  # 50% change
        "medium": 0.3,  # 30% change
        "low": 0.1,  # 10% change
    },
}

MODEL_DEGRADATION = {
    "name": "MODEL_DEGRADATION",
    "type": AlertType.CONCEPT_DRIFT,
    "condition": lambda recall_drop, fpr_increase: recall_drop > 0.05
    or fpr_increase > 0.01,
    "severity": AlertSeverity.CRITICAL,
    "message_template": "Model performance degraded: recall={recall:.3f}, FPR={fpr:.4f}",
    "description": "Triggered when recall drops >5% or FPR increases >1%",
    "thresholds": {
        "recall_drop": {
            "critical": 0.10,  # 10% drop
            "high": 0.05,  # 5% drop
            "medium": 0.03,  # 3% drop
            "low": 0.01,  # 1% drop
        },
        "fpr_increase": {
            "critical": 0.02,  # 2% increase
            "high": 0.01,  # 1% increase
            "medium": 0.005,  # 0.5% increase
            "low": 0.001,  # 0.1% increase
        },
    },
}

ANOMALY_DETECTED = {
    "name": "ANOMALY_DETECTED",
    "type": AlertType.ANOMALY_DETECTED,
    "condition": lambda anomaly_score: anomaly_score > 3.0,
    "severity": AlertSeverity.WARNING,
    "message_template": "Anomaly detected: score = {anomaly_score:.2f} (Z-score)",
    "description": "Triggered when anomaly score exceeds 3 standard deviations",
    "thresholds": {
        "critical": 5.0,  # 5 sigma
        "high": 4.0,  # 4 sigma
        "medium": 3.0,  # 3 sigma
        "low": 2.0,  # 2 sigma
    },
}

# Composite alert rules
RETRAINING_REQUIRED = {
    "name": "RETRAINING_REQUIRED",
    "type": AlertType.MODEL_PERFORMANCE,
    "condition": lambda data_drift, concept_drift: data_drift and concept_drift,
    "severity": AlertSeverity.CRITICAL,
    "message_template": "Model retraining required: both data drift and performance degradation detected",
    "description": "Triggered when both data drift and concept drift are detected",
    "action": "trigger_retraining_pipeline",
}

# Alert rule registry
ALERT_RULES = {
    "HIGH_DATA_DRIFT": HIGH_DATA_DRIFT,
    "HIGH_TARGET_DRIFT": HIGH_TARGET_DRIFT,
    "MODEL_DEGRADATION": MODEL_DEGRADATION,
    "ANOMALY_DETECTED": ANOMALY_DETECTED,
    "RETRAINING_REQUIRED": RETRAINING_REQUIRED,
}


def evaluate_alert_rule(rule_name: str, **kwargs) -> Dict[str, Any]:
    """
    Evaluate an alert rule with given parameters.

    Args:
        rule_name: Name of the alert rule
        **kwargs: Parameters for rule evaluation

    Returns:
        Dictionary with evaluation results
    """
    if rule_name not in ALERT_RULES:
        raise ValueError(f"Unknown alert rule: {rule_name}")

    rule = ALERT_RULES[rule_name]

    # Evaluate condition
    try:
        triggered = rule["condition"](**kwargs)
    except Exception as e:
        triggered = False
        print(f"Error evaluating rule {rule_name}: {e}")

    # Determine severity based on thresholds
    severity = rule["severity"].value
    if "thresholds" in rule and triggered:
        # Find appropriate severity level
        for level in ["critical", "high", "medium", "low"]:
            threshold = rule["thresholds"].get(level)
            if threshold and any(
                v > threshold for v in kwargs.values() if isinstance(v, (int, float))
            ):
                severity = level.upper()
                break

    # Format message
    message = rule["message_template"].format(**kwargs)

    return {
        "rule_name": rule_name,
        "triggered": triggered,
        "severity": severity,
        "message": message,
        "type": rule["type"].value,
        "parameters": kwargs,
    }


def check_all_rules(drift_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check all alert rules against drift detection results.

    Args:
        drift_results: Results from drift detection

    Returns:
        Dictionary with all triggered alerts
    """
    alerts = {"triggered_rules": [], "total_alerts": 0, "highest_severity": "LOW"}

    # Check data drift
    data_drift = drift_results.get("data_drift", {})
    if data_drift.get("drift_detected"):
        result = evaluate_alert_rule(
            "HIGH_DATA_DRIFT", drift_score=data_drift.get("avg_psi", 0)
        )
        if result["triggered"]:
            alerts["triggered_rules"].append(result)

    # Check target drift
    target_drift = drift_results.get("target_drift", {})
    if target_drift.get("drift_detected"):
        result = evaluate_alert_rule(
            "HIGH_TARGET_DRIFT", relative_change=target_drift.get("relative_change", 0)
        )
        if result["triggered"]:
            alerts["triggered_rules"].append(result)

    # Check concept drift
    concept_drift = drift_results.get("concept_drift", {})
    if concept_drift.get("drift_detected"):
        metrics = concept_drift.get("metrics", {})
        result = evaluate_alert_rule(
            "MODEL_DEGRADATION",
            recall_drop=metrics.get("recall_change", 0),
            fpr_increase=metrics.get("fpr_change", 0),
        )
        if result["triggered"]:
            alerts["triggered_rules"].append(result)

    # Update summary
    alerts["total_alerts"] = len(alerts["triggered_rules"])

    if alerts["triggered_rules"]:
        severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        alerts["highest_severity"] = max(
            (rule["severity"] for rule in alerts["triggered_rules"]),
            key=lambda s: severity_order.get(s, 0),
        )

    return alerts
