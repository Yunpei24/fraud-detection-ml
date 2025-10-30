"""
Fraud Detection Airflow Custom Plugins
=======================================
This module provides custom hooks and operators for the fraud detection system.

Hooks:
- FraudPostgresHook: Custom PostgreSQL hook with convenience methods

Operators:
- FraudDetectionAlertOperator: Alert operator for sending notifications
"""
from airflow.plugins_manager import AirflowPlugin

from plugins.hooks.postgres_hook import FraudPostgresHook
from plugins.operators.alert_operator import FraudDetectionAlertOperator


class FraudDetectionPlugin(AirflowPlugin):
    """Fraud Detection custom plugin for Airflow"""
    name = "fraud_detection"
    hooks = [FraudPostgresHook]
    operators = [FraudDetectionAlertOperator]


__all__ = [
    "FraudDetectionPlugin",
    "FraudPostgresHook",
    "FraudDetectionAlertOperator",
]
