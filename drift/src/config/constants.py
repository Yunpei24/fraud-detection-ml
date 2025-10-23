"""
Constants for drift detection thresholds and configuration.
"""

# Drift Thresholds
DATA_DRIFT_THRESHOLD = 0.3  # PSI threshold for data drift
TARGET_DRIFT_THRESHOLD = 0.5  # Relative change in fraud rate
CONCEPT_DRIFT_THRESHOLD = 0.05  # Maximum allowed recall drop

# Performance Baselines (from validation set)
BASELINE_RECALL = 0.98
BASELINE_PRECISION = 0.95
BASELINE_FPR = 0.015
BASELINE_F1_SCORE = 0.965
BASELINE_FRAUD_RATE = 0.002  # 0.2% fraud rate

# Alert Severity Levels
SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"

# Drift Types
DRIFT_TYPE_DATA = "data_drift"
DRIFT_TYPE_TARGET = "target_drift"
DRIFT_TYPE_CONCEPT = "concept_drift"

# Monitoring Windows (in seconds)
WINDOW_1_HOUR = 3600
WINDOW_6_HOURS = 21600
WINDOW_24_HOURS = 86400
WINDOW_7_DAYS = 604800

# Statistical Test Parameters
KS_TEST_ALPHA = 0.05  # Significance level for KS test
CHI_SQUARE_ALPHA = 0.05  # Significance level for chi-square test

# PSI (Population Stability Index) Thresholds
PSI_NO_CHANGE = 0.1  # < 0.1: No significant change
PSI_MODERATE_CHANGE = 0.25  # 0.1-0.25: Moderate change
PSI_SIGNIFICANT_CHANGE = 0.3  # > 0.25: Significant change requiring action

# Retraining Configuration
MIN_SAMPLES_FOR_DRIFT = 1000  # Minimum samples before checking drift
RETRAINING_COOLDOWN_HOURS = 48  # Wait 48h between retraining attempts
CONSECUTIVE_DRIFT_DETECTIONS = 3  # Require 3 consecutive detections

# Alert Configuration
MAX_ALERTS_PER_HOUR = 5  # Rate limit for alerts
ALERT_DEBOUNCE_MINUTES = 30  # Debounce duplicate alerts

# Feature Importance Threshold
IMPORTANT_FEATURE_THRESHOLD = 0.01  # Monitor top features by importance

# Database Table Names
TABLE_DRIFT_METRICS = "drift_metrics"
TABLE_BASELINE_METRICS = "baseline_metrics"
TABLE_DRIFT_ALERTS = "drift_alerts"
TABLE_RETRAINING_HISTORY = "retraining_history"

# Report Configuration
REPORT_RETENTION_DAYS = 90  # Keep reports for 90 days
REPORT_FORMAT = "html"  # html or pdf
VISUALIZATION_DPI = 300  # High quality plots

# Prometheus Metrics Names
METRIC_DATA_DRIFT_PSI = "drift_data_psi"
METRIC_TARGET_FRAUD_RATE = "drift_target_fraud_rate"
METRIC_CONCEPT_RECALL = "drift_concept_recall"
METRIC_CONCEPT_FPR = "drift_concept_fpr"
METRIC_CONCEPT_F1 = "drift_concept_f1"
METRIC_ALERT_COUNTER = "drift_alert_total"
METRIC_RETRAINING_COUNTER = "drift_retraining_total"

# Version
DRIFT_COMPONENT_VERSION = "1.0.0"
