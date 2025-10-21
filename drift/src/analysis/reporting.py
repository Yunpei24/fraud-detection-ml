"""
Reporting module for drift detection.

This module generates comprehensive reports and summaries for drift detection results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import structlog

logger = structlog.get_logger(__name__)


def generate_drift_report(
    drift_results: Dict[str, Any],
    time_period: str = "last_24h"
) -> str:
    """
    Generate a comprehensive drift detection report.
    
    Args:
        drift_results: Dictionary containing all drift detection results
        time_period: Time period for the report (e.g., "last_24h", "last_7d")
        
    Returns:
        Formatted report as string
    """
    logger.info("generating_drift_report", time_period=time_period)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"DRIFT DETECTION REPORT - {time_period.upper()}")
    report_lines.append(f"Generated: {datetime.utcnow().isoformat()}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 80)
    
    data_drift = drift_results.get("data_drift", {})
    target_drift = drift_results.get("target_drift", {})
    concept_drift = drift_results.get("concept_drift", {})
    
    report_lines.append(f"Data Drift Detected: {'YES ⚠️' if data_drift.get('drift_detected') else 'NO ✓'}")
    report_lines.append(f"Target Drift Detected: {'YES ⚠️' if target_drift.get('drift_detected') else 'NO ✓'}")
    report_lines.append(f"Concept Drift Detected: {'YES ⚠️' if concept_drift.get('drift_detected') else 'NO ✓'}")
    report_lines.append("")
    
    # Data Drift Details
    if data_drift:
        report_lines.append("DATA DRIFT ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append(f"Average PSI Score: {data_drift.get('avg_psi', 0):.4f}")
        report_lines.append(f"Threshold: {data_drift.get('threshold', 0.3):.4f}")
        
        drifted_features = data_drift.get('drifted_features', [])
        if drifted_features:
            report_lines.append(f"\nDrifted Features ({len(drifted_features)}):")
            for feature in drifted_features[:10]:  # Top 10
                psi = data_drift.get('psi_scores', {}).get(feature, 0)
                report_lines.append(f"  - {feature}: PSI = {psi:.4f}")
        else:
            report_lines.append("\nNo significant feature drift detected.")
        report_lines.append("")
    
    # Target Drift Details
    if target_drift:
        report_lines.append("TARGET DRIFT ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append(f"Current Fraud Rate: {target_drift.get('current_fraud_rate', 0):.4%}")
        report_lines.append(f"Baseline Fraud Rate: {target_drift.get('baseline_fraud_rate', 0):.4%}")
        report_lines.append(f"Relative Change: {target_drift.get('relative_change', 0):.2%}")
        report_lines.append(f"Severity: {target_drift.get('severity', 'LOW')}")
        report_lines.append("")
    
    # Concept Drift Details
    if concept_drift:
        report_lines.append("CONCEPT DRIFT ANALYSIS (Model Performance)")
        report_lines.append("-" * 80)
        
        metrics = concept_drift.get('metrics', {})
        report_lines.append(f"Recall: {metrics.get('recall', 0):.4f} (Baseline: {metrics.get('baseline_recall', 0.98):.4f})")
        report_lines.append(f"Precision: {metrics.get('precision', 0):.4f} (Baseline: {metrics.get('baseline_precision', 0.95):.4f})")
        report_lines.append(f"FPR: {metrics.get('fpr', 0):.4f} (Baseline: {metrics.get('baseline_fpr', 0.015):.4f})")
        report_lines.append(f"F1 Score: {metrics.get('f1_score', 0):.4f} (Baseline: {metrics.get('baseline_f1', 0.965):.4f})")
        report_lines.append(f"\nSeverity: {concept_drift.get('severity', 'LOW')}")
        report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 80)
    
    recommendations = []
    if data_drift.get('drift_detected'):
        recommendations.append("⚠️ Data drift detected - investigate feature distribution changes")
    if target_drift.get('drift_detected'):
        recommendations.append("⚠️ Target drift detected - fraud rate has changed significantly")
    if concept_drift.get('drift_detected'):
        recommendations.append("🔴 Concept drift detected - model performance degraded, consider retraining")
    
    if not recommendations:
        recommendations.append("✓ No significant drift detected - continue monitoring")
    
    for rec in recommendations:
        report_lines.append(f"  {rec}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report = "\n".join(report_lines)
    
    logger.info("drift_report_generated", length=len(report))
    
    return report


def generate_alert_summary(
    alerts: List[Dict[str, Any]],
    time_window: timedelta = timedelta(hours=24)
) -> str:
    """
    Generate a summary of alerts triggered.
    
    Args:
        alerts: List of alert dictionaries
        time_window: Time window for alert summary
        
    Returns:
        Formatted alert summary as string
    """
    logger.info("generating_alert_summary", num_alerts=len(alerts))
    
    if not alerts:
        return "No alerts triggered in the specified time window."
    
    # Filter recent alerts
    cutoff_time = datetime.utcnow() - time_window
    recent_alerts = [
        alert for alert in alerts
        if datetime.fromisoformat(alert.get('timestamp', '1970-01-01')) >= cutoff_time
    ]
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(f"ALERT SUMMARY - Last {time_window.total_seconds() / 3600:.0f} hours")
    summary_lines.append("=" * 80)
    summary_lines.append(f"\nTotal Alerts: {len(recent_alerts)}")
    summary_lines.append("")
    
    # Group by severity
    by_severity = {}
    for alert in recent_alerts:
        severity = alert.get('severity', 'LOW')
        by_severity.setdefault(severity, []).append(alert)
    
    summary_lines.append("ALERTS BY SEVERITY:")
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = len(by_severity.get(severity, []))
        if count > 0:
            summary_lines.append(f"  {severity}: {count}")
    
    summary_lines.append("")
    
    # Recent alerts
    summary_lines.append("RECENT ALERTS (Last 10):")
    for alert in recent_alerts[:10]:
        timestamp = alert.get('timestamp', 'N/A')
        alert_type = alert.get('type', 'Unknown')
        severity = alert.get('severity', 'LOW')
        message = alert.get('message', 'No message')
        summary_lines.append(f"\n  [{timestamp}] {severity} - {alert_type}")
        summary_lines.append(f"  Message: {message}")
    
    summary_lines.append("")
    summary_lines.append("=" * 80)
    
    return "\n".join(summary_lines)


def export_to_html(
    report: str,
    output_path: str,
    title: str = "Drift Detection Report"
) -> None:
    """
    Export report to HTML format.
    
    Args:
        report: Report text
        output_path: Path to save HTML file
        title: HTML page title
    """
    logger.info("exporting_report_to_html", output_path=output_path)
    
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: #252526;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        pre {{
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .warning {{
            color: #ff9800;
        }}
        .success {{
            color: #4caf50;
        }}
        .error {{
            color: #f44336;
        }}
        h1 {{
            color: #569cd6;
            border-bottom: 2px solid #569cd6;
            padding-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <pre>{report}</pre>
    </div>
</body>
</html>
"""
    
    # Write to file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    logger.info("html_report_exported", path=output_path)


def export_to_json(
    drift_results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Export drift results to JSON format.
    
    Args:
        drift_results: Dictionary containing all drift detection results
        output_path: Path to save JSON file
    """
    logger.info("exporting_results_to_json", output_path=output_path)
    
    # Create output directory
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results_with_metadata = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        },
        "results": drift_results
    }
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_with_metadata, f, indent=2, default=str)
    
    logger.info("json_results_exported", path=output_path)
