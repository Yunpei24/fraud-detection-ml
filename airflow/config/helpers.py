"""
Helper functions for Airflow DAGs
Reusable utilities for metrics, validation, alerts, and logging
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def format_metric_value(value: float, precision: int = 3) -> str:
    """
    Format metric value to consistent string representation.
    
    Args:
        value: Metric value to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value:.{precision}f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Previous value
        new_value: Current value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0 if new_value == 0 else 100.0
    
    return ((new_value - old_value) / old_value) * 100


def validate_training_metrics(
    metrics: Dict[str, float],
    min_recall: float = 0.80,
    min_precision: float = 0.70
) -> Dict[str, Any]:
    """
    Validate training metrics against thresholds.
    
    Args:
        metrics: Dictionary with 'recall' and 'precision' keys
        min_recall: Minimum acceptable recall
        min_precision: Minimum acceptable precision
        
    Returns:
        Dict with validation results and details
    """
    recall = metrics.get('recall', 0.0)
    precision = metrics.get('precision', 0.0)
    
    issues = []
    
    if recall < min_recall:
        issues.append(f"Recall {recall:.3f} below threshold {min_recall}")
    
    if precision < min_precision:
        issues.append(f"Precision {precision:.3f} below threshold {min_precision}")
    
    return {
        'is_valid': len(issues) == 0,
        'recall_ok': recall >= min_recall,
        'precision_ok': precision >= min_precision,
        'issues': issues,
        'recall': recall,
        'precision': precision
    }


def check_drift_severity(drift_metrics: Dict[str, Any]) -> str:
    """
    Determine drift severity level based on metrics.
    
    Args:
        drift_metrics: Dictionary with drift scores
        
    Returns:
        Severity level: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    """
    # Extract drift scores
    psi_score = drift_metrics.get('psi_score', 0.0)
    ks_stat = drift_metrics.get('ks_statistic', 0.0)
    chi2_pvalue = drift_metrics.get('chi2_pvalue', 1.0)
    
    # Critical: Strong drift on multiple metrics
    if psi_score > 0.25 or ks_stat > 0.3 or chi2_pvalue < 0.001:
        return 'CRITICAL'
    
    # High: Moderate drift on multiple metrics
    if psi_score > 0.15 or ks_stat > 0.2 or chi2_pvalue < 0.01:
        return 'HIGH'
    
    # Medium: Minor drift detected
    if psi_score > 0.10 or ks_stat > 0.1 or chi2_pvalue < 0.05:
        return 'MEDIUM'
    
    # Low: No significant drift
    return 'LOW'


def should_trigger_retraining(
    drift_severity: str,
    performance_drop: float,
    feedback_accuracy: float = 1.0
) -> Dict[str, Any]:
    """
    Determine if model retraining should be triggered.
    
    Args:
        drift_severity: Drift severity level (CRITICAL, HIGH, MEDIUM, LOW)
        performance_drop: Performance degradation percentage
        feedback_accuracy: Accuracy from analyst feedback (0-1)
        
    Returns:
        Dict with retraining decision and reasoning
    """
    reasons = []
    should_retrain = False
    
    # Critical drift always triggers retraining
    if drift_severity == 'CRITICAL':
        should_retrain = True
        reasons.append(f"Critical drift detected (severity: {drift_severity})")
    
    # High drift + performance drop triggers retraining
    if drift_severity == 'HIGH' and performance_drop > 5.0:
        should_retrain = True
        reasons.append(f"High drift with {performance_drop:.1f}% performance drop")
    
    # Significant performance drop alone triggers retraining
    if performance_drop > 10.0:
        should_retrain = True
        reasons.append(f"Significant performance drop: {performance_drop:.1f}%")
    
    # Poor feedback accuracy triggers retraining
    if feedback_accuracy < 0.85:
        should_retrain = True
        reasons.append(f"Low feedback accuracy: {feedback_accuracy:.2%}")
    
    return {
        'should_retrain': should_retrain,
        'reasons': reasons,
        'drift_severity': drift_severity,
        'performance_drop': performance_drop,
        'feedback_accuracy': feedback_accuracy
    }


def get_model_uri(model_name: str, version: str, stage: Optional[str] = None) -> str:
    """
    Construct MLflow model URI.
    
    Args:
        model_name: Name of the model
        version: Model version
        stage: Model stage (Production, Staging, None)
        
    Returns:
        MLflow model URI
    """
    if stage:
        return f"models:/{model_name}/{stage}"
    else:
        return f"models:/{model_name}/{version}"


def parse_databricks_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse Databricks job result.
    
    Args:
        result: Raw Databricks result
        
    Returns:
        Parsed result with metrics
    """
    state = result.get('state', {})
    life_cycle_state = state.get('life_cycle_state', 'UNKNOWN')
    result_state = state.get('result_state', 'UNKNOWN')
    
    return {
        'run_id': result.get('run_id'),
        'life_cycle_state': life_cycle_state,
        'result_state': result_state,
        'success': result_state == 'SUCCESS',
        'start_time': result.get('start_time'),
        'end_time': result.get('end_time'),
        'cluster_id': result.get('cluster_spec', {}).get('existing_cluster_id'),
        'notebook_output': result.get('notebook_output', {})
    }


def log_task_metrics(
    task_id: str,
    dag_id: str,
    metrics: Dict[str, Any],
    database_hook: Any
) -> None:
    """
    Log task metrics to airflow_task_metrics table.
    
    Args:
        task_id: Airflow task ID
        dag_id: Airflow DAG ID
        metrics: Metrics to log
        database_hook: Database hook for inserting metrics
    """
    try:
        import json
        
        query = """
            INSERT INTO airflow_task_metrics
            (dag_id, task_id, execution_date, metrics, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        database_hook.execute_query(
            query,
            (
                dag_id,
                task_id,
                datetime.now(),
                json.dumps(metrics),
                datetime.now()
            )
        )
        
        logger.info(f"Logged metrics for {dag_id}.{task_id}")
        
    except Exception as e:
        logger.error(f"Failed to log task metrics: {e}")


def calculate_training_priority(
    drift_score: float,
    performance_drop: float,
    feedback_count: int
) -> int:
    """
    Calculate training priority score (1-10, 10 being highest).
    
    Args:
        drift_score: Drift score (0-1)
        performance_drop: Performance degradation percentage
        feedback_count: Number of feedback samples
        
    Returns:
        Priority score 1-10
    """
    priority = 1
    
    # Drift contribution (0-4 points)
    if drift_score > 0.25:
        priority += 4
    elif drift_score > 0.15:
        priority += 3
    elif drift_score > 0.10:
        priority += 2
    elif drift_score > 0.05:
        priority += 1
    
    # Performance drop contribution (0-4 points)
    if performance_drop > 15:
        priority += 4
    elif performance_drop > 10:
        priority += 3
    elif performance_drop > 5:
        priority += 2
    elif performance_drop > 2:
        priority += 1
    
    # Feedback contribution (0-2 points)
    if feedback_count > 1000:
        priority += 2
    elif feedback_count > 500:
        priority += 1
    
    # Cap at 10
    return min(priority, 10)


def get_time_range_hours(hours: int) -> tuple:
    """
    Get time range for last N hours.
    
    Args:
        hours: Number of hours to look back
        
    Returns:
        Tuple of (start_time, end_time)
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    return start_time, end_time


def create_alert_message(
    alert_type: str,
    summary: str,
    details: str,
    severity: str = 'INFO'
) -> str:
    """
    Create formatted alert message.
    
    Args:
        alert_type: Type of alert (drift, performance, data_quality)
        summary: Brief summary
        details: Detailed information
        severity: Severity level
        
    Returns:
        Formatted alert message
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return f"""
🚨 [{severity}] {alert_type.upper()} ALERT
Time: {timestamp}

Summary: {summary}

Details:
{details}

---
Fraud Detection System - Airflow
"""


def batch_data(data: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Batch data into chunks.
    
    Args:
        data: List of data items
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
