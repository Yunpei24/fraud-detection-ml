"""
Hourly drift monitoring pipeline.

This pipeline runs every hour to:
1. Call the API's drift detection service
2. Check against thresholds
3. Trigger alerts if needed
4. Update monitoring dashboards
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import time
import structlog

from ..api_client import FraudDetectionAPIClient
from ..alerting.alert_manager import AlertManager
from ..monitoring.metrics import update_drift_metrics
from ..retraining.trigger import RetrainingTrigger
from ..config.settings import Settings

logger = structlog.get_logger(__name__)


def fetch_recent_predictions(hours: int = 1, settings: Optional[Settings] = None) -> pd.DataFrame:
    """
    Fetch recent predictions from database.
    
    Args:
        hours: Number of hours to look back
        settings: Configuration settings
        
    Returns:
        DataFrame with recent predictions
    """
    logger.info("fetching_recent_predictions", hours=hours)
    
    try:
        settings = settings or Settings()
        
        # TODO: Implement actual database query
        # For now, return empty DataFrame as placeholder
        df = pd.DataFrame()
        
        logger.info("predictions_fetched", count=len(df))
        return df
    
    except Exception as e:
        logger.error("failed_to_fetch_predictions", error=str(e))
        return pd.DataFrame()


def call_api_drift_detection(
    window_hours: int = 1,
    reference_window_days: int = 30,
    settings: Optional[Settings] = None
) -> Dict[str, Any]:
    """
    Call the API's drift detection service.

    Args:
        window_hours: Hours of current data to analyze
        reference_window_days: Days of reference data for comparison
        settings: Configuration settings

    Returns:
        Drift detection results from the API
    """
    logger.info("calling_api_drift_detection", window_hours=window_hours)

    settings = settings or Settings()

    # Create API client
    api_client = FraudDetectionAPIClient(
        base_url=settings.api_base_url,
        timeout=settings.api_timeout
    )

    # Call the API's comprehensive drift detection
    result = api_client.detect_comprehensive_drift(
        window_hours=window_hours,
        reference_window_days=reference_window_days,
        auth_token=getattr(settings, 'api_auth_token', None)
    )

    if 'error' in result:
        logger.error("api_drift_detection_failed", error=result['error'])
        return result

    # Transform API results to match expected format for downstream processing
    transformed_result = {
        "timestamp": result.get('timestamp', datetime.utcnow().isoformat()),
        "data_drift": result.get('data_drift', {}),
        "target_drift": result.get('target_drift', {}),
        "concept_drift": result.get('concept_drift', {}),
        "multivariate_drift": result.get('multivariate_drift', {}),
        "drift_summary": result.get('drift_summary', {}),
        "api_response": result  # Keep original API response for reference
    }

    logger.info("api_drift_detection_completed")
    return transformed_result


def check_thresholds(drift_results: Dict[str, Any]) -> bool:
    """
    Check if any drift thresholds were exceeded.
    
    Args:
        drift_results: Results from API drift detection
        
    Returns:
        True if any threshold exceeded, False otherwise
    """
    # Check the drift summary from API response
    drift_summary = drift_results.get("drift_summary", {})
    overall_drift_detected = drift_summary.get("overall_drift_detected", False)
    
    # Also check individual drift types for backward compatibility
    data_drift = drift_results.get("data_drift", {}).get("drift_detected", False)
    target_drift = drift_results.get("target_drift", {}).get("drift_detected", False)
    concept_drift = drift_results.get("concept_drift", {}).get("drift_detected", False)
    multivariate_drift = drift_results.get("multivariate_drift", {}).get("drift_detected", False)
    
    exceeded = overall_drift_detected or data_drift or target_drift or concept_drift or multivariate_drift
    
    logger.info(
        "threshold_check_complete",
        overall_drift=overall_drift_detected,
        data_drift=data_drift,
        target_drift=target_drift,
        concept_drift=concept_drift,
        multivariate_drift=multivariate_drift,
        exceeded=exceeded
    )
    
    return exceeded


def trigger_alerts(
    drift_results: Dict[str, Any],
    settings: Optional[Settings] = None
) -> None:
    """
    Trigger alerts if drift detected.
    
    Args:
        drift_results: Results from API drift detection
        settings: Configuration settings
    """
    logger.info("checking_for_alerts")
    
    settings = settings or Settings()
    alert_manager = AlertManager(settings)
    
    # Check overall drift detection from API
    drift_summary = drift_results.get("drift_summary", {})
    if drift_summary.get("overall_drift_detected", False):
        severity = "HIGH" if drift_summary.get("severity_score", 0) >= 3 else "MEDIUM"
        alert_manager.trigger_alert(
            alert_type="drift_detected",
            severity=severity,
            message=f"Drift detected: {drift_summary.get('drift_types_detected', [])}",
            details=drift_summary
        )
        return
    
    # Fallback: Check individual drift types for backward compatibility
    # Data drift alert
    if drift_results.get("data_drift", {}).get("drift_detected"):
        data_drift = drift_results["data_drift"]
        alert_manager.trigger_alert(
            alert_type="data_drift",
            severity="HIGH" if data_drift.get("avg_psi", 0) > 0.5 else "MEDIUM",
            message=f"Data drift detected: PSI = {data_drift.get('avg_psi', 0):.3f}",
            details=data_drift
        )
    
    # Target drift alert
    if drift_results.get("target_drift", {}).get("drift_detected"):
        target_drift = drift_results["target_drift"]
        alert_manager.trigger_alert(
            alert_type="target_drift",
            severity=target_drift.get("severity", "MEDIUM"),
            message=f"Target drift detected: Fraud rate changed by {target_drift.get('relative_change', 0):.1%}",
            details=target_drift
        )
    
    # Concept drift alert
    if drift_results.get("concept_drift", {}).get("drift_detected"):
        concept_drift = drift_results["concept_drift"]
        alert_manager.trigger_alert(
            alert_type="concept_drift",
            severity=concept_drift.get("severity", "MEDIUM"),
            message="Model performance degradation detected",
            details=concept_drift
        )


def update_dashboards(drift_results: Dict[str, Any]) -> None:
    """
    Update monitoring dashboards with latest drift results.
    
    Args:
        drift_results: Results from drift detection
    """
    logger.info("updating_dashboards")
    
    # Update Prometheus metrics
    update_drift_metrics(drift_results)
    
    # TODO: Update Grafana dashboards via API if needed
    logger.debug("dashboards_updated")


def run_hourly_monitoring(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """
    Main entry point for hourly monitoring pipeline.
    
    Args:
        settings: Configuration settings
        
    Returns:
        Dictionary with pipeline execution results
    """
    logger.info("starting_hourly_monitoring_pipeline")
    
    settings = settings or Settings()
    
    try:
        # 1. Call API drift detection service
        drift_results = call_api_drift_detection(
            window_hours=1,  # Analyze last 1 hour of data
            reference_window_days=settings.reference_window_days,
            settings=settings
        )
        
        if 'error' in drift_results:
            logger.error("drift_detection_failed", error=drift_results['error'])
            return {
                "status": "api_error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": drift_results['error']
            }
        
        # 2. Check thresholds using API results
        threshold_exceeded = check_thresholds(drift_results)
        
        # 3. Trigger alerts if needed
        if threshold_exceeded:
            trigger_alerts(drift_results, settings)
        
        # 4. Update dashboards
        update_dashboards(drift_results)
        
        # 6. Check for retraining triggers
        retraining_trigger = RetrainingTrigger(settings)
        should_retrain, reason = retraining_trigger.should_retrain(drift_results)
        
        if should_retrain:
            priority = retraining_trigger.get_retrain_priority(drift_results)
            logger.warning(
                "triggering_retraining",
                priority=priority,
                reason=reason
            )
            
            # Trigger Airflow DAG 01_training_pipeline
            success = retraining_trigger.trigger_airflow_dag(
                dag_id="01_training_pipeline",
                priority=priority,
                conf={
                    "triggered_by": "drift_detection",
                    "drift_results": drift_results,
                    "reason": reason
                }
            )
            
            if success:
                logger.info("retraining_triggered_successfully", dag_id="01_training_pipeline")
            else:
                logger.error("retraining_trigger_failed", dag_id="01_training_pipeline")
        else:
            logger.info("retraining_not_needed", reason=reason)
        
        result = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "drift_results": drift_results,
            "threshold_exceeded": threshold_exceeded,
            "retraining_triggered": should_retrain,
            "retraining_reason": reason if should_retrain else None
        }
        
        logger.info("hourly_monitoring_pipeline_complete", result=result["status"])
        
        return result
    
    except Exception as e:
        logger.error("hourly_monitoring_pipeline_failed", error=str(e))
        return {
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


if __name__ == "__main__":
    # Run pipeline when executed directly
    result = run_hourly_monitoring()
    print(f"Pipeline result: {result}")
