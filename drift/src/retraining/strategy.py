"""
Retraining strategy module.

This module implements different retraining strategies for the fraud detection model.
"""

import structlog
from typing import Dict, Any, Optional
from datetime import datetime

logger = structlog.get_logger(__name__)


def immediate_retrain(
    drift_results: Dict[str, Any],
    training_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute immediate retraining strategy.
    
    This strategy triggers full model retraining immediately when drift is detected.
    Suitable for critical drift situations.
    
    Args:
        drift_results: Results from drift detection
        training_config: Configuration for training process
        
    Returns:
        Dictionary with retraining status and details
    """
    logger.info("executing_immediate_retrain_strategy")
    
    result = {
        "strategy": "immediate",
        "status": "initiated",
        "timestamp": datetime.utcnow().isoformat(),
        "drift_summary": {
            "data_drift": drift_results.get("data_drift", {}).get("drift_detected", False),
            "target_drift": drift_results.get("target_drift", {}).get("drift_detected", False),
            "concept_drift": drift_results.get("concept_drift", {}).get("drift_detected", False)
        },
        "training_mode": "full",
        "estimated_duration_hours": 4
    }
    
    logger.info("immediate_retrain_initiated", result=result)
    
    return result


def incremental_learning(
    drift_results: Dict[str, Any],
    new_data_size: int,
    training_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute incremental learning strategy.
    
    This strategy updates the model with new data without full retraining.
    Suitable for gradual drift or when quick updates are needed.
    
    Args:
        drift_results: Results from drift detection
        new_data_size: Number of new samples to train on
        training_config: Configuration for training process
        
    Returns:
        Dictionary with retraining status and details
    """
    logger.info("executing_incremental_learning_strategy", new_data_size=new_data_size)
    
    result = {
        "strategy": "incremental",
        "status": "initiated",
        "timestamp": datetime.utcnow().isoformat(),
        "drift_summary": {
            "data_drift": drift_results.get("data_drift", {}).get("drift_detected", False),
            "target_drift": drift_results.get("target_drift", {}).get("drift_detected", False),
            "concept_drift": drift_results.get("concept_drift", {}).get("drift_detected", False)
        },
        "training_mode": "incremental",
        "new_data_size": new_data_size,
        "estimated_duration_hours": 1
    }
    
    logger.info("incremental_learning_initiated", result=result)
    
    return result


def scheduled_retrain(
    schedule_time: str,
    drift_results: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Schedule retraining for a specific time.
    
    This strategy schedules model retraining for a future time,
    typically during off-peak hours.
    
    Args:
        schedule_time: ISO format timestamp for scheduled retraining
        drift_results: Results from drift detection (optional)
        training_config: Configuration for training process
        
    Returns:
        Dictionary with scheduling status and details
    """
    logger.info("scheduling_retrain", schedule_time=schedule_time)
    
    result = {
        "strategy": "scheduled",
        "status": "scheduled",
        "timestamp": datetime.utcnow().isoformat(),
        "scheduled_time": schedule_time,
        "drift_summary": {
            "data_drift": drift_results.get("data_drift", {}).get("drift_detected", False) if drift_results else False,
            "target_drift": drift_results.get("target_drift", {}).get("drift_detected", False) if drift_results else False,
            "concept_drift": drift_results.get("concept_drift", {}).get("drift_detected", False) if drift_results else False
        } if drift_results else {},
        "training_mode": "full",
        "estimated_duration_hours": 4
    }
    
    logger.info("retrain_scheduled", result=result)
    
    return result


def select_retraining_strategy(
    drift_results: Dict[str, Any],
    priority: str = "MEDIUM"
) -> str:
    """
    Select the appropriate retraining strategy based on drift severity.
    
    Args:
        drift_results: Results from drift detection
        priority: Priority level
        
    Returns:
        Strategy name: "immediate", "incremental", or "scheduled"
    """
    concept_severity = drift_results.get("concept_drift", {}).get("severity", "LOW")
    
    # Critical situations: immediate retraining
    if priority == "CRITICAL" or concept_severity == "CRITICAL":
        logger.info("strategy_selected", strategy="immediate", reason="critical_priority")
        return "immediate"
    
    # High priority: immediate or incremental based on drift type
    if priority == "HIGH":
        # If only data drift, incremental might be sufficient
        if (drift_results.get("data_drift", {}).get("drift_detected") and
            not drift_results.get("concept_drift", {}).get("drift_detected")):
            logger.info("strategy_selected", strategy="incremental", reason="data_drift_only")
            return "incremental"
        else:
            logger.info("strategy_selected", strategy="immediate", reason="high_priority")
            return "immediate"
    
    # Medium/Low priority: scheduled retraining
    logger.info("strategy_selected", strategy="scheduled", reason="medium_low_priority")
    return "scheduled"
