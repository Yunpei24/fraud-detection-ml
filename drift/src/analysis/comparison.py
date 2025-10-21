"""
Comparison module for drift analysis.

This module provides functions to compare current data with training/baseline data
and identify new patterns and insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import structlog

logger = structlog.get_logger(__name__)


def compare_with_training(
    current_data: pd.DataFrame,
    training_data: pd.DataFrame,
    features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare current data distribution with training data.
    
    Args:
        current_data: Current production data
        training_data: Training dataset
        features: List of features to compare (if None, use all common features)
        
    Returns:
        Dictionary with comparison results
    """
    logger.info("comparing_with_training_data")
    
    if features is None:
        features = list(set(current_data.columns) & set(training_data.columns))
    
    comparison_results = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "num_features_compared": len(features),
        "statistical_tests": {},
        "distribution_shifts": {},
        "summary": {}
    }
    
    significant_shifts = []
    
    for feature in features:
        try:
            # KS test for distribution similarity
            ks_stat, ks_pvalue = stats.ks_2samp(
                training_data[feature].dropna(),
                current_data[feature].dropna()
            )
            
            # Mean and std comparison
            train_mean = training_data[feature].mean()
            current_mean = current_data[feature].mean()
            train_std = training_data[feature].std()
            current_std = current_data[feature].std()
            
            mean_shift = abs(current_mean - train_mean) / (train_std + 1e-10)
            std_ratio = current_std / (train_std + 1e-10)
            
            comparison_results["statistical_tests"][feature] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "significant_shift": ks_pvalue < 0.05
            }
            
            comparison_results["distribution_shifts"][feature] = {
                "train_mean": float(train_mean),
                "current_mean": float(current_mean),
                "train_std": float(train_std),
                "current_std": float(current_std),
                "mean_shift_in_std": float(mean_shift),
                "std_ratio": float(std_ratio)
            }
            
            if ks_pvalue < 0.05:
                significant_shifts.append(feature)
        
        except Exception as e:
            logger.warning(f"Error comparing feature {feature}", error=str(e))
            continue
    
    # Summary
    comparison_results["summary"] = {
        "features_with_significant_shift": significant_shifts,
        "num_shifted_features": len(significant_shifts),
        "shift_percentage": len(significant_shifts) / len(features) * 100 if features else 0
    }
    
    logger.info(
        "training_comparison_complete",
        num_shifted=len(significant_shifts),
        total_features=len(features)
    )
    
    return comparison_results


def identify_new_patterns(
    current_data: pd.DataFrame,
    training_data: pd.DataFrame,
    target_col: str = "Class"
) -> List[Dict[str, Any]]:
    """
    Identify new patterns in current data that weren't present in training.
    
    Args:
        current_data: Current production data
        training_data: Training dataset
        target_col: Name of target column
        
    Returns:
        List of identified new patterns
    """
    logger.info("identifying_new_patterns")
    
    new_patterns = []
    
    # 1. Check for new value ranges
    for col in current_data.columns:
        if col == target_col or col not in training_data.columns:
            continue
        
        try:
            train_min = training_data[col].min()
            train_max = training_data[col].max()
            current_min = current_data[col].min()
            current_max = current_data[col].max()
            
            # Out of range values
            if current_min < train_min or current_max > train_max:
                new_patterns.append({
                    "type": "out_of_range",
                    "feature": col,
                    "training_range": [float(train_min), float(train_max)],
                    "current_range": [float(current_min), float(current_max)],
                    "severity": "HIGH" if (current_min < train_min * 0.5 or current_max > train_max * 1.5) else "MEDIUM"
                })
        except Exception as e:
            logger.warning(f"Error checking range for {col}", error=str(e))
            continue
    
    # 2. Check for unusual correlations
    try:
        train_corr = training_data.corr()
        current_corr = current_data.corr()
        
        # Find correlations that changed significantly
        for i, col1 in enumerate(train_corr.columns):
            for col2 in train_corr.columns[i+1:]:
                train_corr_val = train_corr.loc[col1, col2]
                current_corr_val = current_corr.loc[col1, col2]
                
                # Significant correlation change
                if abs(current_corr_val - train_corr_val) > 0.3:
                    new_patterns.append({
                        "type": "correlation_shift",
                        "features": [col1, col2],
                        "training_correlation": float(train_corr_val),
                        "current_correlation": float(current_corr_val),
                        "change": float(current_corr_val - train_corr_val),
                        "severity": "HIGH" if abs(current_corr_val - train_corr_val) > 0.5 else "MEDIUM"
                    })
    except Exception as e:
        logger.warning("Error checking correlations", error=str(e))
    
    # 3. Check for anomalous clusters
    try:
        # Simple anomaly detection using IQR
        for col in current_data.select_dtypes(include=[np.number]).columns:
            if col == target_col:
                continue
            
            Q1_train = training_data[col].quantile(0.25)
            Q3_train = training_data[col].quantile(0.75)
            IQR_train = Q3_train - Q1_train
            
            lower_bound = Q1_train - 3 * IQR_train
            upper_bound = Q3_train + 3 * IQR_train
            
            outliers = current_data[
                (current_data[col] < lower_bound) | (current_data[col] > upper_bound)
            ]
            
            if len(outliers) > len(current_data) * 0.05:  # More than 5% outliers
                new_patterns.append({
                    "type": "unusual_outliers",
                    "feature": col,
                    "num_outliers": len(outliers),
                    "percentage": len(outliers) / len(current_data) * 100,
                    "severity": "HIGH" if len(outliers) > len(current_data) * 0.1 else "MEDIUM"
                })
    except Exception as e:
        logger.warning("Error detecting outliers", error=str(e))
    
    logger.info("new_patterns_identified", num_patterns=len(new_patterns))
    
    return new_patterns


def extract_insights(
    drift_results: Dict[str, Any],
    comparison_results: Dict[str, Any],
    new_patterns: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract actionable insights from drift detection results.
    
    Args:
        drift_results: Results from drift detection
        comparison_results: Results from training data comparison
        new_patterns: List of new patterns identified
        
    Returns:
        Dictionary with actionable insights
    """
    logger.info("extracting_insights")
    
    insights = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "overall_health": "HEALTHY",
        "critical_issues": [],
        "warnings": [],
        "recommendations": [],
        "monitoring_priorities": []
    }
    
    # Analyze drift severity
    data_drift = drift_results.get("data_drift", {})
    target_drift = drift_results.get("target_drift", {})
    concept_drift = drift_results.get("concept_drift", {})
    
    # Critical issues
    if concept_drift.get("drift_detected") and concept_drift.get("severity") in ["HIGH", "CRITICAL"]:
        insights["overall_health"] = "CRITICAL"
        insights["critical_issues"].append({
            "type": "model_degradation",
            "message": "Model performance has significantly degraded",
            "action": "Immediate retraining recommended"
        })
    
    if target_drift.get("drift_detected") and target_drift.get("severity") == "CRITICAL":
        insights["overall_health"] = "CRITICAL" if insights["overall_health"] != "CRITICAL" else "CRITICAL"
        insights["critical_issues"].append({
            "type": "target_shift",
            "message": f"Fraud rate changed by {target_drift.get('relative_change', 0):.1%}",
            "action": "Investigate business changes and consider retraining"
        })
    
    # Warnings
    if data_drift.get("drift_detected"):
        insights["warnings"].append({
            "type": "feature_drift",
            "message": f"{len(data_drift.get('drifted_features', []))} features showing significant drift",
            "features": data_drift.get('drifted_features', [])[:5]
        })
    
    if len(new_patterns) > 0:
        high_severity_patterns = [p for p in new_patterns if p.get("severity") == "HIGH"]
        if high_severity_patterns:
            insights["warnings"].append({
                "type": "new_patterns",
                "message": f"{len(high_severity_patterns)} high-severity new patterns detected",
                "patterns": high_severity_patterns[:3]
            })
    
    # Recommendations
    if insights["overall_health"] == "CRITICAL":
        insights["recommendations"].append("🔴 URGENT: Schedule immediate model retraining")
        insights["recommendations"].append("🔴 Review recent data quality and business changes")
    
    if data_drift.get("drift_detected"):
        insights["recommendations"].append("⚠️ Investigate root cause of feature distribution changes")
        insights["recommendations"].append("⚠️ Consider feature engineering updates")
    
    if len(new_patterns) > 0:
        insights["recommendations"].append("⚠️ Review new patterns with domain experts")
    
    # Monitoring priorities
    if data_drift.get("drifted_features"):
        insights["monitoring_priorities"].extend(data_drift.get("drifted_features", [])[:10])
    
    if not insights["critical_issues"] and not insights["warnings"]:
        insights["overall_health"] = "HEALTHY"
        insights["recommendations"].append("✓ Continue monitoring - no immediate action required")
    
    logger.info(
        "insights_extracted",
        health=insights["overall_health"],
        critical_issues=len(insights["critical_issues"]),
        warnings=len(insights["warnings"])
    )
    
    return insights
