"""
Daily drift analysis pipeline.

This pipeline runs daily to:
1. Aggregate daily drift metrics
2. Generate comprehensive reports
3. Identify trends
4. Provide actionable recommendations
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog

from ..storage.database import DriftDatabaseService, query_historical_drift
from ..analysis.reporting import generate_drift_report, export_to_html
from ..analysis.comparison import compare_with_training, identify_new_patterns, extract_insights
from ..config.settings import Settings

logger = structlog.get_logger(__name__)


def aggregate_daily_metrics(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """
    Aggregate drift metrics for the last 24 hours.
    
    Args:
        settings: Configuration settings
        
    Returns:
        Dictionary with aggregated metrics
    """
    logger.info("aggregating_daily_metrics")
    
    settings = settings or Settings()
    
    try:
        # Query last 24 hours of drift data
        df = query_historical_drift(
            time_range=timedelta(hours=24),
            settings=settings
        )
        
        if df.empty:
            logger.warning("no_drift_data_for_aggregation")
            return {}
        
        # Aggregate by drift type
        aggregated = {
            "timestamp": datetime.utcnow().isoformat(),
            "period": "last_24h",
            "data_drift": {},
            "target_drift": {},
            "concept_drift": {}
        }
        
        for drift_type in ["data", "target", "concept"]:
            type_data = df[df["drift_type"] == drift_type]
            
            if not type_data.empty:
                aggregated[f"{drift_type}_drift"] = {
                    "avg_score": float(type_data["metric_value"].mean()),
                    "max_score": float(type_data["metric_value"].max()),
                    "min_score": float(type_data["metric_value"].min()),
                    "threshold_exceeded_count": int(type_data["threshold_exceeded"].sum()),
                    "total_measurements": len(type_data)
                }
        
        logger.info("daily_metrics_aggregated")
        return aggregated
    
    except Exception as e:
        logger.error("failed_to_aggregate_daily_metrics", error=str(e))
        return {}


def generate_daily_report(
    aggregated_metrics: Dict[str, Any],
    settings: Optional[Settings] = None
) -> str:
    """
    Generate comprehensive daily drift report.
    
    Args:
        aggregated_metrics: Aggregated daily metrics
        settings: Configuration settings
        
    Returns:
        Report string
    """
    logger.info("generating_daily_report")
    
    try:
        report = generate_drift_report(
            drift_results=aggregated_metrics,
            time_period="last_24h"
        )
        
        # Save report to file
        settings = settings or Settings()
        if settings.report_output_dir:
            timestamp = datetime.utcnow().strftime("%Y%m%d")
            output_path = f"{settings.report_output_dir}/drift_report_{timestamp}.html"
            export_to_html(report, output_path, title="Daily Drift Report")
            logger.info("daily_report_saved", path=output_path)
        
        return report
    
    except Exception as e:
        logger.error("failed_to_generate_daily_report", error=str(e))
        return ""


def identify_trends(settings: Optional[Settings] = None) -> List[Dict[str, Any]]:
    """
    Identify trends in drift metrics over the past week.
    
    Args:
        settings: Configuration settings
        
    Returns:
        List of identified trends
    """
    logger.info("identifying_trends")
    
    settings = settings or Settings()
    trends = []
    
    try:
        # Query last 7 days of data
        df = query_historical_drift(
            time_range=timedelta(days=7),
            settings=settings
        )
        
        if df.empty:
            logger.warning("no_data_for_trend_analysis")
            return trends
        
        # Analyze trends by drift type
        for drift_type in ["data", "target", "concept"]:
            type_data = df[df["drift_type"] == drift_type].copy()
            
            if len(type_data) < 2:
                continue
            
            # Sort by timestamp
            type_data = type_data.sort_values("timestamp")
            
            # Calculate trend (simple linear regression)
            x = range(len(type_data))
            y = type_data["metric_value"].values
            
            # Simple trend calculation (increasing/decreasing)
            if len(y) > 1:
                trend_direction = "increasing" if y[-1] > y[0] else "decreasing"
                trend_magnitude = abs(y[-1] - y[0]) / (y[0] + 1e-10)
                
                trends.append({
                    "drift_type": drift_type,
                    "direction": trend_direction,
                    "magnitude": float(trend_magnitude),
                    "start_value": float(y[0]),
                    "end_value": float(y[-1]),
                    "concern_level": "HIGH" if trend_magnitude > 0.5 else "MEDIUM" if trend_magnitude > 0.2 else "LOW"
                })
        
        logger.info("trends_identified", count=len(trends))
        return trends
    
    except Exception as e:
        logger.error("failed_to_identify_trends", error=str(e))
        return trends


def recommend_actions(
    aggregated_metrics: Dict[str, Any],
    trends: List[Dict[str, Any]]
) -> List[str]:
    """
    Generate actionable recommendations based on metrics and trends.
    
    Args:
        aggregated_metrics: Aggregated daily metrics
        trends: Identified trends
        
    Returns:
        List of recommendations
    """
    logger.info("generating_recommendations")
    
    recommendations = []
    
    # Check aggregated metrics
    data_drift = aggregated_metrics.get("data_drift", {})
    target_drift = aggregated_metrics.get("target_drift", {})
    concept_drift = aggregated_metrics.get("concept_drift", {})
    
    # Data drift recommendations
    if data_drift.get("threshold_exceeded_count", 0) > 12:  # More than half of hourly checks
        recommendations.append(
            "🔴 URGENT: Data drift detected in >50% of checks - investigate feature distribution changes"
        )
    
    # Target drift recommendations
    if target_drift.get("threshold_exceeded_count", 0) > 5:
        recommendations.append(
            "⚠️ Target drift detected - fraud rate has changed significantly, review business context"
        )
    
    # Concept drift recommendations
    if concept_drift.get("threshold_exceeded_count", 0) > 0:
        recommendations.append(
            "🔴 CRITICAL: Model performance degradation detected - schedule retraining immediately"
        )
    
    # Trend-based recommendations
    for trend in trends:
        if trend["direction"] == "increasing" and trend["concern_level"] in ["HIGH", "MEDIUM"]:
            recommendations.append(
                f"⚠️ {trend['drift_type'].capitalize()} drift trend increasing ({trend['magnitude']:.1%}) - monitor closely"
            )
    
    # General recommendations
    if not recommendations:
        recommendations.append("✓ No immediate action required - continue monitoring")
    
    logger.info("recommendations_generated", count=len(recommendations))
    return recommendations


def run_daily_analysis(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """
    Main entry point for daily analysis pipeline.
    
    Args:
        settings: Configuration settings
        
    Returns:
        Dictionary with pipeline execution results
    """
    logger.info("starting_daily_analysis_pipeline")
    
    settings = settings or Settings()
    
    try:
        # 1. Aggregate daily metrics
        aggregated_metrics = aggregate_daily_metrics(settings)
        
        if not aggregated_metrics:
            logger.warning("no_metrics_to_analyze")
            return {
                "status": "no_data",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # 2. Generate daily report
        report = generate_daily_report(aggregated_metrics, settings)
        
        # 3. Identify trends
        trends = identify_trends(settings)
        
        # 4. Generate recommendations
        recommendations = recommend_actions(aggregated_metrics, trends)
        
        result = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "aggregated_metrics": aggregated_metrics,
            "trends": trends,
            "recommendations": recommendations,
            "report_generated": bool(report)
        }
        
        logger.info("daily_analysis_pipeline_complete")
        
        return result
    
    except Exception as e:
        logger.error("daily_analysis_pipeline_failed", error=str(e))
        return {
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


if __name__ == "__main__":
    # Run pipeline when executed directly
    result = run_daily_analysis()
    print(f"Pipeline result: {result}")
