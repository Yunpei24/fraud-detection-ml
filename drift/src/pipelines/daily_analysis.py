"""
Daily drift analysis pipeline.

This pipeline runs daily to:
1. Run sliding window analysis via API
2. Generate comprehensive reports via API
3. Identify trends from API results
4. Provide actionable recommendations
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import structlog

from ..api_client import FraudDetectionAPIClient
from ..config.settings import Settings

logger = structlog.get_logger(__name__)


def aggregate_daily_metrics(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """
    Run sliding window analysis via API to get historical drift metrics.

    Args:
        settings: Configuration settings

    Returns:
        Dictionary with sliding window analysis results
    """
    logger.info("running_sliding_window_analysis_via_api")

    settings = settings or Settings()

    try:
        # Create API client
        api_client = FraudDetectionAPIClient(
            base_url=settings.api_base_url, timeout=settings.api_timeout
        )

        # Run sliding window analysis for the past 7 days
        result = api_client.run_sliding_window_analysis(
            window_size_hours=24,  # Daily windows
            step_hours=24,  # Daily steps
            analysis_period_days=7,  # Past week
            auth_token=getattr(settings, "api_auth_token", None),
        )

        if "error" in result:
            logger.error("sliding_window_analysis_failed", error=result["error"])
            return {}

        # Transform API results to expected format
        aggregated = {
            "timestamp": result.get("timestamp", datetime.utcnow().isoformat()),
            "analysis_period": result.get("analysis_period", "7d"),
            "window_size": result.get("window_size", "24h"),
            "total_windows": len(result.get("windows", [])),
            "windows": result.get("windows", []),
            "drift_summary": _summarize_sliding_window_results(
                result.get("windows", [])
            ),
        }

        logger.info(
            "sliding_window_analysis_completed", windows=len(result.get("windows", []))
        )
        return aggregated

    except Exception as e:
        logger.error("failed_to_run_sliding_window_analysis", error=str(e))
        return {}


def _summarize_sliding_window_results(windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize sliding window analysis results.

    Args:
        windows: List of window analysis results

    Returns:
        Summary statistics
    """
    if not windows:
        return {"total_windows": 0, "drift_detected_windows": 0, "avg_drift_score": 0.0}

    drift_detected_count = sum(1 for w in windows if w.get("drift_detected", False))
    avg_drift_score = sum(w.get("drift_score", 0) for w in windows) / len(windows)

    return {
        "total_windows": len(windows),
        "drift_detected_windows": drift_detected_count,
        "drift_detection_rate": drift_detected_count / len(windows),
        "avg_drift_score": avg_drift_score,
        "max_drift_score": max((w.get("drift_score", 0) for w in windows), default=0),
    }


def generate_daily_report(
    aggregated_metrics: Dict[str, Any], settings: Optional[Settings] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive daily drift report via API.

    Args:
        aggregated_metrics: Aggregated sliding window metrics
        settings: Configuration settings

    Returns:
        Report dictionary from API
    """
    logger.info("generating_daily_report_via_api")

    settings = settings or Settings()

    try:
        # Create API client
        api_client = FraudDetectionAPIClient(
            base_url=settings.api_base_url, timeout=settings.api_timeout
        )

        # Generate report using API
        report = api_client.generate_drift_report(
            analysis_results=aggregated_metrics,
            auth_token=getattr(settings, "api_auth_token", None),
        )

        if "error" in report:
            logger.error("api_report_generation_failed", error=report["error"])
            return {}

        # Save report locally if configured
        if settings.report_output_dir:
            timestamp = datetime.utcnow().strftime("%Y%m%d")
            output_path = f"{settings.report_output_dir}/drift_report_{timestamp}.json"

            import json

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info("daily_report_saved", path=output_path)

        return report

    except Exception as e:
        logger.error("failed_to_generate_daily_report", error=str(e))
        return {}


def identify_trends(settings: Optional[Settings] = None) -> List[Dict[str, Any]]:
    """
    Identify trends from sliding window analysis results.

    Args:
        settings: Configuration settings

    Returns:
        List of identified trends
    """
    logger.info("identifying_trends_from_sliding_window")

    settings = settings or Settings()
    trends = []

    try:
        # Get sliding window analysis results
        aggregated_metrics = aggregate_daily_metrics(settings)
        windows = aggregated_metrics.get("windows", [])

        if not windows:
            logger.warning("no_sliding_window_data_for_trend_analysis")
            return trends

        # Extract drift scores over time
        drift_scores = [w.get("drift_score", 0) for w in windows]

        if len(drift_scores) < 2:
            logger.warning("insufficient_data_for_trend_analysis")
            return trends

        # Calculate trend
        trend_direction = (
            "increasing" if drift_scores[-1] > drift_scores[0] else "decreasing"
        )
        trend_magnitude = abs(drift_scores[-1] - drift_scores[0]) / (
            drift_scores[0] + 1e-10
        )

        # Calculate drift detection frequency
        drift_detected_count = sum(1 for w in windows if w.get("drift_detected", False))
        detection_rate = drift_detected_count / len(windows)

        trends.append(
            {
                "drift_type": "overall",
                "direction": trend_direction,
                "magnitude": float(trend_magnitude),
                "start_value": float(drift_scores[0]),
                "end_value": float(drift_scores[-1]),
                "detection_rate": detection_rate,
                "concern_level": "HIGH"
                if trend_magnitude > 0.5 or detection_rate > 0.5
                else "MEDIUM"
                if trend_magnitude > 0.2 or detection_rate > 0.3
                else "LOW",
            }
        )

        logger.info("trends_identified", count=len(trends))
        return trends

    except Exception as e:
        logger.error("failed_to_identify_trends", error=str(e))
        return trends


def recommend_actions(
    aggregated_metrics: Dict[str, Any], trends: List[Dict[str, Any]]
) -> List[str]:
    """
    Generate actionable recommendations based on sliding window metrics and trends.

    Args:
        aggregated_metrics: Aggregated sliding window metrics
        trends: Identified trends

    Returns:
        List of recommendations
    """
    logger.info("generating_recommendations")

    recommendations = []

    # Check sliding window summary
    drift_summary = aggregated_metrics.get("drift_summary", {})
    total_windows = drift_summary.get("total_windows", 0)
    drift_detected_windows = drift_summary.get("drift_detected_windows", 0)
    avg_drift_score = drift_summary.get("avg_drift_score", 0)

    # High drift detection rate
    if total_windows > 0 and (drift_detected_windows / total_windows) > 0.5:
        recommendations.append(
            f" URGENT: Drift detected in {drift_detected_windows}/{total_windows} analysis windows - investigate immediately"
        )

    # High average drift score
    if avg_drift_score > 0.3:
        recommendations.append(
            f" High average drift score ({avg_drift_score:.2f}) across analysis windows - monitor closely"
        )

    # Trend-based recommendations
    for trend in trends:
        if trend["direction"] == "increasing" and trend["concern_level"] in [
            "HIGH",
            "MEDIUM",
        ]:
            recommendations.append(
                f" Drift trend increasing ({trend['magnitude']:.1%}) with {trend['detection_rate']:.1%} detection rate - monitor closely"
            )

        # High detection rate trend
        if trend.get("detection_rate", 0) > 0.4:
            recommendations.append(
                " CRITICAL: High drift detection frequency - consider model retraining"
            )

    # General recommendations
    if not recommendations:
        recommendations.append(
            "✓ No significant drift patterns detected - continue monitoring"
        )

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
            return {"status": "no_data", "timestamp": datetime.utcnow().isoformat()}

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
            "report_generated": bool(report),
        }

        logger.info("daily_analysis_pipeline_complete")

        return result

    except Exception as e:
        logger.error("daily_analysis_pipeline_failed", error=str(e))
        return {
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }


if __name__ == "__main__":
    # Run pipeline when executed directly
    result = run_daily_analysis()
    print(f"Pipeline result: {result}")
