"""
Drift detection API routes for monitoring data and model drift.
"""

import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from ...api.dependencies import get_drift_service
from ...api.routes.auth import get_current_analyst_user
from ...config import get_logger
from ...models import (ComprehensiveDriftResponse, DriftReportResponse,
                       ErrorResponse, SlidingWindowAnalysisResponse)
from ...services.evidently_drift_service import EvidentlyDriftService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/drift", tags=["drift-detection"])


@router.post(
    "/comprehensive-detect",
    response_model=ComprehensiveDriftResponse,
    status_code=status.HTTP_200_OK,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Comprehensive drift detection",
    description="Run comprehensive drift detection covering all three drift types with Evidently AI",
    dependencies=[Depends(get_current_analyst_user)],
)
async def comprehensive_drift_detection(
    window_hours: int = 24,
    reference_window_days: int = 30,
    drift_service=Depends(get_drift_service),
):
    """
    Run comprehensive drift detection using Evidently AI.

    This endpoint performs:
    - Data drift detection (feature distribution changes)
    - Target drift detection (fraud rate changes)
    - Concept drift detection (feature-target relationship changes)
    - Multivariate drift analysis

    Results are stored in the drift_metrics table.

    Args:
        window_hours: Hours of current data to analyze (default: 24)
        reference_window_days: Days of reference data for comparison (default: 30)

    Returns:
        Comprehensive drift analysis results
    """
    start_time = time.time()

    try:
        result = await drift_service.detect_comprehensive_drift(
            window_hours=window_hours, reference_window_days=reference_window_days
        )

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_code": "E704",
                    "message": "Comprehensive drift detection failed",
                    "details": result,
                },
            )

        result["processing_time"] = time.time() - start_time

        logger.info(
            f"Comprehensive drift detection completed: drift_detected={result.get('drift_summary', {}).get('overall_drift_detected', False)}"
        )
        return result

    except Exception as e:
        logger.error(f"Comprehensive drift detection failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "E704",
                "message": "Comprehensive drift detection failed",
                "details": {"error": str(e)},
            },
        )


@router.post(
    "/sliding-window-analysis",
    response_model=SlidingWindowAnalysisResponse,
    status_code=status.HTTP_200_OK,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Sliding window drift analysis",
    description="Run sliding window analysis for continuous drift monitoring",
    dependencies=[Depends(get_current_analyst_user)],
)
async def sliding_window_analysis(
    window_size_hours: int = 24,
    step_hours: int = 6,
    analysis_period_days: int = 7,
    drift_service=Depends(get_drift_service),
):
    """
    Run sliding window analysis for continuous drift monitoring.

    Args:
        window_size_hours: Size of each analysis window (default: 24)
        step_hours: Hours to slide window each time (default: 6)
        analysis_period_days: Total period to analyze (default: 7)

    Returns:
        Sliding window drift analysis results
    """
    start_time = time.time()

    try:
        result = await drift_service.run_sliding_window_analysis(
            window_size_hours=window_size_hours,
            step_hours=step_hours,
            analysis_period_days=analysis_period_days,
        )

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_code": "E705",
                    "message": "Sliding window analysis failed",
                    "details": result,
                },
            )

        result["processing_time"] = time.time() - start_time

        logger.info(
            f"Sliding window analysis completed: {len(result.get('windows', []))} windows analyzed"
        )
        return result

    except Exception as e:
        logger.error(f"Sliding window analysis failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "E705",
                "message": "Sliding window analysis failed",
                "details": {"error": str(e)},
            },
        )


@router.post(
    "/generate-report",
    response_model=DriftReportResponse,
    status_code=status.HTTP_200_OK,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Generate drift report",
    description="Generate automated drift report with recommendations",
    dependencies=[Depends(get_current_analyst_user)],
)
async def generate_drift_report(
    analysis_results: dict, drift_service=Depends(get_drift_service)
):
    """
    Generate automated drift report with recommendations.

    Args:
        analysis_results: Results from drift analysis

    Returns:
        Drift report with recommendations and alerts
    """
    try:
        report = await drift_service.generate_drift_report(analysis_results)

        if "error" in report:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_code": "E706",
                    "message": "Report generation failed",
                    "details": report,
                },
            )

        logger.info(
            f"Drift report generated: severity={report.get('severity', 'UNKNOWN')}"
        )
        return report

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "E706",
                "message": "Report generation failed",
                "details": {"error": str(e)},
            },
        )
