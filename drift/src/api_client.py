"""
API client for calling the fraud detection API services.

This module provides a simplified client to call the API's drift detection service
from the drift monitoring component.
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

import requests
import structlog

logger = structlog.get_logger(__name__)


class FraudDetectionAPIClient:
    """
    Client for calling Fraud Detection API services.

    This client is used by the drift monitoring component to call the API's
    drift detection service instead of running custom drift detection logic.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize API client.

        Args:
            base_url: Base URL of the fraud detection API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        logger.info("api_client_initialized", base_url=base_url)

    def detect_comprehensive_drift(
        self,
        window_hours: int = 24,
        reference_window_days: int = 30,
        auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call the API's comprehensive drift detection endpoint.

        Args:
            window_hours: Hours of current data to analyze
            reference_window_days: Days of reference data for comparison
            auth_token: Optional authentication token

        Returns:
            Drift detection results from the API
        """
        url = f"{self.base_url}/api/v1/drift/comprehensive-detect"

        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        payload = {
            "window_hours": window_hours,
            "reference_window_days": reference_window_days,
        }

        try:
            logger.info(
                "calling_api_drift_detection", url=url, window_hours=window_hours
            )

            response = self.session.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()

            logger.info("api_drift_detection_successful")
            return result

        except requests.exceptions.RequestException as e:
            logger.error("api_drift_detection_failed", error=str(e))
            return {
                "error": f"API call failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
            }
        except json.JSONDecodeError as e:
            logger.error("api_response_parse_failed", error=str(e))
            return {
                "error": f"Failed to parse API response: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
            }

    def run_sliding_window_analysis(
        self,
        window_size_hours: int = 24,
        step_hours: int = 6,
        analysis_period_days: int = 7,
        auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call the API's sliding window analysis endpoint.

        Args:
            window_size_hours: Size of each analysis window
            step_hours: Hours to slide window each time
            analysis_period_days: Total period to analyze
            auth_token: Optional authentication token

        Returns:
            Sliding window analysis results from the API
        """
        url = f"{self.base_url}/api/v1/drift/sliding-window-analysis"

        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        payload = {
            "window_size_hours": window_size_hours,
            "step_hours": step_hours,
            "analysis_period_days": analysis_period_days,
        }

        try:
            logger.info("calling_api_sliding_window_analysis", url=url)

            response = self.session.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()

            logger.info("api_sliding_window_analysis_successful")
            return result

        except requests.exceptions.RequestException as e:
            logger.error("api_sliding_window_analysis_failed", error=str(e))
            return {
                "error": f"API call failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
            }
        except json.JSONDecodeError as e:
            logger.error("api_response_parse_failed", error=str(e))
            return {
                "error": f"Failed to parse API response: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
            }

    def generate_drift_report(
        self, analysis_results: Dict[str, Any], auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call the API's drift report generation endpoint.

        Args:
            analysis_results: Results from drift analysis
            auth_token: Optional authentication token

        Returns:
            Generated drift report from the API
        """
        url = f"{self.base_url}/api/v1/drift/generate-report"

        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        payload = {"analysis_results": analysis_results}

        try:
            logger.info("calling_api_generate_report", url=url)

            response = self.session.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()

            logger.info("api_generate_report_successful")
            return result

        except requests.exceptions.RequestException as e:
            logger.error("api_generate_report_failed", error=str(e))
            return {
                "error": f"API call failed: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
            }
        except json.JSONDecodeError as e:
            logger.error("api_response_parse_failed", error=str(e))
            return {
                "error": f"Failed to parse API response: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
            }

    def health_check(self) -> bool:
        """
        Check if the API is healthy and accessible.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            url = f"{self.base_url}/health"
            response = self.session.get(url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False


# Convenience function for drift detection
def detect_drift_via_api(
    window_hours: int = 24,
    reference_window_days: int = 30,
    api_base_url: str = "http://localhost:8000",
    auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to detect drift via API.

    Args:
        window_hours: Hours of current data to analyze
        reference_window_days: Days of reference data for comparison
        api_base_url: Base URL of the API
        auth_token: Optional authentication token

    Returns:
        Drift detection results
    """
    client = FraudDetectionAPIClient(base_url=api_base_url)
    return client.detect_comprehensive_drift(
        window_hours=window_hours,
        reference_window_days=reference_window_days,
        auth_token=auth_token,
    )
