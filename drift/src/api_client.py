"""
API client for calling the fraud detection API services.

This module provides a simplified client to call the API's drift detection service
from the drift monitoring component.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

import requests
import structlog

logger = structlog.get_logger(__name__)


class FraudDetectionAPIClient:
    """
    Client for calling Fraud Detection API services with JWT authentication.

    This client is used by the drift monitoring component to call the API's
    drift detection service instead of running custom drift detection logic.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_token: Optional[str] = None,
    ):
        """
        Initialize API client with JWT authentication.

        Args:
            base_url: Base URL of the fraud detection API
            timeout: Request timeout in seconds
            username: API username for authentication
            password: API password for authentication
            auth_token: Pre-existing JWT token
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        # JWT Authentication
        self.username = username or os.getenv("API_USERNAME")
        self.password = password or os.getenv("API_PASSWORD")
        self.auth_token = auth_token or os.getenv("API_TOKEN")

        # Token expiry tracking
        self._token_expiry = None

        logger.info(
            "api_client_initialized",
            base_url=base_url,
            has_credentials=bool(self.username and self.password),
            has_token=bool(self.auth_token),
        )

        # Authenticate if credentials provided
        if not self.auth_token and self.username and self.password:
            self._authenticate()

    def _authenticate(self) -> bool:
        """
        Authenticate to API and obtain JWT token.

        Returns:
            True if successful
        """
        if not self.username or not self.password:
            logger.warning("no_credentials_provided")
            return False

        try:
            url = f"{self.base_url}/auth/login"

            payload = {
                "username": self.username,
                "password": self.password,
            }

            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            logger.info("authenticating_to_api", url=url, username=self.username)

            response = self.session.post(url, data=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                self.auth_token = result.get("access_token")
                expires_in = result.get("expires_in", 3600)
                self._token_expiry = time.time() + expires_in - 300

                logger.info("authentication_successful", expires_in=expires_in)
                return True
            else:
                logger.error(
                    "authentication_failed",
                    status=response.status_code,
                    response=response.text,
                )
                return False

        except Exception as e:
            logger.error("authentication_error", error=str(e))
            return False

    def _refresh_token_if_needed(self) -> bool:
        """
        Refresh JWT token if close to expiry.

        Returns:
            True if token is valid
        """
        if not self.auth_token:
            return self._authenticate()

        if self._token_expiry and time.time() >= self._token_expiry:
            logger.info("refreshing_token")

            try:
                url = f"{self.base_url}/auth/refresh"
                headers = {"Authorization": f"Bearer {self.auth_token}"}

                response = self.session.post(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    self.auth_token = result.get("access_token")
                    expires_in = result.get("expires_in", 3600)
                    self._token_expiry = time.time() + expires_in - 300

                    logger.info("token_refreshed")
                    return True
                else:
                    return self._authenticate()

            except Exception as e:
                logger.error("token_refresh_error", error=str(e))
                return self._authenticate()

        return True

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get headers with JWT authorization.

        Returns:
            Headers dict with Authorization
        """
        self._refresh_token_if_needed()

        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        return headers

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
            auth_token: Optional authentication token (overrides instance token)

        Returns:
            Drift detection results from the API
        """
        url = f"{self.base_url}/api/v1/drift/comprehensive-detect"

        headers = self._get_auth_headers()
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

            if response.status_code == 401:
                logger.warning("unauthorized_retrying_with_fresh_auth")
                self._authenticate()
                headers = self._get_auth_headers()
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
            auth_token: Optional authentication token (overrides instance token)

        Returns:
            Sliding window analysis results from the API
        """
        url = f"{self.base_url}/api/v1/drift/sliding-window-analysis"

        headers = self._get_auth_headers()
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

            if response.status_code == 401:
                logger.warning("unauthorized_retrying_with_fresh_auth")
                self._authenticate()
                headers = self._get_auth_headers()
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
            auth_token: Optional authentication token (overrides instance token)

        Returns:
            Generated drift report from the API
        """
        url = f"{self.base_url}/api/v1/drift/generate-report"

        headers = self._get_auth_headers()
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        payload = {"analysis_results": analysis_results}

        try:
            logger.info("calling_api_generate_report", url=url)

            response = self.session.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )

            if response.status_code == 401:
                logger.warning("unauthorized_retrying_with_fresh_auth")
                self._authenticate()
                headers = self._get_auth_headers()
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
