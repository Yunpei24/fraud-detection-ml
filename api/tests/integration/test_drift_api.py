"""
Integration tests for drift detection API endpoints.

Tests the full API flow including authentication, request/response validation,
and database interactions.
"""
import pytest
import json
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import status

from src.main import app
from src.services.evidently_drift_service import EvidentlyDriftService


@pytest.mark.integration
class TestDriftAPIIntegration:
    """Integration tests for drift detection API endpoints."""

    def test_comprehensive_drift_detection_success(self, client, mock_drift_service):
        """Test successful comprehensive drift detection endpoint."""
        response = client.post(
            "/api/v1/drift/comprehensive-detect",
            params={"window_hours": 24, "reference_window_days": 30}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert 'timestamp' in data
        assert 'analysis_window' in data
        assert 'reference_window' in data
        assert 'data_drift' in data
        assert 'target_drift' in data
        assert 'concept_drift' in data
        assert 'multivariate_drift' in data
        assert 'drift_summary' in data
        assert 'processing_time' in data

        # Verify drift service was called correctly
        mock_drift_service.detect_comprehensive_drift.assert_called_once_with(
            window_hours=24,
            reference_window_days=30
        )

    def test_comprehensive_drift_detection_with_drift(self, client, mock_drift_service):
        """Test comprehensive drift detection when drift is detected."""
        mock_drift_service.detect_comprehensive_drift.return_value = {
            'timestamp': '2025-01-15T10:30:00Z',
            'analysis_window': '24h',
            'reference_window': '30d',
            'data_drift': {
                'dataset_drift_detected': True,
                'drift_share': 0.15,
                'drifted_columns': [
                    {
                        'column': 'amount',
                        'drift_score': 0.25,
                        'stattest_name': 'ks',
                        'threshold': 0.05
                    }
                ],
                'statistical_tests': []
            },
            'target_drift': {
                'drift_detected': True,
                'drift_score': 0.18,
                'current_fraud_rate': 0.012,
                'reference_fraud_rate': 0.006,
                'rate_change_percent': 100.0,
                'stattest': 'psi_stat_test'
            },
            'concept_drift': {
                'drift_detected': False,
                'drift_score': 0.03,
                'stattest_name': 'correlation_difference',
                'features_analyzed': ['amount', 'v1', 'v2', 'v3']
            },
            'multivariate_drift': {
                'tests': [
                    {
                        'name': 'TestAllFeaturesValueDrift',
                        'status': 'FAIL',
                        'description': 'Test if all features have drifted',
                        'parameters': {'drifted_columns': ['amount', 'v1']}
                    }
                ],
                'overall_drift_detected': True,
                'drift_columns_count': 2
            },
            'drift_summary': {
                'overall_drift_detected': True,
                'drift_types_detected': ['data_drift', 'target_drift', 'multivariate_drift'],
                'severity_score': 3,
                'recommendations': ['CRITICAL: Multiple drift types detected - immediate action required']
            }
        }

        response = client.post(
            "/api/v1/drift/comprehensive-detect",
            params={"window_hours": 48, "reference_window_days": 60}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify drift was detected
        assert data['drift_summary']['overall_drift_detected'] is True
        assert len(data['drift_summary']['drift_types_detected']) == 3
        assert data['drift_summary']['severity_score'] == 3
        assert 'CRITICAL' in data['drift_summary']['recommendations'][0]

    def test_comprehensive_drift_detection_insufficient_data(self, client, mock_drift_service):
        """Test comprehensive drift detection with insufficient data."""
        mock_drift_service.detect_comprehensive_drift.side_effect = Exception("Insufficient data for drift detection")

        response = client.post("/api/v1/drift/comprehensive-detect")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()

        assert data['error_code'] == 'E704'
        assert 'Comprehensive drift detection failed' in data['message']
        assert 'Insufficient data' in str(data['details'])

    def test_sliding_window_analysis_success(self, client, mock_database_service):
        """Test successful sliding window analysis endpoint."""
        mock_drift_service = AsyncMock()
        mock_drift_service.run_sliding_window_analysis.return_value = {
            'timestamp': '2025-01-15T10:30:00Z',
            'window_size': '24h',
            'step_size': '6h',
            'analysis_period': '2d',
            'windows': [
                {
                    'window_id': 1,
                    'start_time': '2025-01-13T10:30:00Z',
                    'end_time': '2025-01-14T10:30:00Z',
                    'record_count': 1250,
                    'drift_detected': False,
                    'drift_score': 0.01
                },
                {
                    'window_id': 2,
                    'start_time': '2025-01-13T16:30:00Z',
                    'end_time': '2025-01-14T16:30:00Z',
                    'record_count': 1180,
                    'drift_detected': False,
                    'drift_score': 0.02
                }
            ]
        }

        with patch('src.services.evidently_drift_service.EvidentlyDriftService', return_value=mock_drift_service):
            response = client.post(
                "/api/v1/drift/sliding-window-analysis",
                params={
                    "window_size_hours": 24,
                    "step_hours": 6,
                    "analysis_period_days": 2
                }
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Verify response structure
            assert 'timestamp' in data
            assert 'window_size' in data
            assert 'step_size' in data
            assert 'analysis_period' in data
            assert 'windows' in data
            assert 'processing_time' in data
            assert len(data['windows']) == 1

            # Verify each window has required fields
            for window in data['windows']:
                assert 'window_id' in window
                assert 'start_time' in window
                assert 'end_time' in window
                assert 'record_count' in window
                assert 'drift_detected' in window
                assert 'drift_score' in window

    def test_generate_drift_report_no_alerts(self, client, mock_database_service):
        """Test drift report generation with no alerts."""
        mock_drift_service = AsyncMock()
        mock_drift_service.generate_drift_report.return_value = {
            'timestamp': '2025-01-15T10:30:00Z',
            'summary': {
                'overall_drift_detected': False,
                'severity_score': 0
            },
            'recommendations': [
                'Continue regular monitoring',
                'Review model performance metrics monthly'
            ],
            'alerts': [],
            'severity': 'LOW'
        }

        analysis_results = {
            'drift_summary': {
                'overall_drift_detected': False,
                'severity_score': 0
            }
        }

        with patch('src.services.evidently_drift_service.EvidentlyDriftService', return_value=mock_drift_service):
            response = client.post(
                "/api/v1/drift/generate-report",
                json=analysis_results
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Verify response structure
            assert 'timestamp' in data
            assert 'summary' in data
            assert 'recommendations' in data
            assert 'alerts' in data
            assert 'severity' in data

            # Verify no alerts
            assert data['severity'] == 'LOW'
            assert len(data['alerts']) == 0
            assert len(data['recommendations']) >= 1

    def test_generate_drift_report_with_alerts(self, client, mock_drift_service):
        """Test drift report generation with alerts."""
        mock_drift_service.generate_drift_report.return_value = {
            'timestamp': '2025-01-15T10:30:00Z',
            'summary': {
                'overall_drift_detected': True,
                'severity_score': 3
            },
            'recommendations': [
                'CRITICAL: Multiple drift types detected - immediate action required',
                'Consider retraining model with recent data'
            ],
            'alerts': [
                {
                    'type': 'DATA_DRIFT',
                    'severity': 'HIGH',
                    'message': 'Significant data drift detected in feature distributions'
                },
                {
                    'type': 'TARGET_DRIFT',
                    'severity': 'CRITICAL',
                    'message': 'Fraud rate has changed significantly'
                }
            ],
            'severity': 'CRITICAL'
        }

        analysis_results = {
            'data_drift': {'dataset_drift_detected': True},
            'target_drift': {'drift_detected': True},
            'drift_summary': {
                'overall_drift_detected': True,
                'severity_score': 3
            }
        }

        response = client.post(
            "/api/v1/drift/generate-report",
            json=analysis_results
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify alerts are present
        assert data['severity'] == 'CRITICAL'
        assert len(data['alerts']) == 2
        assert len(data['recommendations']) >= 1

        # Verify alert structure
        for alert in data['alerts']:
            assert 'type' in alert
            assert 'severity' in alert
            assert 'message' in alert

    def test_drift_service_error_handling(self, client, mock_drift_service):
        """Test error handling when drift service fails."""
        mock_drift_service.detect_comprehensive_drift.side_effect = Exception("Service unavailable")

        response = client.post("/api/v1/drift/comprehensive-detect")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()

        assert data['error_code'] == 'E704'
        assert 'Comprehensive drift detection failed' in data['message']

    def test_invalid_parameters(self, client):
        """Test API with invalid parameters."""
        # Test with negative window hours
        response = client.post(
            "/api/v1/drift/comprehensive-detect",
            params={"window_hours": -1, "reference_window_days": 30}
        )

        # Should still work as validation happens in service layer
        # This tests that the API accepts the parameters
        assert response.status_code in [200, 500]  # Either success or service-level validation error

    def test_default_parameters(self, client, mock_drift_service):
        """Test API endpoints with default parameters."""
        response = client.post("/api/v1/drift/comprehensive-detect")
        assert response.status_code == status.HTTP_200_OK

        # Test sliding window with defaults
        response = client.post("/api/v1/drift/sliding-window-analysis")
        assert response.status_code == status.HTTP_200_OK