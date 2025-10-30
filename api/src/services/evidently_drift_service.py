"""
Advanced Drift Detection Service using Evidently AI

This service provides comprehensive drift detection capabilities for fraud detection models,
supporting all three drift types with advanced statistical testing and automated monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDistributionMetric,
    ColumnDriftMetric,
    ColumnSummaryMetric,
    DatasetDriftMetric,
)
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestAllFeaturesValueDrift,
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
)

from .database_service import DatabaseService

logger = logging.getLogger(__name__)


class EvidentlyDriftService:
    """
    Advanced drift detection service using Evidently AI for comprehensive monitoring.

    Supports:
    - Data drift: Feature distribution changes
    - Target drift: Fraud rate changes
    - Concept drift: New fraud patterns
    - Statistical tests: KS, Chi-squared, PSI
    - Sliding windows for continuous monitoring
    - Multivariate drift detection
    - Automated drift reports
    """

    def __init__(self, database_service: DatabaseService):
        self.db = database_service

        # Define column mapping based on actual transaction schema
        self.column_mapping = ColumnMapping(
            target="is_fraud",
            prediction="fraud_score",  # From predictions table
            numerical_features=[
                "amount",
                "v1",
                "v2",
                "v3",
                "v4",
                "v5",
                "v6",
                "v7",
                "v8",
                "v9",
                "v10",
                "v11",
                "v12",
                "v13",
                "v14",
                "v15",
                "v16",
                "v17",
                "v18",
                "v19",
                "v20",
                "v21",
                "v22",
                "v23",
                "v24",
                "v25",
                "v26",
                "v27",
                "v28",
            ],
            categorical_features=[
                "transaction_type",
                "customer_country",
                "merchant_country",
                "currency",
                "device_id",
                "ip_address",
            ],
            datetime_features=["time"],
        )

    async def detect_comprehensive_drift(
        self, window_hours: int = 24, reference_window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Comprehensive drift detection covering all three drift types.

        Args:
            window_hours: Hours of current data to analyze
            reference_window_days: Days of reference data for comparison

        Returns:
            Comprehensive drift analysis results
        """
        try:
            # Get current and reference data
            current_data = await self._get_current_window_data(window_hours)
            reference_data = await self._get_reference_window_data(
                reference_window_days
            )

            if current_data.empty or reference_data.empty:
                return {
                    "error": "Insufficient data for drift detection",
                    "timestamp": datetime.utcnow(),
                }

            # Run comprehensive drift analysis
            results = {
                "timestamp": datetime.utcnow(),
                "analysis_window": f"{window_hours}h",
                "reference_window": f"{reference_window_days}d",
                "data_drift": await self._detect_data_drift(
                    current_data, reference_data
                ),
                "target_drift": await self._detect_target_drift(
                    current_data, reference_data
                ),
                "concept_drift": await self._detect_concept_drift(
                    current_data, reference_data
                ),
                "multivariate_drift": await self._detect_multivariate_drift(
                    current_data, reference_data
                ),
                "drift_summary": {},
            }

            # Generate summary
            results["drift_summary"] = self._generate_drift_summary(results)

            # Store results
            await self._store_comprehensive_results(results)

            # Store individual metrics in drift_metrics table
            await self._store_individual_metrics(results)

            return results

        except Exception as e:
            logger.error(f"Error in comprehensive drift detection: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow()}

    async def _detect_data_drift(
        self, current_data: pd.DataFrame, reference_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect data drift using multiple statistical tests.
        """
        try:
            # Create comprehensive data drift report
            report = Report(
                metrics=[
                    DatasetDriftMetric(),
                    ColumnDriftMetric(column_name="amount", stattest="ks"),
                    ColumnDriftMetric(column_name="v1", stattest="ks"),
                    ColumnDriftMetric(
                        column_name="transaction_type", stattest="chisquare"
                    ),
                    ColumnSummaryMetric(column_name="amount"),
                    ColumnDistributionMetric(column_name="amount"),
                ]
            )

            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )

            results = report.as_dict()

            # Extract key metrics
            data_drift = {
                "dataset_drift_detected": results["metrics"][0]["result"][
                    "dataset_drift"
                ],
                "drift_share": results["metrics"][0]["result"]["drift_share"],
                "drifted_columns": [],
                "statistical_tests": [],
            }

            # Analyze individual column drift
            for metric in results["metrics"]:
                if metric["metric"] == "ColumnDriftMetric":
                    result_data = metric["result"]
                    if result_data.get("drift_detected"):
                        data_drift["drifted_columns"].append(
                            {
                                "column": result_data.get("column_name", "unknown"),
                                "drift_score": result_data.get("drift_score", 0),
                                "stattest_name": result_data.get(
                                    "stattest_name", "unknown"
                                ),
                                "threshold": result_data.get(
                                    "threshold", 0.05
                                ),  # Default threshold
                            }
                        )

            return data_drift

        except Exception as e:
            logger.error(f"Error detecting data drift: {e}")
            return {"error": str(e)}

    async def _detect_target_drift(
        self, current_data: pd.DataFrame, reference_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect target drift (changes in fraud rate).
        """
        try:
            # Calculate fraud rates
            current_fraud_rate = current_data["is_fraud"].mean()
            reference_fraud_rate = reference_data["is_fraud"].mean()

            # Use PSI (Population Stability Index) for target drift
            target_drift_score = self._calculate_psi(
                reference_fraud_rate, current_fraud_rate
            )

            # Determine if drift is detected based on PSI threshold
            drift_detected = abs(target_drift_score) > 0.1  # PSI threshold

            return {
                "drift_detected": drift_detected,
                "drift_score": target_drift_score,
                "current_fraud_rate": current_fraud_rate,
                "reference_fraud_rate": reference_fraud_rate,
                "rate_change_percent": (
                    (current_fraud_rate - reference_fraud_rate) / reference_fraud_rate
                )
                * 100,
                "stattest": "psi_stat_test",
            }

        except Exception as e:
            logger.error(f"Error detecting target drift: {e}")
            return {"error": str(e)}

    async def _detect_concept_drift(
        self, current_data: pd.DataFrame, reference_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect concept drift (changes in relationship between features and target).
        Concept drift is detected by comparing the relationship between features and fraud labels.
        """
        try:
            # For concept drift, we compare the correlation between features and target
            # between reference and current datasets

            concept_drift_score = 0.0
            drift_detected = False

            # Check correlation changes for key numerical features
            key_features = ["amount", "v1", "v2", "v3"]

            for feature in key_features:
                if (
                    feature in current_data.columns
                    and feature in reference_data.columns
                ):
                    # Calculate correlation with target
                    current_corr = current_data[feature].corr(current_data["is_fraud"])
                    reference_corr = reference_data[feature].corr(
                        reference_data["is_fraud"]
                    )

                    if not (np.isnan(current_corr) or np.isnan(reference_corr)):
                        corr_change = abs(current_corr - reference_corr)
                        concept_drift_score += corr_change

                        # If correlation changed significantly, flag as drift
                        if corr_change > 0.2:  # Threshold for significant change
                            drift_detected = True

            # Normalize the score
            concept_drift_score = (
                concept_drift_score / len(key_features) if key_features else 0
            )

            return {
                "drift_detected": drift_detected,
                "drift_score": concept_drift_score,
                "stattest_name": "correlation_difference",
                "features_analyzed": key_features,
            }

        except Exception as e:
            logger.error(f"Error detecting concept drift: {e}")
            return {"error": str(e)}

    async def _detect_multivariate_drift(
        self, current_data: pd.DataFrame, reference_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect multivariate drift using dataset-level analysis.
        """
        try:
            # Use comprehensive test suite for multivariate analysis
            test_suite = TestSuite(
                tests=[
                    TestAllFeaturesValueDrift(),
                    TestShareOfDriftedColumns(),
                    TestNumberOfDriftedColumns(),
                ]
            )

            test_suite.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )

            results = test_suite.as_dict()

            multivariate_drift = {
                "tests": [],
                "overall_drift_detected": False,
                "drift_columns_count": 0,
            }

            for test in results["tests"]:
                test_result = {
                    "name": test["name"],
                    "status": test["status"],
                    "description": test["description"],
                    "parameters": test["parameters"],
                }
                multivariate_drift["tests"].append(test_result)

                if test["status"] == "FAIL":
                    multivariate_drift["overall_drift_detected"] = True
                    if "drifted_columns" in test["parameters"]:
                        multivariate_drift["drift_columns_count"] = len(
                            test["parameters"]["drifted_columns"]
                        )

            return multivariate_drift

        except Exception as e:
            logger.error(f"Error detecting multivariate drift: {e}")
            return {"error": str(e)}

    async def run_sliding_window_analysis(
        self,
        window_size_hours: int = 24,
        step_hours: int = 6,
        analysis_period_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Run sliding window analysis for continuous drift monitoring.

        Args:
            window_size_hours: Size of each analysis window
            step_hours: Hours to slide window each time
            analysis_period_days: Total period to analyze

        Returns:
            Sliding window drift analysis results
        """
        try:
            results = {
                "timestamp": datetime.utcnow(),
                "window_size": f"{window_size_hours}h",
                "step_size": f"{step_hours}h",
                "analysis_period": f"{analysis_period_days}d",
                "windows": [],
            }

            # Calculate number of windows
            total_hours = analysis_period_days * 24
            num_windows = int((total_hours - window_size_hours) / step_hours) + 1

            for i in range(num_windows):
                window_start = datetime.utcnow() - timedelta(
                    hours=total_hours - i * step_hours
                )
                window_end = window_start + timedelta(hours=window_size_hours)

                # Get data for this window
                window_data = await self._get_window_data(window_start, window_end)
                reference_data = await self._get_reference_window_data(
                    30
                )  # 30 days reference

                if not window_data.empty:
                    # Quick drift check for this window
                    window_drift = await self._quick_drift_check(
                        window_data, reference_data
                    )

                    results["windows"].append(
                        {
                            "window_id": i + 1,
                            "start_time": window_start,
                            "end_time": window_end,
                            "record_count": len(window_data),
                            "drift_detected": window_drift["drift_detected"],
                            "drift_score": window_drift["drift_score"],
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Error in sliding window analysis: {e}")
            return {"error": str(e)}

    async def generate_drift_report(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate automated drift report with recommendations.
        """
        try:
            report = {
                "timestamp": datetime.utcnow(),
                "summary": analysis_results.get("drift_summary", {}),
                "recommendations": [],
                "alerts": [],
                "severity": "LOW",
            }

            # Analyze results and generate recommendations
            if analysis_results.get("data_drift", {}).get("dataset_drift_detected"):
                report["alerts"].append(
                    {
                        "type": "DATA_DRIFT",
                        "severity": "HIGH",
                        "message": "Significant data drift detected in feature distributions",
                    }
                )
                report["recommendations"].append(
                    "Consider retraining model with recent data"
                )
                report["severity"] = "HIGH"

            if analysis_results.get("target_drift", {}).get("drift_detected"):
                report["alerts"].append(
                    {
                        "type": "TARGET_DRIFT",
                        "severity": "CRITICAL",
                        "message": "Fraud rate has changed significantly",
                    }
                )
                report["recommendations"].append(
                    "Immediate investigation required - fraud patterns may have changed"
                )
                report["severity"] = "CRITICAL"

            if analysis_results.get("concept_drift", {}).get("drift_detected"):
                report["alerts"].append(
                    {
                        "type": "CONCEPT_DRIFT",
                        "severity": "MEDIUM",
                        "message": "Relationship between features and fraud has changed",
                    }
                )
                report["recommendations"].append(
                    "Model may need recalibration or retraining"
                )

            # Store report
            await self._store_drift_report(report)

            return report

        except Exception as e:
            logger.error(f"Error generating drift report: {e}")
            return {"error": str(e)}

    def _calculate_psi(self, expected: float, actual: float, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) for drift detection.
        """
        try:
            # For binary target, create bins
            expected_dist = np.array([1 - expected, expected])
            actual_dist = np.array([1 - actual, actual])

            # Avoid division by zero
            expected_dist = np.where(expected_dist == 0, 0.0001, expected_dist)
            actual_dist = np.where(actual_dist == 0, 0.0001, actual_dist)

            psi = np.sum(
                (actual_dist - expected_dist) * np.log(actual_dist / expected_dist)
            )
            return psi

        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0

    def _generate_drift_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive drift summary.
        """
        summary = {
            "overall_drift_detected": False,
            "drift_types_detected": [],
            "severity_score": 0,
            "recommendations": [],
        }

        # Check each drift type
        drift_types = [
            "data_drift",
            "target_drift",
            "concept_drift",
            "multivariate_drift",
        ]

        for drift_type in drift_types:
            drift_result = results.get(drift_type, {})
            if isinstance(drift_result, dict) and drift_result.get("drift_detected"):
                summary["overall_drift_detected"] = True
                summary["drift_types_detected"].append(drift_type)
                summary["severity_score"] += 1

        # Generate recommendations based on severity
        if summary["severity_score"] >= 3:
            summary["recommendations"].append(
                "CRITICAL: Multiple drift types detected - immediate action required"
            )
        elif summary["severity_score"] >= 2:
            summary["recommendations"].append(
                "HIGH: Multiple drift patterns detected - investigate and consider retraining"
            )
        elif summary["severity_score"] >= 1:
            summary["recommendations"].append(
                "MEDIUM: Drift detected - monitor closely"
            )
        else:
            summary["recommendations"].append(
                "LOW: No significant drift detected - continue monitoring"
            )

        return summary

    async def _get_current_window_data(self, window_hours: int) -> pd.DataFrame:
        """
        Get current window data for drift analysis.
        """
        query = """
        SELECT
            t.transaction_id,
            t.amount,
            t.currency,
            t.time,
            t.is_fraud,
            t.v1, t.v2, t.v3, t.v4, t.v5, t.v6, t.v7, t.v8, t.v9, t.v10,
            t.v11, t.v12, t.v13, t.v14, t.v15, t.v16, t.v17, t.v18, t.v19, t.v20,
            t.v21, t.v22, t.v23, t.v24, t.v25, t.v26, t.v27, t.v28,
            t.transaction_type,
            t.customer_country,
            t.merchant_country,
            t.device_id,
            t.ip_address,
            p.fraud_score
        FROM transactions t
        LEFT JOIN predictions p ON t.transaction_id = p.transaction_id
        WHERE t.time >= %s
        ORDER BY t.time DESC
        LIMIT 10000
        """

        cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
        rows = await self.db.fetch_all(query, (cutoff_time,))
        return pd.DataFrame(rows)

    async def _get_reference_window_data(self, reference_days: int) -> pd.DataFrame:
        """
        Get reference window data for drift analysis.
        """
        query = """
        SELECT
            t.transaction_id,
            t.amount,
            t.currency,
            t.time,
            t.is_fraud,
            t.v1, t.v2, t.v3, t.v4, t.v5, t.v6, t.v7, t.v8, t.v9, t.v10,
            t.v11, t.v12, t.v13, t.v14, t.v15, t.v16, t.v17, t.v18, t.v19, t.v20,
            t.v21, t.v22, t.v23, t.v24, t.v25, t.v26, t.v27, t.v28,
            t.transaction_type,
            t.customer_country,
            t.merchant_country,
            t.device_id,
            t.ip_address,
            p.fraud_score
        FROM transactions t
        LEFT JOIN predictions p ON t.transaction_id = p.transaction_id
        WHERE t.time >= %s
        AND t.time < %s
        AND t.is_fraud = false  -- Use legitimate transactions as reference
        ORDER BY t.time DESC
        LIMIT 50000
        """

        start_time = datetime.utcnow() - timedelta(days=reference_days + 1)
        end_time = datetime.utcnow() - timedelta(days=1)
        rows = await self.db.fetch_all(query, (start_time, end_time))
        return pd.DataFrame(rows)

    async def _get_window_data(
        self, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """
        Get data for a specific time window.
        """
        query = """
        SELECT * FROM transactions
        WHERE time >= %s AND time < %s
        LIMIT 10000
        """
        rows = await self.db.fetch_all(query, (start_time, end_time))
        return pd.DataFrame(rows)

    async def _quick_drift_check(
        self, current_data: pd.DataFrame, reference_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Quick drift check for sliding window analysis.
        """
        try:
            # Simple drift check using dataset drift metric
            report = Report(metrics=[DatasetDriftMetric()])
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )

            results = report.as_dict()
            return {
                "drift_detected": results["metrics"][0]["result"]["dataset_drift"],
                "drift_score": results["metrics"][0]["result"]["drift_share"],
            }
        except:
            return {"drift_detected": False, "drift_score": 0.0}

    async def _store_comprehensive_results(self, results: Dict[str, Any]) -> None:
        """
        Store comprehensive drift analysis results.
        """
        query = """
        INSERT INTO drift_analysis_results (
            timestamp, analysis_window, reference_window,
            data_drift, target_drift, concept_drift, multivariate_drift,
            drift_summary
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        await self.db.execute(
            query,
            (
                results["timestamp"],
                results["analysis_window"],
                results["reference_window"],
                str(results["data_drift"]),
                str(results["target_drift"]),
                str(results["concept_drift"]),
                str(results["multivariate_drift"]),
                str(results["drift_summary"]),
            ),
        )

    async def _store_drift_report(self, report: Dict[str, Any]) -> None:
        """
        Store drift report.
        """
        query = """
        INSERT INTO drift_reports (timestamp, summary, recommendations, alerts, severity)
        VALUES (%s, %s, %s, %s, %s)
        """

        await self.db.execute(
            query,
            (
                report["timestamp"],
                str(report["summary"]),
                str(report["recommendations"]),
                str(report["alerts"]),
                report["severity"],
            ),
        )

    async def _store_individual_metrics(self, results: Dict[str, Any]) -> None:
        """
        Extract and store individual drift metrics in drift_metrics table.
        """
        try:
            metrics_to_store = []

            # Data drift metrics
            if "data_drift" in results and isinstance(results["data_drift"], dict):
                data_drift = results["data_drift"]
                if "dataset_drift_detected" in data_drift:
                    metrics_to_store.append(
                        {
                            "metric_type": "data_drift",
                            "metric_name": "dataset_drift_score",
                            "metric_value": float(data_drift.get("drift_score", 0)),
                            "threshold": 0.05,
                            "threshold_exceeded": data_drift.get(
                                "dataset_drift_detected", False
                            ),
                            "severity": (
                                "HIGH"
                                if data_drift.get("dataset_drift_detected")
                                else "LOW"
                            ),
                            "feature_name": None,
                            "details": {
                                "drifted_columns": data_drift.get("drifted_columns", [])
                            },
                        }
                    )

                # Individual column drift metrics
                for column_info in data_drift.get("drifted_columns", []):
                    if isinstance(column_info, dict):
                        metrics_to_store.append(
                            {
                                "metric_type": "data_drift",
                                "metric_name": f'column_drift_{column_info.get("column", "unknown")}',
                                "metric_value": float(
                                    column_info.get("drift_score", 0)
                                ),
                                "threshold": 0.05,
                                "threshold_exceeded": column_info.get(
                                    "drift_detected", False
                                ),
                                "severity": "MEDIUM",
                                "feature_name": column_info.get("column"),
                                "details": column_info,
                            }
                        )

            # Target drift metrics
            if "target_drift" in results and isinstance(results["target_drift"], dict):
                target_drift = results["target_drift"]
                if "drift_score" in target_drift:
                    metrics_to_store.append(
                        {
                            "metric_type": "target_drift",
                            "metric_name": "fraud_rate_psi",
                            "metric_value": float(target_drift.get("drift_score", 0)),
                            "threshold": 0.1,
                            "threshold_exceeded": target_drift.get(
                                "drift_detected", False
                            ),
                            "severity": (
                                "CRITICAL"
                                if target_drift.get("drift_detected")
                                else "LOW"
                            ),
                            "feature_name": "is_fraud",
                            "details": {
                                "current_fraud_rate": target_drift.get(
                                    "current_fraud_rate"
                                ),
                                "reference_fraud_rate": target_drift.get(
                                    "reference_fraud_rate"
                                ),
                                "rate_change_percent": target_drift.get(
                                    "rate_change_percent"
                                ),
                            },
                        }
                    )

            # Concept drift metrics
            if "concept_drift" in results and isinstance(
                results["concept_drift"], dict
            ):
                concept_drift = results["concept_drift"]
                if "drift_score" in concept_drift:
                    metrics_to_store.append(
                        {
                            "metric_type": "concept_drift",
                            "metric_name": "correlation_drift",
                            "metric_value": float(concept_drift.get("drift_score", 0)),
                            "threshold": 0.2,
                            "threshold_exceeded": concept_drift.get(
                                "drift_detected", False
                            ),
                            "severity": (
                                "MEDIUM"
                                if concept_drift.get("drift_detected")
                                else "LOW"
                            ),
                            "feature_name": None,
                            "details": {
                                "stattest_name": concept_drift.get("stattest_name"),
                                "features_analyzed": concept_drift.get(
                                    "features_analyzed", []
                                ),
                            },
                        }
                    )

            # Store all metrics
            if metrics_to_store:
                await self._store_drift_metrics("drift_detection", metrics_to_store)

        except Exception as e:
            logger.error(f"Error storing individual metrics: {e}")
            # Don't fail the whole operation if metric storage fails
