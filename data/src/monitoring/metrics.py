"""
Metrics collection for Prometheus monitoring
"""

import logging
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects metrics for monitoring data pipeline health
    Integrates with Prometheus
    """

    def __init__(self, namespace: str = "fraud_detection_data"):
        """
        Initialize metrics collector
        
        Args:
            namespace: Prometheus metric namespace
        """
        self.namespace = namespace
        self.metrics = {}
        self._init_prometheus()

    def _init_prometheus(self) -> None:
        """Initialize Prometheus metrics"""
        try:
            from prometheus_client import Counter, Histogram, Gauge

            # Counters
            self.transactions_processed = Counter(
                f'{self.namespace}_transactions_processed_total',
                'Total transactions processed'
            )
            self.transactions_ingested = Counter(
                f'{self.namespace}_transactions_ingested_total',
                'Total transactions ingested'
            )
            self.validation_errors = Counter(
                f'{self.namespace}_validation_errors_total',
                'Total validation errors'
            )
            self.data_quality_issues = Counter(
                f'{self.namespace}_data_quality_issues_total',
                'Total data quality issues'
            )

            # Histograms
            self.ingestion_latency = Histogram(
                f'{self.namespace}_ingestion_latency_seconds',
                'Time taken to ingest batch'
            )
            self.processing_latency = Histogram(
                f'{self.namespace}_processing_latency_seconds',
                'Time taken to process batch'
            )
            self.validation_latency = Histogram(
                f'{self.namespace}_validation_latency_seconds',
                'Time taken to validate batch'
            )

            # Gauges
            self.active_connections = Gauge(
                f'{self.namespace}_active_connections',
                'Number of active connections'
            )
            self.queue_size = Gauge(
                f'{self.namespace}_queue_size',
                'Size of processing queue'
            )
            self.last_processed_timestamp = Gauge(
                f'{self.namespace}_last_processed_timestamp',
                'Timestamp of last processed transaction'
            )

            logger.info("Prometheus metrics initialized")

        except ImportError:
            logger.warning("prometheus_client not installed, metrics disabled")
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {str(e)}")

    def record_transaction_processed(self, count: int = 1) -> None:
        """Record processed transactions"""
        self.transactions_processed.inc(count)

    def record_transaction_ingested(self, count: int = 1) -> None:
        """Record ingested transactions"""
        self.transactions_ingested.inc(count)

    def record_validation_error(self) -> None:
        """Record validation error"""
        self.validation_errors.inc()

    def record_data_quality_issue(self, count: int = 1) -> None:
        """Record data quality issues"""
        self.data_quality_issues.inc(count)

    def record_ingestion_latency(self, seconds: float) -> None:
        """Record ingestion latency"""
        self.ingestion_latency.observe(seconds)

    def record_processing_latency(self, seconds: float) -> None:
        """Record processing latency"""
        self.processing_latency.observe(seconds)

    def record_validation_latency(self, seconds: float) -> None:
        """Record validation latency"""
        self.validation_latency.observe(seconds)

    def set_active_connections(self, count: int) -> None:
        """Set number of active connections"""
        self.active_connections.set(count)

    def set_queue_size(self, size: int) -> None:
        """Set queue size"""
        self.queue_size.set(size)

    def set_last_processed_timestamp(self) -> None:
        """Set last processed timestamp"""
        self.last_processed_timestamp.set(datetime.utcnow().timestamp())

    def get_metrics_summary(self) -> Dict:
        """Get summary of all collected metrics"""
        return {
            "namespace": self.namespace,
            "metrics_initialized": True,
            "collection_time": datetime.utcnow().isoformat()
        }
