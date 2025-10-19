"""
Real-time data pipeline for streaming transactions
"""

import logging
import time
from typing import Callable, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RealtimePipeline:
    """
    Real-time pipeline for processing streaming transactions from Event Hub or Kafka
    Flow: Ingest -> Validate -> Transform -> Store
    """

    def __init__(self, batch_size: int = 100, flush_interval_seconds: int = 60):
        """
        Initialize real-time pipeline
        
        Args:
            batch_size: Batch size for processing
            flush_interval_seconds: Flush accumulated data after this interval
        """
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.buffer = []
        self.last_flush = datetime.utcnow()
        self.metrics = {
            "total_processed": 0,
            "total_errors": 0,
            "batches_processed": 0
        }

    def process_event(
        self,
        event: dict,
        validator,
        transformer,
        storage_service,
        metrics_collector=None
    ) -> bool:
        """
        Process a single event in the pipeline
        
        Args:
            event: Raw event data
            validator: Schema validator instance
            transformer: Data transformer instance
            storage_service: Database service instance
            metrics_collector: Optional metrics collector
        
        Returns:
            Success status
        """
        try:
            # 1. Validation
            is_valid, validation_report = validator.validate(event)
            
            if not is_valid:
                logger.warning(f"Validation failed: {validation_report['errors']}")
                self.metrics["total_errors"] += 1
                if metrics_collector:
                    metrics_collector.record_validation_error()
                return False

            # 2. Add to buffer
            self.buffer.append(event)

            # 3. Check if buffer should be flushed
            should_flush = (
                len(self.buffer) >= self.batch_size or
                (datetime.utcnow() - self.last_flush).total_seconds() > self.flush_interval_seconds
            )

            if should_flush:
                self._flush_buffer(transformer, storage_service, metrics_collector)

            self.metrics["total_processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Error processing event: {str(e)}")
            self.metrics["total_errors"] += 1
            return False

    def _flush_buffer(self, transformer, storage_service, metrics_collector=None) -> None:
        """
        Flush accumulated events in buffer to storage
        
        Args:
            transformer: Data transformer
            storage_service: Storage service
            metrics_collector: Optional metrics collector
        """
        if not self.buffer:
            return

        try:
            start_time = time.time()

            # Transform batch
            transformed_data = transformer.clean_pipeline(self.buffer)

            # Store batch
            storage_service.insert_transactions(self.buffer)

            # Record metrics
            elapsed = time.time() - start_time
            self.metrics["batches_processed"] += 1
            self.last_flush = datetime.utcnow()

            if metrics_collector:
                metrics_collector.record_transaction_processed(len(self.buffer))
                metrics_collector.record_processing_latency(elapsed)

            logger.info(f"Flushed {len(self.buffer)} events in {elapsed:.2f}s")
            self.buffer = []

        except Exception as e:
            logger.error(f"Error flushing buffer: {str(e)}")
            self.metrics["total_errors"] += len(self.buffer)

    def get_metrics(self) -> dict:
        """Get pipeline metrics"""
        return {
            **self.metrics,
            "buffer_size": len(self.buffer),
            "last_flush": self.last_flush.isoformat()
        }

    def shutdown(self, transformer, storage_service, metrics_collector=None) -> None:
        """
        Graceful shutdown - flush remaining buffer
        
        Args:
            transformer: Data transformer
            storage_service: Storage service
            metrics_collector: Optional metrics collector
        """
        logger.info("Shutting down real-time pipeline...")
        self._flush_buffer(transformer, storage_service, metrics_collector)
        logger.info(f"Pipeline metrics: {self.get_metrics()}")
