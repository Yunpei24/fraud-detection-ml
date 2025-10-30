"""
Real-time Pipeline - Consumes Kafka, predicts via API, saves to PostgreSQL

This unified pipeline replaces batch_pipeline.py and handles:
1. Kafka consumption (batch or stream)
2. Cleaning and preprocessing
3. Prediction via API (batch or unit)
4. Saving to PostgreSQL
5. Sending results to Web App

Usage:
    # Batch mode (1000 transactions)
    python -m src.pipelines.realtime_pipeline --mode batch --count 1000

    # Streaming mode (continuous, 10 txn/sec)
    python -m src.pipelines.realtime_pipeline --mode stream --interval 10

Translated with DeepL.com (free version)
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("kafka-python not installed")

from src.config.constants import (API_TIMEOUT_SECONDS, API_URL,
                                  KAFKA_AUTO_OFFSET_RESET,
                                  KAFKA_BOOTSTRAP_SERVERS,
                                  KAFKA_CONSUMER_GROUP, KAFKA_MAX_POLL_RECORDS,
                                  KAFKA_TIMEOUT_MS, KAFKA_TOPIC,
                                  WEBAPP_TIMEOUT_SECONDS, WEBAPP_URL)
from src.storage.database import DatabaseService
from src.transformation.cleaner import DataCleaner
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RealtimePipeline:
    """
    Real-time pipeline for Kafka consumption → API prediction → PostgreSQL

    Main functions:
    - load_from_kafka_batch(): Consumes 1,000 Kafka messages
    - load_from_kafka_stream(): Consumes continuous streaming
    - clean_and_preprocess(): Cleans and prepares data
    - predict_batch(): Calls API /predict/batch
    - predict_stream(): Calls API /predict (one by one)
    - save_to_database(): Saves to PostgreSQL
    """

    def __init__(
        self,
        kafka_bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        kafka_topic: str = KAFKA_TOPIC,
        api_url: str = API_URL,
        webapp_url: Optional[str] = WEBAPP_URL,
        db_service: Optional[DatabaseService] = None,
        connect_db: bool = True,
    ):
        """
        Initialize realtime pipeline

        Args:
            kafka_bootstrap_servers: Kafka broker address (default from constants.py → env vars)
            kafka_topic: Kafka topic to consume (default from constants.py → env vars)
            api_url: Fraud detection API URL (default from constants.py → env vars)
            webapp_url: Web application URL for sending results (default from constants.py → env vars)
            db_service: Database service instance (optional, creates new if None)
                       If not provided, DatabaseService() will be created and will load
                       connection config from settings.py (reads DB_SERVER, DB_USER, etc.)
            connect_db: Whether to automatically connect to database on initialization (default: True)
                       Set to False for testing to avoid database connection attempts

        Configuration sources (priority order):
        1. Explicit parameters passed to __init__
        2. constants.py (reads from environment variables)
        3. Hardcoded defaults in constants.py

        Example:
            # Use all defaults (from env vars or constants.py)
            pipeline = RealtimePipeline()

            # Override specific configs
            pipeline = RealtimePipeline(
                kafka_bootstrap_servers='localhost:29092',
                api_url='http://localhost:8000'
            )

            # For testing (no database connection)
            pipeline = RealtimePipeline(connect_db=False)
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.api_url = api_url
        self.webapp_url = webapp_url

        # Database: create new service or use provided one
        # DatabaseService will load connection from settings.py on connect()
        # Settings reads: DB_SERVER, DB_USER, DB_PASSWORD, DB_PORT, DB_NAME
        self.db_service = db_service or DatabaseService()
        if not self.db_service._initialized and connect_db:
            logger.info("Connecting to database (config from settings.py)...")
            self.db_service.connect()

        # Data cleaner
        self.cleaner = DataCleaner()

        # Kafka consumer
        self.consumer: Optional[KafkaConsumer] = None

        # Metrics
        self.metrics = {
            "total_consumed": 0,
            "total_cleaned": 0,
            "total_predicted": 0,
            "total_fraud_detected": 0,
            "total_saved": 0,
            "errors": 0,
        }

        logger.info(f"RealtimePipeline initialized")
        logger.info(f"  Kafka: {kafka_bootstrap_servers}")
        logger.info(f"  Topic: {kafka_topic}")
        logger.info(f"  API: {api_url}")

    def _create_kafka_consumer(
        self, group_id: str = KAFKA_CONSUMER_GROUP
    ) -> KafkaConsumer:
        """
        Create Kafka consumer

        Args:
            group_id: Consumer group ID

        Returns:
            KafkaConsumer instance
        """
        if not KAFKA_AVAILABLE:
            raise ImportError(
                "kafka-python required. Install with: pip install kafka-python"
            )

        try:
            consumer = KafkaConsumer(
                self.kafka_topic,
                bootstrap_servers=self.kafka_bootstrap_servers,
                group_id=group_id,
                auto_offset_reset=KAFKA_AUTO_OFFSET_RESET,
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                consumer_timeout_ms=KAFKA_TIMEOUT_MS,
                max_poll_records=KAFKA_MAX_POLL_RECORDS,
            )
            logger.info(f"Connected to Kafka: {self.kafka_bootstrap_servers}")
            logger.info(f"Consuming from topic: {self.kafka_topic}")
            return consumer
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {str(e)}")
            raise

    def load_from_kafka_batch(
        self, count: int = 1000, timeout_seconds: int = 60
    ) -> pd.DataFrame:
        """
        Load a batch of transactions from Kafka

        Args:
            count: Number of transactions to consume
            timeout_seconds: Max time to wait for messages

        Returns:
            DataFrame with transactions
        """
        logger.info(f"Loading BATCH from Kafka: {count} transactions")

        consumer = self._create_kafka_consumer(group_id="fraud-detection-batch")
        transactions = []
        start_time = time.time()

        try:
            for message in consumer:
                transaction = message.value
                transactions.append(transaction)

                self.metrics["total_consumed"] += 1

                # Log progress every 100 messages
                if len(transactions) % 100 == 0:
                    logger.info(f"   Consumed {len(transactions)}/{count} transactions")

                # Stop when count reached
                if len(transactions) >= count:
                    break

                # Timeout check
                if (time.time() - start_time) > timeout_seconds:
                    logger.warning(f"⚠️  Timeout reached after {timeout_seconds}s")
                    break

        finally:
            consumer.close()

        df = pd.DataFrame(transactions)
        logger.info(f"Loaded {len(df)} transactions from Kafka")

        return df

    def load_from_kafka_stream(
        self, callback: Any, max_messages: Optional[int] = None
    ) -> None:
        """
        Load transactions from Kafka in streaming mode (continuous)

        Args:
            callback: Function to call for each transaction
            max_messages: Max messages to consume (None = infinite)
        """
        logger.info(f"Starting STREAM consumption from Kafka")
        if max_messages:
            logger.info(f"Max messages: {max_messages}")
        else:
            logger.info(f"   Mode: CONTINUOUS (Ctrl+C to stop)")

        consumer = self._create_kafka_consumer(group_id="fraud-detection-stream")
        message_count = 0

        try:
            for message in consumer:
                transaction = message.value

                # Call callback for each transaction
                callback(transaction)

                message_count += 1
                self.metrics["total_consumed"] += 1

                # Log every 50 messages
                if message_count % 50 == 0:
                    logger.info(f"   Processed {message_count} transactions")

                # Stop if max reached
                if max_messages and message_count >= max_messages:
                    logger.info(f"Reached max messages: {max_messages}")
                    break

        except KeyboardInterrupt:
            logger.info("Stream interrupted by user")

        finally:
            consumer.close()
            logger.info(f"Stream completed: {message_count} messages processed")

    def clean_and_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess transactions

        Args:
            df: Raw transactions DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"🧹 Cleaning and preprocessing {len(df)} transactions")

        initial_count = len(df)

        # 1. Remove duplicates
        df = df.drop_duplicates(subset=["transaction_id"], keep="first")

        # 2. Handle missing values (drop rows with NaN in critical features)
        critical_features = ["Time", "amount"] + [f"V{i}" for i in range(1, 29)]
        df = df.dropna(subset=critical_features)

        # 3. Remove outliers in amount (above 99.9 percentile)
        amount_threshold = df["amount"].quantile(0.999)
        df = df[df["amount"] <= amount_threshold]

        # 4. Ensure correct data types
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        for i in range(1, 29):
            df[f"V{i}"] = pd.to_numeric(df[f"V{i}"], errors="coerce")

        # Drop any rows with coercion errors
        df = df.dropna()

        final_count = len(df)
        removed = initial_count - final_count

        logger.info(f"Cleaning completed:")
        logger.info(f"   Initial: {initial_count} transactions")
        logger.info(f"   Final: {final_count} transactions")
        if initial_count > 0:
            logger.info(f"   Removed: {removed} ({removed/initial_count*100:.1f}%)")
        else:
            logger.info(f"   Removed: {removed} (N/A%)")

        self.metrics["total_cleaned"] = final_count

        return df

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make batch predictions via API

        Args:
            df: DataFrame with cleaned transactions

        Returns:
            DataFrame with predictions added
        """
        logger.info(f"Making BATCH predictions via API: {len(df)} transactions")

        # Prepare payload for API (features only)
        features = ["Time", "amount"] + [f"V{i}" for i in range(1, 29)]
        payload = {"transactions": df[features].to_dict(orient="records")}

        try:
            # Call API batch endpoint
            url = f"{self.api_url}/api/v1/batch-predict"
            logger.info(f"   Calling: POST {url}")

            response = requests.post(
                url,
                json=payload,
                timeout=API_TIMEOUT_SECONDS,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                results = response.json()

                # Add predictions to DataFrame
                df["predicted_fraud"] = [r["is_fraud"] for r in results["predictions"]]
                df["fraud_probability"] = [
                    r["fraud_probability"] for r in results["predictions"]
                ]
                df["prediction_timestamp"] = datetime.utcnow().isoformat()

                fraud_count = df["predicted_fraud"].sum()
                logger.info(f"Batch prediction completed")
                logger.info(
                    f"   Frauds detected: {fraud_count}/{len(df)} ({fraud_count/len(df)*100:.1f}%)"
                )

                self.metrics["total_predicted"] = len(df)
                self.metrics["total_fraud_detected"] += fraud_count
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                df["predicted_fraud"] = None
                df["fraud_probability"] = None
                self.metrics["errors"] += 1

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            df["predicted_fraud"] = None
            df["fraud_probability"] = None
            self.metrics["errors"] += 1

        return df

    def predict_stream(self, transaction: Dict) -> Dict:
        """
        Make single transaction prediction via API

        Args:
            transaction: Single transaction dict

        Returns:
            Transaction with prediction added
        """
        # Prepare features
        features = {"Time": transaction["Time"], "amount": transaction["amount"]}
        for i in range(1, 29):
            features[f"V{i}"] = transaction[f"V{i}"]

        try:
            # Call API single prediction endpoint
            url = f"{self.api_url}/api/v1/predict"

            response = requests.post(
                url,
                json=features,
                timeout=5,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                transaction["predicted_fraud"] = result["is_fraud"]
                transaction["fraud_probability"] = result["fraud_probability"]
                transaction["prediction_timestamp"] = datetime.utcnow().isoformat()

                self.metrics["total_predicted"] += 1
                if result["is_fraud"]:
                    self.metrics["total_fraud_detected"] += 1
                    logger.info(
                        f"FRAUD DETECTED: txn {transaction['transaction_id']} "
                        f"(prob: {result['fraud_probability']:.2%})"
                    )
            else:
                logger.error(f"API error: {response.status_code}")
                transaction["predicted_fraud"] = None
                transaction["fraud_probability"] = None
                self.metrics["errors"] += 1

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            transaction["predicted_fraud"] = None
            transaction["fraud_probability"] = None
            self.metrics["errors"] += 1

        return transaction

    def save_to_database(self, df: pd.DataFrame) -> int:
        """
        Save transactions to PostgreSQL

        Args:
            df: DataFrame with transactions and predictions

        Returns:
            Number of rows saved
        """
        logger.info(f"Saving {len(df)} transactions to PostgreSQL")

        try:
            # Convert to list of dicts
            records = df.to_dict(orient="records")

            # Save using DatabaseService
            count = self.db_service.insert_transactions(records)

            logger.info(f"Saved {count} transactions to database")
            self.metrics["total_saved"] = count

            return count

        except Exception as e:
            logger.error(f"Failed to save to database: {str(e)}", exc_info=True)
            self.metrics["errors"] += 1
            return 0

    def send_to_webapp(self, results: List[Dict]) -> bool:
        """
        Send prediction results to web application

        Args:
            results: List of transactions with predictions

        Returns:
            Success status
        """
        if not self.webapp_url:
            logger.warning("Web app URL not configured, skipping")
            return False

        logger.info(f"Sending {len(results)} results to web app")

        try:
            url = f"{self.webapp_url}/api/predictions"
            response = requests.post(
                url,
                json={"predictions": results},
                timeout=WEBAPP_TIMEOUT_SECONDS,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                logger.info(f"Results sent to web app")
                return True
            else:
                logger.error(f"Web app error: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to send to web app: {str(e)}")
            return False

    def execute_batch(self, count: int = 1000) -> Dict[str, Any]:
        """
        Execute complete batch pipeline

        Flow: Kafka → Clean → Predict Batch → Save DB → Send Web App

        Args:
            count: Number of transactions to process

        Returns:
            Execution summary
        """
        logger.info(f"Starting BATCH pipeline: {count} transactions")
        start_time = time.time()

        try:
            # 1. Load from Kafka
            df = self.load_from_kafka_batch(count=count)

            if df.empty:
                logger.warning("No transactions loaded from Kafka")
                return {"status": "error", "message": "No transactions in Kafka"}

            # 2. Clean and preprocess
            df = self.clean_and_preprocess(df)

            if df.empty:
                logger.warning("All transactions filtered during cleaning")
                return {"status": "error", "message": "All transactions filtered"}

            # 3. Predict (batch)
            df = self.predict_batch(df)

            # 4. Save to database
            saved_count = self.save_to_database(df)

            # 5. Send to web app (fraud alerts only)
            fraud_df = df[df["predicted_fraud"] == 1]
            if not fraud_df.empty:
                fraud_results = fraud_df.to_dict(orient="records")
                self.send_to_webapp(fraud_results)

            # Summary
            elapsed = time.time() - start_time

            summary = {
                "status": "success",
                "mode": "batch",
                "consumed": len(df),
                "cleaned": len(df),
                "predicted": len(df),
                "fraud_detected": int(fraud_df.shape[0]) if not fraud_df.empty else 0,
                "saved": saved_count,
                "elapsed_seconds": elapsed,
                "metrics": self.metrics,
            }

            logger.info(f"Batch pipeline completed in {elapsed:.2f}s")
            logger.info(f"  Summary:")
            logger.info(f"      Consumed: {summary['consumed']}")
            logger.info(
                f"      Frauds: {summary['fraud_detected']} ({summary['fraud_detected']/summary['consumed']*100:.1f}%)"
            )
            logger.info(f"      Saved: {summary['saved']}")

            return summary

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Batch pipeline failed after {elapsed:.2f}s: {str(e)}", exc_info=True
            )
            return {"status": "error", "message": str(e), "elapsed_seconds": elapsed}

    def execute_stream(self, interval_seconds: int = 10) -> None:
        """
        Execute streaming pipeline (continuous)

        Flow: Kafka Stream → Clean → Predict Single → Save DB → Send Web App

        Args:
            interval_seconds: Process batch every N seconds
        """
        logger.info(f"🌊 Starting STREAM pipeline (interval: {interval_seconds}s)")

        buffer = []
        last_process_time = [
            time.time()
        ]  # Use list to allow modification in nested function

        def process_transaction(transaction: Dict) -> None:
            """Callback for each Kafka message"""
            buffer.append(transaction)

            # Process buffer every interval_seconds
            if (time.time() - last_process_time[0]) >= interval_seconds:
                self._process_stream_buffer(buffer.copy())
                buffer.clear()
                last_process_time[0] = time.time()

        try:
            # Start streaming consumption
            self.load_from_kafka_stream(callback=process_transaction)

        except KeyboardInterrupt:
            logger.info("Stream interrupted")
            # Process remaining buffer
            if buffer:
                self._process_stream_buffer(buffer)

    def _process_stream_buffer(self, transactions: List[Dict]) -> None:
        """Process accumulated stream buffer"""
        if not transactions:
            return

        logger.info(f"Processing stream buffer: {len(transactions)} transactions")

        try:
            # Convert to DataFrame
            df = pd.DataFrame(transactions)

            # Clean
            df = self.clean_and_preprocess(df)

            # Predict batch
            df = self.predict_batch(df)

            # Save
            self.save_to_database(df)

            # Send frauds to web app
            fraud_df = df[df["predicted_fraud"] == 1]
            if not fraud_df.empty:
                self.send_to_webapp(fraud_df.to_dict(orient="records"))

        except Exception as e:
            logger.error(f"Stream buffer processing failed: {str(e)}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        return self.metrics.copy()


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Realtime Pipeline - Kafka → API → PostgreSQL"
    )
    parser.add_argument(
        "--mode", choices=["batch", "stream"], default="batch", help="Pipeline mode"
    )
    parser.add_argument(
        "--count", type=int, default=1000, help="Number of transactions (batch mode)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Processing interval in seconds (stream mode)",
    )
    parser.add_argument(
        "--kafka", default=KAFKA_BOOTSTRAP_SERVERS, help="Kafka bootstrap servers"
    )
    parser.add_argument("--topic", default=KAFKA_TOPIC, help="Kafka topic")
    parser.add_argument("--api-url", default=API_URL, help="API URL")
    parser.add_argument("--webapp-url", default=WEBAPP_URL, help="Web app URL")

    args = parser.parse_args()

    # Create pipeline
    pipeline = RealtimePipeline(
        kafka_bootstrap_servers=args.kafka,
        kafka_topic=args.topic,
        api_url=args.api_url,
        webapp_url=args.webapp_url,
    )

    try:
        if args.mode == "batch":
            result = pipeline.execute_batch(count=args.count)
            return 0 if result["status"] == "success" else 1
        else:  # stream
            pipeline.execute_stream(interval_seconds=args.interval)
            return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
