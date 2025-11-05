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

import argparse
import json
import os
import sys
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

from src.config.constants import (
    API_TIMEOUT_SECONDS,
    API_URL,
    KAFKA_AUTO_OFFSET_RESET,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_CONSUMER_GROUP,
    KAFKA_MAX_POLL_RECORDS,
    KAFKA_TIMEOUT_MS,
    KAFKA_TOPIC,
    WEBAPP_TIMEOUT_SECONDS,
    WEBAPP_URL,
)
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
        api_username: Optional[str] = None,
        api_password: Optional[str] = None,
        api_token: Optional[str] = None,
    ):
        """
        Initialize the realtime pipeline with JWT authentication.

        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
            kafka_topic: Kafka topic to consume from
            api_url: Fraud detection API URL
            webapp_url: Web application URL for alerts
            db_service: Database service instance
            connect_db: Whether to connect to database
            api_username: Username for API authentication (from env: API_USERNAME)
            api_password: Password for API authentication (from env: API_PASSWORD)
            api_token: Pre-existing JWT token (from env: API_TOKEN)
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.api_url = api_url.rstrip("/")
        self.webapp_url = webapp_url

        # JWT Authentication
        self.api_username = api_username or os.getenv("API_USERNAME")
        self.api_password = api_password or os.getenv("API_PASSWORD")
        self.api_token = api_token or os.getenv("API_TOKEN")

        # Session for connection pooling
        self.session = requests.Session()

        # Token expiry tracking
        self._token_expiry = None
        self._token_issued_at = 0

        self.metrics = {
            "total_consumed": 0,
            "total_cleaned": 0,
            "total_predicted": 0,
            "total_fraud_detected": 0,
            "total_saved": 0,
            "errors": 0,
            "auth_failures": 0,
        }

        # Initialize database service
        if connect_db:
            self.db_service = db_service or DatabaseService()
        else:
            self.db_service = None

        logger.info(
            f"RealtimePipeline initialized: kafka={kafka_bootstrap_servers}, "
            f"topic={kafka_topic}, api={api_url}"
        )
        logger.info(
            f"Auth: {'JWT Token' if self.api_token else 'Username/Password' if self.api_username else 'None'}"
        )
        logger.info(
            f"Webapp URL: {self.webapp_url if self.webapp_url else 'Not configured (fraud alerts disabled)'}"
        )

        # Data cleaner
        self.cleaner = DataCleaner()

        # Kafka consumer
        self.consumer: Optional[KafkaConsumer] = None

        # Authenticate at startup if credentials provided
        if not self.api_token and self.api_username and self.api_password:
            logger.info("Authenticating to API at startup...")
            self._authenticate()

    def _authenticate(self) -> bool:
        """
        Authenticate to API and obtain JWT token

        Returns:
            True if successful
        """
        if not self.api_username or not self.api_password:
            logger.warning("No credentials provided for authentication")
            return False

        try:
            url = f"{self.api_url}/auth/login"

            # OAuth2PasswordRequestForm format (form-data, not JSON)
            payload = {
                "username": self.api_username,
                "password": self.api_password,
            }

            logger.info(f"Authenticating: POST {url}")

            response = self.session.post(
                url, data=payload, timeout=10  # data, not json (OAuth2 form)
            )

            if response.status_code == 200:
                result = response.json()
                self.api_token = result.get("access_token")
                expires_in = result.get("expires_in", 3600)

                # Track expiry (refresh 5 min before)
                self._token_expiry = time.time() + expires_in - 300
                self._token_issued_at = time.time()

                logger.info("Authentication successful")
                logger.info(f"Token expires in: {expires_in}s")

                user_info = result.get("user", {})
                if user_info:
                    logger.info(f"User: {user_info.get('username')}")

                return True
            else:
                logger.error(f"Authentication failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                self.metrics["auth_failures"] += 1
                return False

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}", exc_info=True)
            self.metrics["auth_failures"] += 1
            return False

    def _refresh_token_if_needed(self) -> bool:
        """
        Refresh JWT token if close to expiry

        Returns:
            True if token is valid
        """
        # No token? Try to authenticate
        if not self.api_token:
            logger.warning("No token available, authenticating...")
            return self._authenticate()

        # Check expiry
        if self._token_expiry and time.time() >= self._token_expiry:
            logger.info("Token expiring soon, refreshing...")

            try:
                url = f"{self.api_url}/auth/refresh"

                headers = {
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                }

                response = self.session.post(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    self.api_token = result.get("access_token")
                    expires_in = result.get("expires_in", 3600)
                    self._token_expiry = time.time() + expires_in - 300

                    logger.info("Token refreshed successfully")
                    return True
                else:
                    logger.warning(f"Token refresh failed: {response.status_code}")
                    # Fall back to re-authentication
                    return self._authenticate()

            except Exception as e:
                logger.error(f"Token refresh error: {str(e)}")
                return self._authenticate()

        # Token still valid
        return True

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get headers with JWT Bearer token authorization

        Returns:
            Headers dict with Authorization Bearer token
        """
        headers = {"Content-Type": "application/json"}

        # Refresh token if needed
        self._refresh_token_if_needed()

        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
            logger.debug("Using JWT Bearer token authentication")
        else:
            logger.warning("No JWT token available for authentication")

        return headers

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
                    logger.warning(f"  Timeout reached after {timeout_seconds}s")
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

    def predict_batch(self, df: pd.DataFrame, max_retries: int = 3) -> pd.DataFrame:
        """
        Make batch predictions via API with JWT authentication

        Args:
            df: DataFrame with cleaned transactions
            max_retries: Number of retry attempts

        Returns:
            DataFrame with predictions added
        """
        logger.info(f"Making BATCH predictions via API: {len(df)} transactions")

        # Prepare payload for API (features in correct format)
        # API expects: {transactions: [{transaction_id: str, features: [30 floats]}]}
        # Features order: Time, V1-V28, amount (30 total)
        transactions_payload = []
        for _, row in df.iterrows():
            # Build features array: [Time, V1-V28, amount]
            features = [float(row["Time"])]
            features += [float(row[f"V{i}"]) for i in range(1, 29)]
            features.append(float(row["amount"]))

            transactions_payload.append(
                {
                    "transaction_id": str(
                        row.get("transaction_id", f"txn_{len(transactions_payload)}")
                    ),
                    "features": features,
                }
            )

        payload = {"transactions": transactions_payload}

        url = f"{self.api_url}/api/v1/batch-predict"

        for attempt in range(max_retries):
            try:
                # Get authenticated headers
                headers = self._get_auth_headers()

                logger.info(
                    f"Calling: POST {url} (attempt {attempt + 1}/{max_retries})"
                )

                response = self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=API_TIMEOUT_SECONDS,
                )

                if response.status_code == 200:
                    # Success
                    results = response.json()

                    # Extract predictions and map to DataFrame
                    # API returns: {predictions: [{transaction_id, prediction, fraud_score, confidence, model_version, ...}]}
                    predictions_list = results["predictions"]

                    # Create mapping by transaction_id for safety
                    predictions_map = {p["transaction_id"]: p for p in predictions_list}

                    # Apply predictions to DataFrame
                    df["predicted_fraud"] = df["transaction_id"].apply(
                        lambda tid: bool(
                            predictions_map.get(str(tid), {}).get("prediction", 0)
                        )
                    )
                    df["fraud_probability"] = df["transaction_id"].apply(
                        lambda tid: float(
                            predictions_map.get(str(tid), {}).get("fraud_score", 0.0)
                        )
                    )
                    df["model_version"] = df["transaction_id"].apply(
                        lambda tid: str(
                            predictions_map.get(str(tid), {}).get(
                                "model_version", "unknown"
                            )
                        )
                    )
                    df["prediction_timestamp"] = datetime.utcnow().isoformat()

                    fraud_count = df["predicted_fraud"].sum()
                    logger.info(" Batch prediction completed")
                    logger.info(
                        f"   Frauds detected: {fraud_count}/{len(df)} ({fraud_count/len(df)*100:.1f}%)"
                    )

                    self.metrics["total_predicted"] += len(df)
                    self.metrics["total_fraud_detected"] += int(fraud_count)
                    return df

                elif response.status_code == 401:
                    # Unauthorized - refresh and retry
                    logger.warning("401 Unauthorized, refreshing token...")
                    self.metrics["auth_failures"] += 1

                    if self._authenticate():
                        if attempt < max_retries - 1:
                            continue
                    raise Exception("Authentication failed")

                elif response.status_code == 403:
                    # Forbidden - insufficient permissions
                    logger.error("403 Forbidden - insufficient permissions")
                    logger.error(f"User: {self.api_username}")
                    self.metrics["auth_failures"] += 1
                    break

                elif response.status_code >= 500:
                    # Server error - retry
                    logger.warning(f"Server error {response.status_code}, retrying...")
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)  # Exponential backoff
                        continue
                    raise Exception(f"Server error: {response.status_code}")

                else:
                    # Other client error
                    logger.error(f"API error: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    break

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise

            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                if attempt == max_retries - 1:
                    break
                time.sleep(2**attempt)

        # All attempts failed
        logger.error(f"All {max_retries} attempts failed")
        df["predicted_fraud"] = None
        df["fraud_probability"] = None
        self.metrics["errors"] += 1

        return df

    def predict_stream(self, transaction: Dict) -> Dict:
        """
        Make single transaction prediction via API with JWT authentication

        Args:
            transaction: Single transaction dict

        Returns:
            Transaction with prediction added
        """
        # Prepare features
        features = {"Time": transaction["Time"], "amount": transaction["amount"]}
        for i in range(1, 29):
            features[f"V{i}"] = transaction[f"V{i}"]

        url = f"{self.api_url}/api/v1/predict"

        try:
            # Get authenticated headers
            headers = self._get_auth_headers()

            response = self.session.post(
                url,
                json=features,
                headers=headers,
                timeout=5,
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
                        f"FRAUD DETECTED: txn {transaction.get('transaction_id')} "
                        f"(prob: {result['fraud_probability']:.2%})"
                    )

            elif response.status_code == 401:
                logger.warning("401 Unauthorized, refreshing token...")
                self._authenticate()
                # Retry once
                headers = self._get_auth_headers()
                response = self.session.post(
                    url, json=features, headers=headers, timeout=5
                )
                if response.status_code == 200:
                    result = response.json()
                    transaction["predicted_fraud"] = result["is_fraud"]
                    transaction["fraud_probability"] = result["fraud_probability"]
                else:
                    transaction["predicted_fraud"] = None
                    self.metrics["auth_failures"] += 1

            else:
                logger.error(f"API error: {response.status_code}")
                transaction["predicted_fraud"] = None
                self.metrics["errors"] += 1

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            transaction["predicted_fraud"] = None
            self.metrics["errors"] += 1

        return transaction

    def save_to_database(self, df: pd.DataFrame) -> int:
        """
        Save transactions AND predictions to PostgreSQL

        This method performs TWO insertions:
        1. insert_transactions(): Saves RAW transactions (Time, V1-V28, amount, Class)
        2. insert_predictions(): Saves API predictions (fraud_score, is_fraud_predicted, model_version)

        Args:
            df: DataFrame with Kaggle features + API predictions

        Returns:
            Number of rows saved (transactions count)
        """
        logger.info(f" Saving {len(df)} transactions + predictions to PostgreSQL")

        try:
            # Connect to database if not already connected
            if not self.db_service._initialized:
                self.db_service.connect()

            # Convert DataFrame to list of dicts
            transactions = df.to_dict(orient="records")

            # Save RAW TRANSACTIONS (Time, V1-V28, amount, Class)
            logger.info(
                f"1 Saving {len(transactions)} RAW transactions to 'transactions' table"
            )
            transactions_saved = self.db_service.insert_transactions(transactions)
            logger.info(f" Saved {transactions_saved} raw transactions")

            # Save PREDICTIONS (fraud_score, is_fraud_predicted, model_version)
            # Only save predictions if API returned results
            predictions = []
            for txn in transactions:
                if "fraud_probability" in txn and txn["fraud_probability"] is not None:
                    pred = {
                        "transaction_id": txn["transaction_id"],
                        "fraud_score": float(txn.get("fraud_probability", 0.0)),
                        "is_fraud_predicted": bool(txn.get("predicted_fraud", False)),
                        "model_version": str(txn.get("model_version", "unknown")),
                        "confidence": float(txn.get("fraud_probability", 0.0)),
                        "prediction_time": datetime.utcnow(),
                    }
                    predictions.append(pred)

            if predictions:
                logger.info(
                    f" Saving {len(predictions)} predictions to 'predictions' table"
                )
                predictions_saved = self.db_service.insert_predictions(predictions)
                logger.info(f" Saved {predictions_saved} predictions")
            else:
                logger.warning(" No predictions to save (API may have failed)")

            self.metrics["total_saved"] = transactions_saved

            return transactions_saved

        except Exception as e:
            logger.error(f" Failed to save to database: {str(e)}", exc_info=True)
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
            # Check authentication before starting
            if not self._refresh_token_if_needed():
                return {
                    "status": "error",
                    "message": "API authentication failed",
                    "elapsed_seconds": 0,
                }

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
        logger.info(f"Starting STREAM pipeline (interval: {interval_seconds}s)")

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
    """CLI entry point with JWT authentication support"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Realtime Pipeline - Kafka → API → PostgreSQL with JWT Auth"
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

    # JWT Authentication arguments
    parser.add_argument(
        "--api-username",
        default=os.getenv("API_USERNAME"),
        help="API username for JWT authentication",
    )
    parser.add_argument(
        "--api-password",
        default=os.getenv("API_PASSWORD"),
        help="API password for JWT authentication",
    )
    parser.add_argument(
        "--api-token", default=os.getenv("API_TOKEN"), help="Pre-existing JWT token"
    )

    args = parser.parse_args()

    # Log configuration for debugging
    logger.info("=" * 80)
    logger.info("REALTIME PIPELINE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Count: {args.count}")
    logger.info(f"Kafka: {args.kafka}")
    logger.info(f"Topic: {args.topic}")
    logger.info(f"API URL: {args.api_url}")
    logger.info(f"Webapp URL: {args.webapp_url}")
    logger.info(f"API Username: {args.api_username}")
    logger.info(f"API Password: {'***' if args.api_password else 'Not set'}")
    logger.info(f"API Token: {'***' if args.api_token else 'Not set'}")
    logger.info("=" * 80)

    # Create pipeline with JWT authentication
    pipeline = RealtimePipeline(
        kafka_bootstrap_servers=args.kafka,
        kafka_topic=args.topic,
        api_url=args.api_url,
        webapp_url=args.webapp_url,
        api_username=args.api_username,
        api_password=args.api_password,
        api_token=args.api_token,
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
