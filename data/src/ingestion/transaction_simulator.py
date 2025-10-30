"""
Transaction Simulator - Generates transactions with PCA features (v1-v28) + amount + time

Simulates the Kaggle Credit Card Fraud Detection dataset:
- 28 PCA features (V1-V28) - principal components
- amount - transaction amount
- Time - seconds since first transaction
- Class - 0 (legitimate) or 1 (fraudulent)

Usage:
    # Batch of 1000 transactions
    python -m src.ingestion.transaction_simulator --mode batch --count 1000
    
    # Continuous streaming (10 txn/sec)
    python -m src.ingestion.transaction_simulator --mode stream --rate 10
"""

import json
import logging
import random
import time
import uuid
from datetime import datetime
from typing import Dict, Optional

import numpy as np

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("⚠️  kafka-python not installed. Install with: pip install kafka-python")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TransactionSimulator:
    """
    Génère des transactions synthétiques basées sur le dataset Kaggle Credit Card Fraud

    Features générées:
    - V1-V28: PCA features (distributions normales avec patterns spécifiques)
    - Time: Secondes écoulées depuis première transaction
    - amount: Montant de la transaction (distribution log-normale)
    - Class: 0 (légitime) ou 1 (fraude)

    Patterns de fraude réalistes:
    - Montants élevés (> $1000)
    - Patterns PCA anormaux (outliers dans V1-V28)
    - Timing suspect (transactions multiples rapides)
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:29092",
        topic: str = "fraud-detection-transactions",
        fraud_rate: float = 0.05,
    ):
        """
        Initialize transaction simulator

        Args:
            bootstrap_servers: Kafka broker address
            topic: Kafka topic name
            fraud_rate: Proportion of fraudulent transactions (0.0-1.0)
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.fraud_rate = fraud_rate
        self.producer: Optional[KafkaProducer] = None

        # Counters
        self.transaction_count = 0
        self.fraud_count = 0
        self.start_time = time.time()

        # PCA feature parameters (approximation from Kaggle dataset statistics)
        # Legitimate transactions: mean ≈ 0, std ≈ 1 for most V features
        self.legitimate_params = {
            f"V{i}": {"mean": 0.0, "std": random.uniform(0.8, 1.2)}
            for i in range(1, 29)
        }

        # Fraudulent transactions: shifted distributions for some V features
        self.fraud_params = {
            f"V{i}": {"mean": random.uniform(-2, 2), "std": random.uniform(1.0, 2.5)}
            for i in range(1, 29)
        }

        # Specific fraud patterns for key features (based on Kaggle insights)
        # V4, V11, V12, V14 are often discriminative for fraud
        self.fraud_params["V4"] = {"mean": -3.5, "std": 2.0}
        self.fraud_params["V11"] = {"mean": 2.5, "std": 1.8}
        self.fraud_params["V12"] = {"mean": -4.0, "std": 2.2}
        self.fraud_params["V14"] = {"mean": -5.0, "std": 2.5}

        if not KAFKA_AVAILABLE:
            raise ImportError(
                "kafka-python is required. Install with: pip install kafka-python"
            )

    def connect(self) -> None:
        """Connect to Kafka broker"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
                max_in_flight_requests_per_connection=5,
            )
            logger.info(f"Connected to Kafka: {self.bootstrap_servers}")
            logger.info(f"Publishing to topic: {self.topic}")
            logger.info(f"Fraud rate: {self.fraud_rate:.1%}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            raise

    def disconnect(self) -> None:
        """Disconnect from Kafka"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            self.producer = None
            logger.info("Disconnected from Kafka")

    def generate_pca_features(self, is_fraud: bool) -> Dict[str, float]:
        """
        Generate PCA features V1-V28

        Args:
            is_fraud: Whether this is a fraudulent transaction

        Returns:
            Dict with V1-V28 features
        """
        params = self.fraud_params if is_fraud else self.legitimate_params

        features = {}
        for i in range(1, 29):
            feature_name = f"V{i}"
            mean = params[feature_name]["mean"]
            std = params[feature_name]["std"]

            # Generate value from normal distribution
            value = np.random.normal(mean, std)

            # Add some outliers for fraud (10% chance)
            if is_fraud and random.random() < 0.1:
                value += random.choice([-5, 5]) * random.uniform(1, 3)

            features[feature_name] = round(float(value), 6)

        return features

    def generate_legitimate_transaction(self) -> Dict:
        """
        Generate a legitimate transaction

        Returns:
            Transaction dict with 28 PCA features + amount + Time + Class
        """
        # Time: seconds elapsed since simulation start
        elapsed_time = round(time.time() - self.start_time, 2)

        # amount: log-normal distribution (most transactions are small)
        # Mean ≈ $88, but can go up to several thousand
        amount = round(np.random.lognormal(mean=3.5, sigma=1.5), 2)
        amount = min(amount, 5000.0)  # Cap at $5000 for legitimate
        amount = max(amount, 1.0)  # Min $1

        # Generate PCA features
        pca_features = self.generate_pca_features(is_fraud=False)

        # Build transaction
        transaction = {
            "Time": elapsed_time,
            "amount": amount,
            "Class": 0,
            **pca_features,  # V1-V28
            # Metadata for tracking
            "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "simulator",
        }

        return transaction

    def generate_fraudulent_transaction(self) -> Dict:
        """
        Generate a fraudulent transaction with anomalous patterns

        Returns:
            Transaction dict with fraud indicators
        """
        # Time: seconds elapsed
        elapsed_time = round(time.time() - self.start_time, 2)

        # amount: fraudulent transactions tend to be higher or very low
        fraud_amount_type = random.choice(["high", "low", "normal"])

        if fraud_amount_type == "high":
            # High-value fraud ($1000-$10000)
            amount = round(random.uniform(1000, 10000), 2)
        elif fraud_amount_type == "low":
            # Low-value fraud testing ($0.01-$10)
            amount = round(random.uniform(0.01, 10), 2)
        else:
            # Normal range but suspicious
            amount = round(np.random.lognormal(mean=4.0, sigma=1.8), 2)

        # Generate PCA features with fraud patterns
        pca_features = self.generate_pca_features(is_fraud=True)

        # Build transaction
        transaction = {
            "Time": elapsed_time,
            "amount": amount,
            "Class": 1,
            **pca_features,  # V1-V28
            # Metadata
            "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "simulator",
            "fraud_amount_type": fraud_amount_type,
        }

        return transaction

    def generate_transaction(self) -> Dict:
        """
        Generate a transaction (fraudulent or legitimate based on fraud_rate)

        Returns:
            Transaction dict
        """
        if random.random() < self.fraud_rate:
            self.fraud_count += 1
            return self.generate_fraudulent_transaction()
        else:
            return self.generate_legitimate_transaction()

    def send_transaction(self, transaction: Dict) -> bool:
        """
        Send transaction to Kafka

        Args:
            transaction: Transaction dict to send

        Returns:
            Success status
        """
        try:
            future = self.producer.send(self.topic, value=transaction)
            record_metadata = future.get(timeout=10)

            self.transaction_count += 1

            # Log every 100 transactions
            if self.transaction_count % 100 == 0:
                fraud_pct = (self.fraud_count / self.transaction_count) * 100
                elapsed = time.time() - self.start_time
                rate = self.transaction_count / elapsed if elapsed > 0 else 0

                logger.info(
                    f"Sent {self.transaction_count} txn "
                    f"({self.fraud_count} fraud, {fraud_pct:.1f}%) "
                    f"| Rate: {rate:.1f} txn/s "
                    f"| Partition: {record_metadata.partition}"
                )

            return True

        except KafkaError as e:
            logger.error(f"Failed to send transaction: {str(e)}")
            return False

    def simulate_batch(self, count: int = 1000) -> Dict:
        """
        Simulate a batch of transactions

        Args:
            count: Number of transactions to generate

        Returns:
            Summary statistics
        """
        logger.info(f" Starting BATCH simulation: {count} transactions")
        logger.info(f"   Fraud rate: {self.fraud_rate:.1%}")

        batch_start = time.time()
        success_count = 0

        for i in range(count):
            transaction = self.generate_transaction()
            if self.send_transaction(transaction):
                success_count += 1

            # Small delay to avoid overwhelming Kafka
            if i % 50 == 0 and i > 0:
                time.sleep(0.05)

        elapsed = time.time() - batch_start
        rate = count / elapsed if elapsed > 0 else 0

        summary = {
            "mode": "batch",
            "total_transactions": count,
            "successful": success_count,
            "failed": count - success_count,
            "fraud_count": self.fraud_count,
            "fraud_rate": self.fraud_count / count if count > 0 else 0,
            "elapsed_seconds": elapsed,
            "transactions_per_second": rate,
        }

        logger.info(f"Batch simulation completed:")
        logger.info(f"   Total: {summary['total_transactions']}")
        logger.info(
            f"   Fraudulent: {summary['fraud_count']} ({summary['fraud_rate']:.1%})"
        )
        logger.info(f"   Success rate: {summary['successful']}/{count}")
        logger.info(f"    Duration: {elapsed:.2f}s")
        logger.info(f"    Rate: {rate:.1f} txn/sec")

        return summary

    def simulate_stream(
        self,
        transactions_per_second: float = 10.0,
        duration_seconds: Optional[int] = None,
    ) -> None:
        """
        Simulate continuous transaction stream

        Args:
            transactions_per_second: Target rate
            duration_seconds: How long to run (None = infinite)
        """
        logger.info(f"🌊 Starting STREAM simulation: {transactions_per_second} txn/sec")
        if duration_seconds:
            logger.info(f"   Duration: {duration_seconds}s")
        else:
            logger.info(f"   Duration: INFINITE (Ctrl+C to stop)")

        delay = 1.0 / transactions_per_second
        stream_start = time.time()

        try:
            while True:
                transaction = self.generate_transaction()
                self.send_transaction(transaction)

                time.sleep(delay)

                # Check duration
                if (
                    duration_seconds
                    and (time.time() - stream_start) >= duration_seconds
                ):
                    logger.info(f"Stream simulation completed ({duration_seconds}s)")
                    break

        except KeyboardInterrupt:
            logger.info(" Stream simulation interrupted by user")

        finally:
            elapsed = time.time() - stream_start
            rate = self.transaction_count / elapsed if elapsed > 0 else 0

            logger.info(f" Stream summary:")
            logger.info(f"    Total: {self.transaction_count} transactions")
            logger.info(
                f"    Fraudulent: {self.fraud_count} ({self.fraud_count/self.transaction_count:.1%})"
            )
            logger.info(f"     Duration: {elapsed:.1f}s")
            logger.info(f"    Avg rate: {rate:.1f} txn/sec")


def main():
    """CLI entry point for transaction simulator"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Transaction Simulator - Generates PCA features V1-V28 + amount + Time"
    )
    parser.add_argument(
        "--kafka", default="localhost:29092", help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--topic", default="fraud-detection-transactions", help="Kafka topic"
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.05,
        help="Fraud rate (0.0-1.0), default 0.05 (5%%)",
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "stream"],
        default="batch",
        help="Simulation mode: batch (fixed count) or stream (continuous)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of transactions (batch mode only)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Transactions per second (stream mode only)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Duration in seconds (stream mode only, default: infinite)",
    )

    args = parser.parse_args()

    # Create simulator
    simulator = TransactionSimulator(
        bootstrap_servers=args.kafka, topic=args.topic, fraud_rate=args.fraud_rate
    )

    try:
        # Connect to Kafka
        simulator.connect()

        # Run simulation based on mode
        if args.mode == "batch":
            simulator.simulate_batch(count=args.count)
        else:  # stream
            simulator.simulate_stream(
                transactions_per_second=args.rate, duration_seconds=args.duration
            )

    except Exception as e:
        logger.error(f" Simulation failed: {str(e)}", exc_info=True)
        return 1

    finally:
        simulator.disconnect()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
