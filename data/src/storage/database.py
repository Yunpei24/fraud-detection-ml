"""
Database service - stores transactions and predictions in SQL database

Configuration is loaded from:
1. Explicit connection_string parameter (if provided)
2. Environment variables via settings.py (fallback)
3. Settings reads from: .env file or OS environment variables

Environment variables used:
- DB_SERVER (default: localhost)
- DB_NAME (default: fraud_db)
- DB_USER (default: postgres)
- DB_PASSWORD (default: postgres)
- DB_PORT (default: 5432)

Connection URL format: postgresql://user:password@host:port/database
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Service for persisting transaction data and predictions to database
    Supports SQL Server, PostgreSQL with SQLAlchemy
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database service

        Args:
            connection_string: Database connection URL (optional)
                              Format: postgresql://user:password@host:port/database
                              If not provided, will be loaded from settings.py on connect()

        Example:
            # Option 1: Explicit connection string
            db = DatabaseService("postgresql://user:pass@localhost:5432/frauddb")

            # Option 2: Load from environment (via settings.py)
            db = DatabaseService()  # Reads DB_SERVER, DB_USER, DB_PASSWORD, etc.

            # Option 3: Inject via pipeline
            from src.config.settings import settings
            db = DatabaseService(settings.database_url)
        """
        self.connection_string = connection_string
        self.engine = None
        self.session = None
        self._initialized = False

        # Log configuration source
        if connection_string:
            logger.debug("DatabaseService initialized with explicit connection string")
        else:
            logger.debug(
                "DatabaseService will load connection from settings on connect()"
            )

    def connect(self) -> None:
        """
        Establish database connection

        Connection string resolution order:
        1. Use explicit connection_string from __init__ (if provided)
        2. Check DATABASE_URL environment variable
        3. Load from settings.database_url (reads DB_* environment variables)

        Environment variables (loaded by settings.py):
        - DATABASE_URL: Full connection string (highest priority)
        - DB_HOST (default: localhost)
        - DB_NAME (default: fraud_db)
        - DB_USER (default: postgres)
        - DB_PASSWORD (default: postgres)
        - DB_PORT (default: 5432)
        """
        try:
            import os
            from sqlalchemy import create_engine

            # Load connection string from settings if not provided
            if not self.connection_string:
                # Priority 1: Check DATABASE_URL environment variable
                database_url = os.getenv("DATABASE_URL")
                if database_url:
                    self.connection_string = database_url
                    logger.info(
                        f"Using DATABASE_URL from environment: {database_url.split('@')[1] if '@' in database_url else 'masked'}"
                    )
                else:
                    # Priority 2: Load from settings (reads DB_* env vars)
                    from ..config.settings import settings

                    self.connection_string = settings.database.url
                    logger.info(
                        f"Loading database config from settings: {settings.database.host}:{settings.database.port}/{settings.database.name}"
                    )
            else:
                logger.info("Using explicit database connection string")

            self.engine = create_engine(
                self.connection_string,
                pool_size=20,
                max_overflow=40,
                pool_recycle=3600,
                echo=False,
            )

            # Test connection
            with self.engine.connect() as conn:
                from sqlalchemy import text

                conn.execute(text("SELECT 1"))

            self._initialized = True
            logger.info(" Database connection established")
        except Exception as e:
            logger.error(f" Failed to connect to database: {str(e)}")
            raise

    def disconnect(self) -> None:
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self._initialized = False
            logger.info("Database connection closed")

    def insert_transactions(self, transactions: List[Dict]) -> int:
        """
        Insert Kaggle format transaction records into database

        This method stores RAW transactions from Kafka (transaction_simulator output)
        into the transactions table. Format: Time, V1-V28, amount, Class.

        Args:
            transactions: List of transaction dictionaries with Kaggle format
                         Required: transaction_id, Time, V1-V28, amount, Class
                         Optional: source, timestamp

        Returns:
            Number of rows inserted
        """
        if not self._initialized:
            self.connect()

        try:
            from sqlalchemy import text

            # Build INSERT query with all V1-V28 columns
            v_columns = ", ".join([f"v{i}" for i in range(1, 29)])
            v_values = ", ".join([f":v{i}" for i in range(1, 29)])

            insert_query = text(
                f"""
                INSERT INTO transactions (
                    transaction_id, time, {v_columns}, amount, class,
                    source, timestamp, ingestion_timestamp
                ) VALUES (
                    :transaction_id, :time, {v_values}, :amount, :class,
                    :source, :timestamp, :ingestion_timestamp
                )
            """
            )

            with self.engine.connect() as conn:
                for transaction in transactions:
                    # Add default metadata if not present
                    if "source" not in transaction:
                        transaction["source"] = "simulator"
                    if "timestamp" not in transaction:
                        transaction["timestamp"] = datetime.utcnow()
                    if "ingestion_timestamp" not in transaction:
                        transaction["ingestion_timestamp"] = datetime.utcnow()

                    # Ensure all V1-V28 are present (set to 0 if missing)
                    for i in range(1, 29):
                        v_key = f"V{i}"
                        if v_key not in transaction:
                            transaction[v_key] = 0.0
                        # Normalize key to lowercase for SQLAlchemy
                        transaction[f"v{i}"] = transaction.pop(
                            v_key, transaction.get(f"v{i}", 0.0)
                        )

                    # Normalize Time and Class keys
                    if "Time" in transaction:
                        transaction["time"] = transaction.pop("Time")
                    if "Class" in transaction:
                        transaction["class"] = transaction.pop("Class")

                    conn.execute(insert_query, transaction)
                conn.commit()

            logger.info(f" Inserted {len(transactions)} transactions into database")
            return len(transactions)

        except Exception as e:
            logger.error(f" Failed to insert transactions: {str(e)}")
            logger.error(
                f"Sample transaction: {transactions[0] if transactions else 'empty'}"
            )
            raise

    def insert_predictions(self, predictions: List[Dict]) -> int:
        """
        Insert prediction records into database

        Args:
            predictions: List of prediction dictionaries
                        Should contain: transaction_id, fraud_score, is_fraud_predicted, model_version, etc

        Returns:
            Number of rows inserted
        """
        if not self._initialized:
            self.connect()

        try:
            from sqlalchemy import text

            insert_query = text(
                """
                INSERT INTO predictions (
                    transaction_id, fraud_score, is_fraud_predicted,
                    model_version, prediction_time, confidence
                ) VALUES (
                    :transaction_id, :fraud_score, :is_fraud_predicted,
                    :model_version, :prediction_time, :confidence
                )
            """
            )

            with self.engine.connect() as conn:
                for pred in predictions:
                    # Add prediction timestamp if not present
                    if "prediction_time" not in pred or pred["prediction_time"] is None:
                        pred["prediction_time"] = datetime.utcnow()

                    conn.execute(insert_query, pred)
                conn.commit()

            logger.info(f"Inserted {len(predictions)} predictions into database")
            return len(predictions)

        except Exception as e:
            logger.error(f"Failed to insert predictions: {str(e)}")
            raise

    def query_transactions(self, limit: int = 1000, offset: int = 0) -> List[Dict]:
        """
        Query transaction records

        Args:
            limit: Maximum number of records
            offset: Number of records to skip

        Returns:
            List of transaction records
        """
        if not self._initialized:
            self.connect()

        try:
            from sqlalchemy import text

            query = text(
                "SELECT * FROM transactions ORDER BY time DESC LIMIT :limit OFFSET :offset"
            )

            with self.engine.connect() as conn:
                result = conn.execute(query, {"limit": limit, "offset": offset})
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result]

            logger.info(f"Retrieved {len(rows)} transactions from database")
            return rows

        except Exception as e:
            logger.error(f"Failed to query transactions: {str(e)}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dictionary with statistics
        """
        if not self._initialized:
            self.connect()

        try:
            from sqlalchemy import text

            with self.engine.connect() as conn:
                # Transaction count
                result = conn.execute(
                    text("SELECT COUNT(*) as count FROM transactions")
                )
                transaction_count = result.scalar()

                # Fraud count
                result = conn.execute(
                    text(
                        "SELECT COUNT(*) as count FROM transactions WHERE is_fraud = 1"
                    )
                )
                fraud_count = result.scalar()

                # Average amount
                result = conn.execute(
                    text("SELECT AVG(amount) as avg_amount FROM transactions")
                )
                avg_amount = result.scalar()

            return {
                "total_transactions": transaction_count,
                "total_frauds": fraud_count,
                "fraud_rate": (
                    fraud_count / transaction_count if transaction_count > 0 else 0
                ),
                "average_amount": avg_amount,
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}
