"""
Database service - stores transactions and predictions in SQL database
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

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
            connection_string: Database connection URL
                              Falls back to settings if not provided
        """
        self.connection_string = connection_string
        self.engine = None
        self.session = None
        self._initialized = False

    def connect(self) -> None:
        """Establish database connection"""
        try:
            from sqlalchemy import create_engine
            
            if not self.connection_string:
                from ..config.settings import settings
                self.connection_string = settings.database_url

            self.engine = create_engine(
                self.connection_string,
                pool_size=20,
                max_overflow=40,
                pool_recycle=3600,
                echo=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            self._initialized = True
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def disconnect(self) -> None:
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self._initialized = False
            logger.info("Database connection closed")

    def insert_transactions(self, transactions: List[Dict]) -> int:
        """
        Insert transaction records into database
        
        Args:
            transactions: List of transaction dictionaries
        
        Returns:
            Number of rows inserted
        """
        if not self._initialized:
            self.connect()

        try:
            from sqlalchemy import text
            
            # Assuming table name is 'transactions'
            insert_query = text("""
                INSERT INTO transactions (
                    transaction_id, customer_id, merchant_id,
                    amount, currency, transaction_time,
                    customer_zip, merchant_zip,
                    customer_country, merchant_country,
                    device_id, session_id, ip_address,
                    mcc, transaction_type,
                    is_fraud, is_disputed,
                    source_system, ingestion_timestamp
                ) VALUES (
                    :transaction_id, :customer_id, :merchant_id,
                    :amount, :currency, :transaction_time,
                    :customer_zip, :merchant_zip,
                    :customer_country, :merchant_country,
                    :device_id, :session_id, :ip_address,
                    :mcc, :transaction_type,
                    :is_fraud, :is_disputed,
                    :source_system, :ingestion_timestamp
                )
            """)

            with self.engine.connect() as conn:
                for transaction in transactions:
                    # Add ingestion timestamp if not present
                    if 'ingestion_timestamp' not in transaction or transaction['ingestion_timestamp'] is None:
                        transaction['ingestion_timestamp'] = datetime.utcnow()
                    
                    conn.execute(insert_query, transaction)
                conn.commit()

            logger.info(f"Inserted {len(transactions)} transactions into database")
            return len(transactions)

        except Exception as e:
            logger.error(f"Failed to insert transactions: {str(e)}")
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
            
            insert_query = text("""
                INSERT INTO predictions (
                    transaction_id, fraud_score, is_fraud_predicted,
                    model_version, prediction_time, confidence
                ) VALUES (
                    :transaction_id, :fraud_score, :is_fraud_predicted,
                    :model_version, :prediction_time, :confidence
                )
            """)

            with self.engine.connect() as conn:
                for pred in predictions:
                    # Add prediction timestamp if not present
                    if 'prediction_time' not in pred or pred['prediction_time'] is None:
                        pred['prediction_time'] = datetime.utcnow()
                    
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
            
            query = text("SELECT * FROM transactions ORDER BY transaction_time DESC LIMIT :limit OFFSET :offset")
            
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
                result = conn.execute(text("SELECT COUNT(*) as count FROM transactions"))
                transaction_count = result.scalar()
                
                # Fraud count
                result = conn.execute(text("SELECT COUNT(*) as count FROM transactions WHERE is_fraud = 1"))
                fraud_count = result.scalar()
                
                # Average amount
                result = conn.execute(text("SELECT AVG(amount) as avg_amount FROM transactions"))
                avg_amount = result.scalar()

            return {
                "total_transactions": transaction_count,
                "total_frauds": fraud_count,
                "fraud_rate": fraud_count / transaction_count if transaction_count > 0 else 0,
                "average_amount": avg_amount
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}
