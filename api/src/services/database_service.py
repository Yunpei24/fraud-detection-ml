"""
Database service for storing predictions and audit logs.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from ..config import get_logger, settings
from ..utils import DatabaseException

# Try to import psycopg2 for test compatibility
try:
    import psycopg2
    from psycopg2 import pool

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logger = get_logger(__name__)
Base = declarative_base()


# Exception classes
class DatabaseConnectionError(DatabaseException):
    """Exception raised when database connection fails."""

    pass


class QueryExecutionError(DatabaseException):
    """Exception raised when query execution fails."""

    pass


class TransactionError(DatabaseException):
    """Exception raised when transaction operations fail."""

    pass


class PredictionLog(Base):
    """Model for prediction logs."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), unique=True, nullable=False, index=True)
    prediction = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    fraud_score = Column(Float, nullable=False)
    risk_level = Column(String(20))
    processing_time = Column(Float)
    model_version = Column(String(50))
    features = Column(Text)  # JSON string
    explanation = Column(Text)  # JSON string
    request_metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AuditLog(Base):
    """Model for audit logs."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), nullable=False, index=True)
    action = Column(String(50), nullable=False)
    user_id = Column(String(100))
    ip_address = Column(String(50))
    details = Column(Text)  # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


class AnalystLabel(Base):
    """Model for analyst feedback labels."""

    __tablename__ = "analyst_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), nullable=False, index=True)
    predicted_label = Column(Integer)
    analyst_label = Column(Integer, nullable=False)
    analyst_id = Column(String(100), nullable=False)
    confidence = Column(Float)
    notes = Column(Text)
    labeled_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DatabaseService:
    """Service for database operations."""

    def __init__(self, settings=None, database_url: Optional[str] = None):
        """
        Initialize database service.

        Args:
            settings: Database settings object (for tests)
            database_url: Database connection URL (optional)
        """
        # Set logger first, before any operations that might fail
        self.logger = logger

        from ..config import settings as global_settings

        if settings:
            # Test mode - use psycopg2 with settings
            self.settings = settings
            self.connection_pool = None
            self.connection_string = self._build_connection_string()
            self.database_url = None
            self.engine = None
            self.SessionLocal = None
        else:
            # Production mode - use SQLAlchemy
            self.database_url = database_url or global_settings.database_url
            self.settings = None
            self.connection_pool = None
            self.connection_string = None
            self.engine = None
            self.SessionLocal = None
            self._initialize_engine()

    def _initialize_engine(self):
        """Initialize SQLAlchemy engine and session factory."""
        try:
            self.engine = create_engine(
                self.database_url, pool_pre_ping=True, pool_size=10, max_overflow=20
            )
            self.SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )
            self.logger.info("Database engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize database engine: {e}")
            raise DatabaseException(
                "Failed to initialize database", details={"error": str(e)}
            )

    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created")
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise DatabaseException(
                "Failed to create database tables", details={"error": str(e)}
            )

    def get_session(self) -> Session:
        """
        Get database session.

        Returns:
            SQLAlchemy session
        """
        return self.SessionLocal()

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from settings."""
        return (
            f"host={self.settings.db_host} "
            f"port={self.settings.db_port} "
            f"dbname={self.settings.db_name} "
            f"user={self.settings.db_user} "
            f"password={self.settings.db_password}"
        )

    def connect(self):
        """Connect to database (psycopg2 mode)."""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 not available")

        if self.settings:
            if self.connection_pool is None:
                # For tests, just return a mock connection
                # In real implementation, this would create a connection pool
                try:
                    self.connection_pool = psycopg2.connect(self.connection_string)
                except Exception as e:
                    self.logger.error(f"Database connection failed: {e}")
                    raise DatabaseConnectionError(f"Failed to connect to database: {e}")
            return self.connection_pool
        else:
            # SQLAlchemy mode - return session
            return self.get_session()

    async def save_prediction(
        self, transaction_id: str, prediction: Dict[str, Any]
    ) -> bool:
        """
        Save prediction to database.

        Args:
            transaction_id: Transaction identifier
            prediction: Prediction data

        Returns:
            True if saved successfully
        """
        session = self.get_session()
        try:
            import json

            log = PredictionLog(
                transaction_id=transaction_id,
                prediction=prediction.get("prediction"),
                confidence=prediction.get("confidence"),
                fraud_score=prediction.get("fraud_score"),
                risk_level=prediction.get("risk_level"),
                processing_time=prediction.get("processing_time"),
                model_version=prediction.get("model_version"),
                features=json.dumps(prediction.get("features", [])),
                explanation=json.dumps(prediction.get("explanation", {})),
                request_metadata=json.dumps(prediction.get("metadata", {})),
            )

            session.add(log)
            session.commit()

            self.logger.info(f"Saved prediction for transaction {transaction_id}")
            return True

        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Failed to save prediction: {e}")
            raise DatabaseException(
                "Failed to save prediction",
                details={"transaction_id": transaction_id, "error": str(e)},
            )
        finally:
            session.close()

    async def save_audit_log(
        self,
        transaction_id: str,
        action: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save audit log entry.

        Args:
            transaction_id: Transaction identifier
            action: Action performed
            user_id: User identifier (optional)
            ip_address: IP address (optional)
            details: Additional details (optional)

        Returns:
            True if saved successfully
        """
        session = self.get_session()
        try:
            import json

            log = AuditLog(
                transaction_id=transaction_id,
                action=action,
                user_id=user_id,
                ip_address=ip_address,
                details=json.dumps(details or {}),
            )

            session.add(log)
            session.commit()

            self.logger.debug(f"Saved audit log for transaction {transaction_id}")
            return True

        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Failed to save audit log: {e}")
            return False
        finally:
            session.close()

    async def save_analyst_label(
        self,
        transaction_id: str,
        predicted_label: int,
        analyst_label: int,
        analyst_id: str,
        confidence: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Save analyst feedback label.

        Args:
            transaction_id: Transaction identifier
            predicted_label: Model's prediction
            analyst_label: Analyst's label
            analyst_id: Analyst identifier
            confidence: Analyst confidence (optional)
            notes: Additional notes (optional)

        Returns:
            True if saved successfully
        """
        session = self.get_session()
        try:
            label = AnalystLabel(
                transaction_id=transaction_id,
                predicted_label=predicted_label,
                analyst_label=analyst_label,
                analyst_id=analyst_id,
                confidence=confidence,
                notes=notes,
            )

            session.add(label)
            session.commit()

            self.logger.info(
                f"Saved analyst label for transaction {transaction_id} "
                f"by analyst {analyst_id}"
            )
            return True

        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Failed to save analyst label: {e}")
            raise DatabaseException(
                "Failed to save analyst label",
                details={"transaction_id": transaction_id, "error": str(e)},
            )
        finally:
            session.close()

    async def get_analyst_labels(
        self,
        transaction_id: Optional[str] = None,
        analyst_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get analyst labels with optional filtering.

        Args:
            transaction_id: Filter by transaction ID
            analyst_id: Filter by analyst ID
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of analyst label records
        """
        session = self.get_session()
        try:
            query = session.query(AnalystLabel)

            # Apply filters
            if transaction_id:
                query = query.filter(AnalystLabel.transaction_id == transaction_id)
            if analyst_id:
                query = query.filter(AnalystLabel.analyst_id == analyst_id)

            # Apply ordering and pagination
            query = (
                query.order_by(AnalystLabel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )

            labels = query.all()

            result = []
            for label in labels:
                result.append(
                    {
                        "id": label.id,
                        "transaction_id": label.transaction_id,
                        "predicted_label": label.predicted_label,
                        "analyst_label": label.analyst_label,
                        "analyst_id": label.analyst_id,
                        "confidence": label.confidence,
                        "notes": label.notes,
                        "created_at": label.created_at.isoformat(),
                    }
                )

            self.logger.debug(f"Retrieved {len(result)} analyst label records")
            return result

        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get analyst labels: {e}")
            return []
        finally:
            session.close()

    async def get_prediction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get prediction by transaction ID.

        Args:
            transaction_id: Transaction identifier

        Returns:
            Prediction data or None if not found
        """
        session = self.get_session()
        try:
            import json

            log = (
                session.query(PredictionLog)
                .filter_by(transaction_id=transaction_id)
                .first()
            )

            if not log:
                return None

            return {
                "transaction_id": log.transaction_id,
                "prediction": log.prediction,
                "confidence": log.confidence,
                "fraud_score": log.fraud_score,
                "risk_level": log.risk_level,
                "processing_time": log.processing_time,
                "model_version": log.model_version,
                "created_at": log.created_at.isoformat(),
            }

        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get prediction: {e}")
            return None
        finally:
            session.close()

    async def get_audit_logs(
        self,
        transaction_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get audit logs with optional filtering.

        Args:
            transaction_id: Filter by transaction ID
            user_id: Filter by user ID
            action: Filter by action type
            limit: Maximum number of records to return
            offset: Number of records to skip
            start_date: Filter logs after this date
            end_date: Filter logs before this date

        Returns:
            List of audit log records
        """
        session = self.get_session()
        try:
            import json

            query = session.query(AuditLog)

            # Apply filters
            if transaction_id:
                query = query.filter(AuditLog.transaction_id == transaction_id)
            if user_id:
                query = query.filter(AuditLog.user_id == user_id)
            if action:
                query = query.filter(AuditLog.action == action)
            if start_date:
                query = query.filter(AuditLog.timestamp >= start_date)
            if end_date:
                query = query.filter(AuditLog.timestamp <= end_date)

            # Apply ordering and pagination
            query = (
                query.order_by(AuditLog.timestamp.desc()).limit(limit).offset(offset)
            )

            logs = query.all()

            result = []
            for log in logs:
                result.append(
                    {
                        "id": log.id,
                        "transaction_id": log.transaction_id,
                        "action": log.action,
                        "user_id": log.user_id,
                        "ip_address": log.ip_address,
                        "details": json.loads(log.details) if log.details else {},
                        "timestamp": log.timestamp.isoformat(),
                    }
                )

            self.logger.debug(f"Retrieved {len(result)} audit log records")
            return result

        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get audit logs: {e}")
            return []
        finally:
            session.close()

    async def get_audit_log_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get audit log summary statistics.

        Args:
            days: Number of days to look back

        Returns:
            Summary statistics
        """
        session = self.get_session()
        try:
            from sqlalchemy import func

            # Calculate date threshold
            threshold_date = datetime.utcnow() - timedelta(days=days)

            # Get total count
            total_count = (
                session.query(func.count(AuditLog.id))
                .filter(AuditLog.timestamp >= threshold_date)
                .scalar()
            )

            # Get action breakdown
            action_counts = (
                session.query(AuditLog.action, func.count(AuditLog.id))
                .filter(AuditLog.timestamp >= threshold_date)
                .group_by(AuditLog.action)
                .all()
            )

            # Get daily activity
            daily_activity = (
                session.query(func.date(AuditLog.timestamp), func.count(AuditLog.id))
                .filter(AuditLog.timestamp >= threshold_date)
                .group_by(func.date(AuditLog.timestamp))
                .all()
            )

            return {
                "total_logs": total_count,
                "action_breakdown": dict(action_counts),
                "daily_activity": [
                    {"date": str(date), "count": count}
                    for date, count in daily_activity
                ],
                "period_days": days,
            }

        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get audit log summary: {e}")
            return {
                "total_logs": 0,
                "action_breakdown": {},
                "daily_activity": [],
                "period_days": days,
            }
        finally:
            session.close()
        """
        Check database connection health.
        
        Returns:
            True if database is healthy
        """
        try:
            session = self.get_session()
            session.execute("SELECT 1")
            session.close()
            return True
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False

    def execute_query(
        self, query: str, params: Optional[tuple] = None, timeout: Optional[int] = None
    ):
        """
        Execute a SELECT query and return results as DataFrame.

        Args:
            query: SQL query string
            params: Query parameters
            timeout: Query timeout in seconds

        Returns:
            DataFrame with results or affected rows count for non-SELECT
        """
        if self.settings:
            # psycopg2 mode
            conn = self.connect()
            try:
                with conn.cursor() as cursor:
                    if timeout:
                        cursor.execute("SET statement_timeout = %s", (timeout * 1000,))
                    cursor.execute(query, params or ())

                    if query.strip().upper().startswith("SELECT"):
                        results = cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description]
                        self.logger.debug(
                            f"Executed query successfully, returned {len(results)} rows"
                        )
                        return pd.DataFrame(results, columns=columns)
                    else:
                        self.logger.debug(
                            f"Executed non-SELECT query, affected {cursor.rowcount} rows"
                        )
                        return cursor.rowcount
            except Exception as e:
                self.logger.error(f"Query execution failed: {e}")
                raise QueryExecutionError(f"Failed to execute query: {e}")
            finally:
                # For tests, don't close the connection
                pass
        else:
            # SQLAlchemy mode - not implemented for general queries
            raise NotImplementedError("execute_query not supported in SQLAlchemy mode")

    async def fetch_all(
        self, query: str, params: Optional[tuple] = None, timeout: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results as list of dictionaries.

        This is an async wrapper around execute_query for compatibility with
        async drift detection services.

        Args:
            query: SQL query string
            params: Query parameters
            timeout: Query timeout in seconds

        Returns:
            List of dictionaries with query results
        """
        if self.settings:
            # psycopg2 mode
            conn = self.connect()
            try:
                with conn.cursor() as cursor:
                    if timeout:
                        cursor.execute("SET statement_timeout = %s", (timeout * 1000,))
                    cursor.execute(query, params or ())

                    if query.strip().upper().startswith("SELECT"):
                        results = cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description]
                        # Convert to list of dicts
                        return [dict(zip(columns, row)) for row in results]
                    else:
                        return []
            except Exception as e:
                self.logger.error(f"Query execution failed: {e}")
                raise QueryExecutionError(f"Failed to execute query: {e}")
        else:
            # SQLAlchemy mode
            session = self.get_session()
            try:
                from sqlalchemy import text

                # Convert tuple params to dict for SQLAlchemy
                if params:
                    # SQLAlchemy text() expects positional params as list
                    # We need to replace %s with :param1, :param2, etc.
                    param_dict = {}
                    modified_query = query
                    for i, param_value in enumerate(params):
                        placeholder = f":param{i}"
                        modified_query = modified_query.replace("%s", placeholder, 1)
                        param_dict[f"param{i}"] = param_value

                    result = session.execute(text(modified_query), param_dict)
                else:
                    result = session.execute(text(query))

                columns = result.keys()
                rows = result.fetchall()

                # Convert to list of dicts
                return [dict(zip(columns, row)) for row in rows]
            except SQLAlchemyError as e:
                self.logger.error(f"Query execution failed: {e}")
                raise QueryExecutionError(f"Failed to execute query: {e}")
            finally:
                session.close()

    async def execute(
        self, query: str, params: Optional[tuple] = None, timeout: Optional[int] = None
    ) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query and return affected rows count.

        This is an async wrapper for non-SELECT queries.

        Args:
            query: SQL query string (INSERT, UPDATE, DELETE)
            params: Query parameters
            timeout: Query timeout in seconds

        Returns:
            Number of affected rows
        """
        if self.settings:
            # psycopg2 mode
            conn = self.connect()
            try:
                with conn.cursor() as cursor:
                    if timeout:
                        cursor.execute("SET statement_timeout = %s", (timeout * 1000,))
                    cursor.execute(query, params or ())
                    conn.commit()
                    return cursor.rowcount
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Query execution failed: {e}")
                raise QueryExecutionError(f"Failed to execute query: {e}")
        else:
            # SQLAlchemy mode
            session = self.get_session()
            try:
                from sqlalchemy import text

                # Convert tuple params to dict for SQLAlchemy
                if params:
                    param_dict = {}
                    modified_query = query
                    for i, param_value in enumerate(params):
                        placeholder = f":param{i}"
                        modified_query = modified_query.replace("%s", placeholder, 1)
                        param_dict[f"param{i}"] = param_value

                    result = session.execute(text(modified_query), param_dict)
                else:
                    result = session.execute(text(query))

                session.commit()
                return result.rowcount
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"Query execution failed: {e}")
                raise QueryExecutionError(f"Failed to execute query: {e}")
            finally:
                session.close()

    def insert_transaction(self, transaction_data: Dict[str, Any]) -> int:
        """
        Insert a transaction record.

        Args:
            transaction_data: Transaction data dictionary with Time, V1-V28, amount, Class

        Returns:
            Transaction ID
        """
        if self.settings:
            # Build column list and placeholders for Time, V1-V28, amount, Class
            columns = ["time"] + [f"v{i}" for i in range(1, 29)] + ["amount", "class"]
            placeholders = ", ".join(["%s"] * len(columns))
            column_str = ", ".join(columns)

            query = f"""
                INSERT INTO transactions ({column_str})
                VALUES ({placeholders})
                RETURNING id
            """

            # Extract values in correct order
            params = (
                transaction_data["time"],
                *[transaction_data[f"v{i}"] for i in range(1, 29)],
                transaction_data["amount"],
                transaction_data.get("Class", 0),  # Default to 0 if not provided
            )

            conn = self.connect()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    transaction_id = cursor.fetchone()[0]
                    conn.commit()
                    return transaction_id
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Transaction insertion failed: {e}")
                raise TransactionError(f"Failed to insert transaction: {e}")
            finally:
                if hasattr(conn, "putconn"):
                    conn.putconn(conn)
        else:
            raise NotImplementedError(
                "insert_transaction not supported in SQLAlchemy mode"
            )

    def bulk_insert_transactions(self, transactions: List[Dict[str, Any]]) -> int:
        """
        Bulk insert multiple transactions.

        Args:
            transactions: List of transaction dictionaries with Time, V1-V28, amount, Class

        Returns:
            Number of rows inserted
        """
        if not transactions:
            return 0  # Handle empty batch

        if self.settings:
            # Build column list and placeholders for Time, V1-V28, amount, Class
            columns = ["time"] + [f"v{i}" for i in range(1, 29)] + ["amount", "class"]
            placeholders = ", ".join(["%s"] * len(columns))
            column_str = ", ".join(columns)

            query = f"""
                INSERT INTO transactions ({column_str})
                VALUES ({placeholders})
            """

            # Build params list with values in correct order
            params = []
            for t in transactions:
                row = (
                    t["time"],
                    *[t[f"v{i}"] for i in range(1, 29)],
                    t["amount"],
                    t.get("Class", 0),  # Default to 0 if not provided
                )
                params.append(row)

            conn = self.connect()
            try:
                with conn.cursor() as cursor:
                    cursor.executemany(query, params)
                    conn.commit()
                    return cursor.rowcount
            except Exception:
                conn.rollback()
                raise
            finally:
                if hasattr(conn, "putconn"):
                    conn.putconn(conn)
        else:
            raise NotImplementedError(
                "bulk_insert_transactions not supported in SQLAlchemy mode"
            )

    def get_transaction_by_id(self, transaction_id: int):
        """
        Get transaction by ID.

        Args:
            transaction_id: Transaction ID

        Returns:
            Transaction data as Series or None
        """
        if self.settings:
            query = "SELECT * FROM transactions WHERE id = %s"
            df = self.execute_query(query, (transaction_id,))
            if len(df) == 0:
                return None
            return df.iloc[0]
        else:
            raise NotImplementedError(
                "get_transaction_by_id not supported in SQLAlchemy mode"
            )

    def update_transaction_status(self, transaction_id: int, status: str) -> bool:
        """
        Update transaction status.

        Args:
            transaction_id: Transaction ID
            status: New status

        Returns:
            True if updated, False if not found
        """
        if self.settings:
            query = "UPDATE transactions SET status = %s WHERE id = %s"
            result = self.execute_query(query, (status, transaction_id))
            return result > 0
        else:
            raise NotImplementedError(
                "update_transaction_status not supported in SQLAlchemy mode"
            )

    def get_recent_transactions(self, limit: int = 100):
        """
        Get recent transactions.

        Args:
            limit: Maximum number of transactions to return

        Returns:
            DataFrame with recent transactions
        """
        if self.settings:
            query = "SELECT * FROM transactions ORDER BY id DESC LIMIT %s"
            return self.execute_query(query, (limit,))
        else:
            raise NotImplementedError(
                "get_recent_transactions not supported in SQLAlchemy mode"
            )

    def get_fraud_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get fraud statistics for the specified time period.

        Args:
            hours: Number of hours to look back

        Returns:
            Statistics dictionary
        """
        if self.settings:
            query = """
                SELECT 
                    COUNT(*) as total_transactions,
                    SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) as fraud_count,
                    AVG(CASE WHEN is_fraud = 1 THEN 1.0 ELSE 0.0 END) as fraud_rate,
                    AVG(CASE WHEN is_fraud = 1 THEN amount END) as avg_fraud_amount,
                    MAX(CASE WHEN is_fraud = 1 THEN amount END) as max_fraud_amount
                FROM transactions 
                WHERE created_at >= NOW() - INTERVAL '%s hours'
            """
            df = self.execute_query(query, (hours,))
            if len(df) == 0:
                return {
                    "total_transactions": 0,
                    "fraud_count": 0,
                    "fraud_rate": 0.0,
                    "avg_fraud_amount": 0.0,
                    "max_fraud_amount": 0.0,
                }
            row = df.iloc[0]
            return {
                "total_transactions": int(row["total_transactions"]),
                "fraud_count": int(row["fraud_count"] or 0),
                "fraud_rate": float(row["fraud_rate"] or 0.0),
                "avg_fraud_amount": float(row["avg_fraud_amount"] or 0.0),
                "max_fraud_amount": float(row["max_fraud_amount"] or 0.0),
            }
        else:
            raise NotImplementedError(
                "get_fraud_statistics not supported in SQLAlchemy mode"
            )

    def transaction(self):
        """Context manager for database transactions."""
        if self.settings:

            class TransactionContext:
                def __init__(self, service):
                    self.service = service
                    self.conn = None

                def __enter__(self):
                    self.conn = self.service.connect()
                    return self.conn

                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type:
                        self.conn.rollback()
                    else:
                        self.conn.commit()
                    if hasattr(self.conn, "putconn"):
                        self.conn.putconn(self.conn)

            return TransactionContext(self)
        else:
            raise NotImplementedError(
                "transaction context manager not supported in SQLAlchemy mode"
            )

    def health_check(self) -> Dict[str, Any]:
        """
        Check database health.

        Returns:
            Health status dictionary
        """
        if self.settings:
            try:
                conn = self.connect()
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                if hasattr(conn, "putconn"):
                    conn.putconn(conn)
                return {"status": "healthy", "connection": "ok"}
            except Exception as e:
                return {"status": "unhealthy", "connection_error": str(e)}
        else:
            # SQLAlchemy mode
            try:
                session = self.get_session()
                session.execute("SELECT 1")
                session.close()
                return {"status": "healthy", "connection": "ok"}
            except Exception as e:
                return {"status": "unhealthy", "connection_error": str(e)}
