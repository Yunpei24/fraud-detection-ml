"""
Database service for storing predictions and audit logs.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from ..config import get_logger, settings
from ..utils import DatabaseException

logger = get_logger(__name__)
Base = declarative_base()


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
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database service.
        
        Args:
            database_url: Database connection URL (optional)
        """
        self.database_url = database_url or settings.database_url
        self.engine = None
        self.SessionLocal = None
        self.logger = logger
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize SQLAlchemy engine and session factory."""
        try:
            self.engine = create_engine(
                self.database_url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            self.logger.info("Database engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize database engine: {e}")
            raise DatabaseException(
                "Failed to initialize database",
                details={"error": str(e)}
            )
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created")
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise DatabaseException(
                "Failed to create database tables",
                details={"error": str(e)}
            )
    
    def get_session(self) -> Session:
        """
        Get database session.
        
        Returns:
            SQLAlchemy session
        """
        return self.SessionLocal()
    
    async def save_prediction(
        self,
        transaction_id: str,
        prediction: Dict[str, Any]
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
                request_metadata=json.dumps(prediction.get("metadata", {}))
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
                details={
                    "transaction_id": transaction_id,
                    "error": str(e)
                }
            )
        finally:
            session.close()
    
    async def save_audit_log(
        self,
        transaction_id: str,
        action: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
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
                details=json.dumps(details or {})
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
        notes: Optional[str] = None
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
                notes=notes
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
                details={
                    "transaction_id": transaction_id,
                    "error": str(e)
                }
            )
        finally:
            session.close()
    
    async def get_prediction(
        self,
        transaction_id: str
    ) -> Optional[Dict[str, Any]]:
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
            
            log = session.query(PredictionLog).filter_by(
                transaction_id=transaction_id
            ).first()
            
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
                "created_at": log.created_at.isoformat()
            }
            
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get prediction: {e}")
            return None
        finally:
            session.close()
    
    def check_health(self) -> bool:
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
