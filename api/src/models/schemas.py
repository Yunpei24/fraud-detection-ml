"""
Pydantic schemas for request/response validation.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict


class TransactionRequest(BaseModel):
    """Request schema for fraud prediction - Credit Card Dataset."""
    
    transaction_id: str = Field(..., min_length=1, max_length=100)
    features: List[float] = Field(..., min_length=30, max_length=30)
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN-001",
                "features": [0.0] + [-1.36, -0.07] + [0.0] * 26 + [149.62],
                "metadata": {"source": "test"}
            }
        }
    )
    
    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        """Validate 30 features: Time, V1-V28, Amount."""
        if len(v) != 30:
            raise ValueError(f"Expected exactly 30 features, got {len(v)}")
        
        for i, feature in enumerate(v):
            if not isinstance(feature, (int, float)):
                raise ValueError(f"Feature {i} must be numeric")
            if feature != feature or abs(feature) == float('inf'):  # NaN or Inf
                raise ValueError(f"Feature {i} is NaN or Inf")
        
        return v


class BatchTransactionRequest(BaseModel):
    """Batch prediction request schema."""
    
    transactions: List[TransactionRequest] = Field(..., min_length=1, max_length=100)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transactions": [
                    {
                        "transaction_id": "TXN-001",
                        "features": [0.0] * 30,
                        "metadata": {"index": 0}
                    },
                    {
                        "transaction_id": "TXN-002",
                        "features": [0.0] * 30,
                        "metadata": {"index": 1}
                    }
                ]
            }
        }
    )
    
    @field_validator("transactions")
    @classmethod
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 transactions")
        return v


class PredictionResponse(BaseModel):
    """Response schema for fraud prediction."""
    
    transaction_id: str
    prediction: int = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    processing_time: float
    model_version: str
    timestamp: float
    explanation: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "transaction_id": "TXN-001",
                "prediction": 1,
                "confidence": 0.95,
                "fraud_score": 0.89,
                "risk_level": "HIGH",
                "processing_time": 0.045,
                "model_version": "1.0.0",
                "timestamp": 1697712000.0
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""
    
    total_transactions: int
    successful_predictions: int
    failed_predictions: int
    fraud_detected: int
    fraud_rate: float
    predictions: List[PredictionResponse]
    processing_time: float
    avg_processing_time: float


class HealthCheckResponse(BaseModel):
    """Response schema for health check."""
    
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float


class DetailedHealthCheckResponse(BaseModel):
    """Response schema for detailed health check."""
    
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    components: Dict[str, Dict[str, Any]]
    environment: str


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ModelVersionResponse(BaseModel):
    """Response schema for model version info."""
    
    version: str
    models: Dict[str, str]
    loaded_at: str

