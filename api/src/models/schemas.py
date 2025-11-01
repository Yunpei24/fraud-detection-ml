"""
Pydantic schemas for request/response validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
                "metadata": {"source": "test"},
            }
        }
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        """Validate 30 features: Time, V1-V28, amount, time."""
        if len(v) != 30:
            raise ValueError(f"Expected exactly 30 features, got {len(v)}")

        for i, feature in enumerate(v):
            if not isinstance(feature, (int, float)):
                raise ValueError(f"Feature {i} must be numeric")
            if feature != feature or abs(feature) == float("inf"):  # NaN or Inf
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
                        "metadata": {"index": 0},
                    },
                    {
                        "transaction_id": "TXN-002",
                        "features": [0.0] * 30,
                        "metadata": {"index": 1},
                    },
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
                "timestamp": 1697712000.0,
            }
        },
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


class TokenResponse(BaseModel):
    """Response schema for JWT token."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "user": {"username": "admin", "role": "admin", "is_active": True},
            }
        }
    )


class UserResponse(BaseModel):
    """Response schema for user information."""

    username: str
    role: str
    is_active: bool

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"username": "admin", "role": "admin", "is_active": True}
        }
    )


# ==============================================================================
# USER MANAGEMENT SCHEMAS (Admin Endpoints)
# ==============================================================================


class UserCreateRequest(BaseModel):
    """Request schema for creating a new user (Admin only)."""

    username: str = Field(..., min_length=3, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    first_name: Optional[str] = Field(default=None, max_length=100)
    last_name: Optional[str] = Field(default=None, max_length=100)
    role: str = Field(default="analyst", pattern="^(admin|analyst|viewer)$")
    department: Optional[str] = Field(default=None, max_length=100)
    is_active: bool = Field(default=True)
    is_verified: bool = Field(default=False)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "john_analyst",
                "email": "john.doe@company.com",
                "password": "SecurePass123!",
                "first_name": "John",
                "last_name": "Doe",
                "role": "analyst",
                "department": "Fraud Detection",
                "is_active": True,
                "is_verified": False,
            }
        }
    )

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        """Basic email validation."""
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email format")
        return v.lower()

    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        """Validate username format."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Username can only contain letters, numbers, underscores and hyphens"
            )
        return v.lower()


class UserUpdateRequest(BaseModel):
    """Request schema for updating an existing user (Admin only)."""

    email: Optional[str] = Field(default=None, max_length=255)
    first_name: Optional[str] = Field(default=None, max_length=100)
    last_name: Optional[str] = Field(default=None, max_length=100)
    role: Optional[str] = Field(default=None, pattern="^(admin|analyst|viewer)$")
    department: Optional[str] = Field(default=None, max_length=100)
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "john.doe@newcompany.com",
                "first_name": "John",
                "last_name": "Doe",
                "role": "analyst",
                "department": "Risk Management",
                "is_active": True,
                "is_verified": True,
            }
        }
    )

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        """Basic email validation."""
        if v is not None:
            if "@" not in v or "." not in v.split("@")[-1]:
                raise ValueError("Invalid email format")
            return v.lower()
        return v


class UserPasswordResetRequest(BaseModel):
    """Request schema for resetting user password (Admin only)."""

    new_password: str = Field(..., min_length=8, max_length=128)

    model_config = ConfigDict(
        json_schema_extra={"example": {"new_password": "NewSecurePass456!"}}
    )


class UserRoleUpdateRequest(BaseModel):
    """Request schema for updating user role (Admin only)."""

    role: str = Field(..., pattern="^(admin|analyst|viewer)$")

    model_config = ConfigDict(json_schema_extra={"example": {"role": "admin"}})


class UserActivationRequest(BaseModel):
    """Request schema for activating/deactivating a user (Admin only)."""

    is_active: bool

    model_config = ConfigDict(json_schema_extra={"example": {"is_active": True}})


class UserDetailResponse(BaseModel):
    """Detailed response schema for user information (Admin view)."""

    id: int
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: str
    department: Optional[str] = None
    is_active: bool
    is_verified: bool
    created_at: str
    last_login: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 123,
                "username": "john_analyst",
                "email": "john.doe@company.com",
                "first_name": "John",
                "last_name": "Doe",
                "role": "analyst",
                "department": "Fraud Detection",
                "is_active": True,
                "is_verified": True,
                "created_at": "2025-01-15T10:30:00Z",
                "last_login": "2025-01-20T14:25:00Z",
            }
        }
    )


class UserListResponse(BaseModel):
    """Response schema for listing users (Admin only)."""

    users: List[UserDetailResponse]
    total_count: int
    limit: int
    offset: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "users": [
                    {
                        "id": 1,
                        "username": "admin",
                        "email": "admin@company.com",
                        "first_name": "Admin",
                        "last_name": "User",
                        "role": "admin",
                        "department": "IT",
                        "is_active": True,
                        "is_verified": True,
                        "created_at": "2025-01-01T00:00:00Z",
                        "last_login": "2025-01-20T09:15:00Z",
                    },
                    {
                        "id": 2,
                        "username": "john_analyst",
                        "email": "john@company.com",
                        "first_name": "John",
                        "last_name": "Doe",
                        "role": "analyst",
                        "department": "Fraud Detection",
                        "is_active": True,
                        "is_verified": True,
                        "created_at": "2025-01-10T00:00:00Z",
                        "last_login": "2025-01-19T16:30:00Z",
                    },
                ],
                "total_count": 2,
                "limit": 50,
                "offset": 0,
            }
        }
    )


class UserOperationResponse(BaseModel):
    """Response schema for user operations (create, update, delete)."""

    success: bool
    message: str
    user: Optional[UserDetailResponse] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "User created successfully",
                "user": {
                    "id": 123,
                    "username": "john_analyst",
                    "email": "john@company.com",
                    "first_name": "John",
                    "last_name": "Doe",
                    "role": "analyst",
                    "department": "Fraud Detection",
                    "is_active": True,
                    "is_verified": False,
                    "created_at": "2025-01-20T10:30:00Z",
                    "last_login": None,
                },
            }
        }
    )


class ModelVersionResponse(BaseModel):
    """Response schema for model version info."""

    version: str
    models: Dict[str, str]
    loaded_at: str


class ExplanationRequest(BaseModel):
    """Request schema for model explanation."""

    transaction_id: str = Field(..., min_length=1, max_length=100)
    features: List[float] = Field(..., min_length=30, max_length=30)
    model_type: Optional[str] = Field(
        default="ensemble",
        description="Model to explain: 'ensemble', 'xgboost', 'neural_network', 'isolation_forest'",
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN-001",
                "features": [0.0] + [-1.36, -0.07] + [0.0] * 26 + [149.62],
                "model_type": "xgboost",
                "metadata": {"source": "explanation_request"},
            }
        }
    )

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v):
        """Validate model type."""
        valid_types = ["ensemble", "xgboost", "neural_network", "isolation_forest"]
        if v not in valid_types:
            raise ValueError(f"model_type must be one of: {valid_types}")
        return v

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        """Validate 30 features: Time, V1-V28, amount."""
        if len(v) != 30:
            raise ValueError(f"Expected exactly 30 features, got {len(v)}")

        for i, feature in enumerate(v):
            if not isinstance(feature, (int, float)):
                raise ValueError(f"Feature {i} must be numeric")
            if feature != feature or abs(feature) == float("inf"):  # NaN or Inf
                raise ValueError(f"Feature {i} is NaN or Inf")

        return v


class FeatureImportance(BaseModel):
    """Schema for individual feature importance."""

    feature_name: str
    importance_score: float
    rank: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"feature_name": "V10", "importance_score": 0.234, "rank": 1}
        }
    )


class SHAPExplanationResponse(BaseModel):
    """Response schema for SHAP explanation."""

    transaction_id: str
    model_type: str
    method: str = "SHAP"
    top_features: List[FeatureImportance]
    base_value: float
    prediction_value: float
    shap_values_summary: Dict[str, Any]
    processing_time: float
    timestamp: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN-001",
                "model_type": "xgboost",
                "method": "SHAP",
                "top_features": [
                    {"feature_name": "V10", "importance_score": 0.234, "rank": 1},
                    {"feature_name": "V4", "importance_score": -0.156, "rank": 2},
                ],
                "base_value": 0.001,
                "prediction_value": 0.85,
                "shap_values_summary": {
                    "positive_contributors": 3,
                    "negative_contributors": 7,
                    "total_features": 30,
                },
                "processing_time": 0.045,
                "timestamp": 1697712000.0,
            }
        }
    )


class FeatureImportanceResponse(BaseModel):
    """Response schema for global feature importance."""

    model_type: str
    method: str = "feature_importance"
    feature_importances: List[FeatureImportance]
    total_features: int
    processing_time: float
    timestamp: float


class DriftStatusResponse(BaseModel):
    """Response schema for drift detection status."""

    overall_drift_detected: bool
    data_drift_detected: bool
    target_drift_detected: bool
    concept_drift_detected: bool
    last_check: str
    baseline_available: bool
    modules_available: bool
    recent_metrics: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "overall_drift_detected": False,
                "data_drift_detected": False,
                "target_drift_detected": False,
                "concept_drift_detected": False,
                "last_check": "2025-01-15T10:30:00Z",
                "baseline_available": True,
                "modules_available": True,
                "recent_metrics": {
                    "data_drift": [],
                    "target_drift": [],
                    "concept_drift": [],
                },
            }
        }
    )


class DriftDetectionRequest(BaseModel):
    """Request schema for drift detection."""

    features_data: List[List[float]] = Field(..., min_length=1, max_length=1000)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features_data": [
                    [
                        0.5,
                        -1.36,
                        2.54,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        149.62,
                    ],
                    [
                        1.2,
                        0.45,
                        -0.89,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        75.30,
                    ],
                ]
            }
        }
    )

    @field_validator("features_data")
    @classmethod
    def validate_features_data(cls, v):
        """Validate features data."""
        if not v:
            raise ValueError("features_data cannot be empty")

        first_row_length = len(v[0])
        for i, row in enumerate(v):
            if len(row) != 30:
                raise ValueError(
                    f"Row {i} must have exactly 30 features, got {len(row)}"
                )
            if len(row) != first_row_length:
                raise ValueError(
                    f"All rows must have the same number of features. Row 0 has {first_row_length}, row {i} has {len(row)}"
                )

        return v


class DriftDetectionResponse(BaseModel):
    """Response schema for drift detection results."""

    drift_detected: bool
    data_drift: Optional[Dict[str, Any]] = None
    target_drift: Optional[Dict[str, Any]] = None
    concept_drift: Optional[Dict[str, Any]] = None
    timestamp: str
    sample_size: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "drift_detected": False,
                "data_drift": {
                    "drift_score": 0.15,
                    "features_drifted": ["feature_10", "feature_15"],
                    "threshold": 0.3,
                },
                "target_drift": None,
                "concept_drift": {"drift_score": 0.02, "threshold": 0.05},
                "timestamp": "2025-01-15T10:30:00Z",
                "sample_size": 100,
            }
        }
    )


class BaselineUpdateRequest(BaseModel):
    """Request schema for baseline update."""

    baseline_data: List[List[float]] = Field(..., min_length=100, max_length=10000)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "baseline_data": [
                    [
                        0.5,
                        -1.36,
                        2.54,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        149.62,
                    ],
                    [
                        1.2,
                        0.45,
                        -0.89,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        75.30,
                    ],
                ]
            }
        }
    )

    @field_validator("baseline_data")
    @classmethod
    def validate_baseline_data(cls, v):
        """Validate baseline data."""
        if len(v) < 100:
            raise ValueError("Baseline data must contain at least 100 samples")

        first_row_length = len(v[0])
        for i, row in enumerate(v):
            if len(row) != 30:
                raise ValueError(
                    f"Row {i} must have exactly 30 features, got {len(row)}"
                )
            if len(row) != first_row_length:
                raise ValueError(f"All rows must have the same number of features")

        return v


class BaselineUpdateResponse(BaseModel):
    """Response schema for baseline update."""

    success: bool
    message: str
    timestamp: str
    sample_count: Optional[int] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Baseline updated with 1000 samples",
                "timestamp": "2025-01-15T10:30:00Z",
                "sample_count": 1000,
            }
        }
    )


class BaselineInfoResponse(BaseModel):
    """Response schema for baseline information."""

    baseline_available: bool
    sample_count: int
    feature_count: int
    last_updated: Optional[str] = None
    source: Optional[str] = None


class ComprehensiveDriftResponse(BaseModel):
    """Response schema for comprehensive drift detection using Evidently AI."""

    timestamp: datetime
    analysis_window: str
    reference_window: str
    data_drift: Dict[str, Any]
    target_drift: Dict[str, Any]
    concept_drift: Dict[str, Any]
    multivariate_drift: Dict[str, Any]
    drift_summary: Dict[str, Any]
    processing_time: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2025-01-15T10:30:00Z",
                "analysis_window": "24h",
                "reference_window": "30d",
                "data_drift": {
                    "dataset_drift_detected": False,
                    "drift_share": 0.02,
                    "drifted_columns": [],
                    "statistical_tests": [],
                },
                "target_drift": {
                    "drift_detected": False,
                    "drift_score": 0.05,
                    "current_fraud_rate": 0.0058,
                    "reference_fraud_rate": 0.0059,
                    "rate_change_percent": -1.69,
                    "stattest": "psi_stat_test",
                },
                "concept_drift": {
                    "drift_detected": False,
                    "drift_score": 0.02,
                    "stattest_name": "correlation_difference",
                    "features_analyzed": ["amount", "v1", "v2", "v3"],
                },
                "multivariate_drift": {
                    "tests": [
                        {
                            "name": "TestAllFeaturesValueDrift",
                            "status": "SUCCESS",
                            "description": "Test if all features have drifted",
                            "parameters": {},
                        }
                    ],
                    "overall_drift_detected": False,
                    "drift_columns_count": 0,
                },
                "drift_summary": {
                    "overall_drift_detected": False,
                    "drift_types_detected": [],
                    "severity_score": 0,
                    "recommendations": [
                        "LOW: No significant drift detected - continue monitoring"
                    ],
                },
            }
        }
    )


class SlidingWindowAnalysisResponse(BaseModel):
    """Response schema for sliding window drift analysis."""

    timestamp: datetime
    window_size: str
    step_size: str
    analysis_period: str
    windows: List[Dict[str, Any]]
    processing_time: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2025-01-15T10:30:00Z",
                "window_size": "24h",
                "step_size": "6h",
                "analysis_period": "7d",
                "windows": [
                    {
                        "window_id": 1,
                        "start_time": "2025-01-08T10:30:00Z",
                        "end_time": "2025-01-09T10:30:00Z",
                        "record_count": 1250,
                        "drift_detected": False,
                        "drift_score": 0.01,
                    },
                    {
                        "window_id": 2,
                        "start_time": "2025-01-08T16:30:00Z",
                        "end_time": "2025-01-09T16:30:00Z",
                        "record_count": 1180,
                        "drift_detected": False,
                        "drift_score": 0.02,
                    },
                ],
            }
        }
    )


class DriftReportResponse(BaseModel):
    """Response schema for automated drift report generation."""

    timestamp: datetime
    summary: Dict[str, Any]
    recommendations: List[str]
    alerts: List[Dict[str, Any]]
    severity: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2025-01-15T10:30:00Z",
                "summary": {
                    "overall_drift_detected": False,
                    "drift_types_detected": [],
                    "severity_score": 0,
                    "recommendations": [
                        "LOW: No significant drift detected - continue monitoring"
                    ],
                },
                "recommendations": [
                    "Continue regular monitoring",
                    "Review model performance metrics monthly",
                ],
                "alerts": [],
                "severity": "LOW",
            }
        }
    )


class AuditLogEntry(BaseModel):
    """Schema for individual audit log entry."""

    id: int
    transaction_id: str
    action: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    details: Dict[str, Any]
    timestamp: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 123,
                "transaction_id": "TXN-001",
                "action": "prediction",
                "user_id": "user123",
                "ip_address": "192.168.1.100",
                "details": {"prediction": 1, "confidence": 0.95},
                "timestamp": "2025-01-15T10:30:00Z",
            }
        }
    )


class AuditLogsResponse(BaseModel):
    """Response schema for audit logs query."""

    logs: List[AuditLogEntry]
    total_count: int
    limit: int
    offset: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "logs": [
                    {
                        "id": 123,
                        "transaction_id": "TXN-001",
                        "action": "prediction",
                        "user_id": "user123",
                        "ip_address": "192.168.1.100",
                        "details": {"prediction": 1, "confidence": 0.95},
                        "timestamp": "2025-01-15T10:30:00Z",
                    }
                ],
                "total_count": 150,
                "limit": 50,
                "offset": 0,
            }
        }
    )


class AuditLogSummaryResponse(BaseModel):
    """Response schema for audit log summary."""

    total_logs: int
    action_breakdown: Dict[str, int]
    daily_activity: List[Dict[str, Any]]
    period_days: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_logs": 1250,
                "action_breakdown": {
                    "prediction": 1000,
                    "explanation": 150,
                    "drift_detection": 100,
                },
                "daily_activity": [
                    {"date": "2025-01-15", "count": 45},
                    {"date": "2025-01-16", "count": 52},
                ],
                "period_days": 30,
            }
        }
    )


class AuditQueryRequest(BaseModel):
    """Request schema for audit log queries."""

    transaction_id: Optional[str] = None
    user_id: Optional[str] = None
    action: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN-001",
                "action": "prediction",
                "limit": 50,
                "offset": 0,
                "start_date": "2025-01-01T00:00:00Z",
                "end_date": "2025-01-31T23:59:59Z",
            }
        }
    )


class TransactionUpdateRequest(BaseModel):
    """Request schema for updating transaction class prediction."""

    transaction_id: str = Field(..., min_length=1, max_length=100)
    analyst_label: int = Field(
        ..., ge=0, le=1, description="Analyst's fraud label: 0=legitimate, 1=fraud"
    )
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Analyst confidence score"
    )
    notes: Optional[str] = Field(
        default=None, max_length=1000, description="Additional notes from analyst"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN-001",
                "analyst_label": 1,
                "confidence": 0.95,
                "notes": "Suspicious transaction pattern detected",
            }
        }
    )


class TransactionUpdateResponse(BaseModel):
    """Response schema for transaction update."""

    transaction_id: str
    analyst_label: int
    analyst_id: str
    confidence: Optional[float] = None
    notes: Optional[str] = None
    labeled_at: str
    success: bool
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN-001",
                "analyst_label": 1,
                "analyst_id": "analyst123",
                "confidence": 0.95,
                "notes": "Suspicious transaction pattern detected",
                "labeled_at": "2025-01-15T10:30:00Z",
                "success": True,
                "message": "Transaction label updated successfully",
            }
        }
    )


class TransactionLabelHistoryResponse(BaseModel):
    """Response schema for transaction label history."""

    transaction_id: str
    labels: List[Dict[str, Any]]
    total_labels: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN-001",
                "labels": [
                    {
                        "analyst_label": 1,
                        "analyst_id": "analyst123",
                        "confidence": 0.95,
                        "notes": "Suspicious pattern",
                        "labeled_at": "2025-01-15T10:30:00Z",
                    },
                    {
                        "analyst_label": 0,
                        "analyst_id": "analyst456",
                        "confidence": 0.80,
                        "notes": "False positive",
                        "labeled_at": "2025-01-16T14:20:00Z",
                    },
                ],
                "total_labels": 2,
            }
        }
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "baseline_available": True,
                "sample_count": 10000,
                "feature_count": 30,
                "last_updated": "2025-01-15T10:30:00Z",
                "source": "training_data",
            }
        }
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_type": "xgboost",
                "method": "feature_importance",
                "feature_importances": [
                    {"feature_name": "V10", "importance_score": 0.234, "rank": 1},
                    {"feature_name": "V4", "importance_score": 0.189, "rank": 2},
                ],
                "total_features": 30,
                "processing_time": 0.023,
                "timestamp": 1697712000.0,
            }
        }
    )
