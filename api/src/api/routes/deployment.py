"""
Admin Deployment Routes
=======================

Endpoints for managing canary deployments, promotions, and rollbacks.
These endpoints are called by Airflow DAGs to orchestrate model deployments.

Security:
- All endpoints require admin authentication
- JWT token must be provided in Authorization header
- Only admin role can access these endpoints

Flow:
1. deploy-canary: Deploy models with specified traffic percentage
2. deployment-status: Monitor deployment configuration
3. promote-to-production: Transition models to 100% production
4. rollback-deployment: Revert to champion models
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import os
import json
import logging
from pathlib import Path
from datetime import datetime

from .auth import get_current_admin_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/deployment", tags=["deployment"])


# ============================================================================
# Request/Response Models
# ============================================================================


class CanaryDeploymentRequest(BaseModel):
    """Request for canary deployment."""

    model_uris: List[str] = Field(
        ...,
        description="List of MLflow model URIs (e.g., models:/model_name/Staging)",
        min_items=1,
        max_items=10,
    )
    traffic_percentage: int = Field(
        ...,
        description="Percentage of traffic to route to canary (0-100)",
        ge=0,
        le=100,
    )

    @validator("model_uris")
    def validate_model_uris(cls, v):
        """Validate model URI format."""
        for uri in v:
            if not uri.startswith("models:/"):
                raise ValueError(f"Invalid model URI format: {uri}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "model_uris": [
                    "models:/fraud_detection_xgboost/Staging",
                    "models:/fraud_detection_random_forest/Staging",
                ],
                "traffic_percentage": 5,
            }
        }


class PromoteToProductionRequest(BaseModel):
    """Request to promote models to production."""

    model_uris: List[str] = Field(
        ...,
        description="List of MLflow model URIs to promote",
        min_items=1,
        max_items=10,
    )

    @validator("model_uris")
    def validate_model_uris(cls, v):
        """Validate model URI format."""
        for uri in v:
            if not uri.startswith("models:/"):
                raise ValueError(f"Invalid model URI format: {uri}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "model_uris": [
                    "models:/fraud_detection_xgboost/Staging",
                    "models:/fraud_detection_random_forest/Staging",
                ]
            }
        }


class DeploymentResponse(BaseModel):
    """Response for deployment operations."""

    status: str = Field(..., description="Operation status: success or error")
    message: str = Field(..., description="Human-readable message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional details about the operation"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Operation timestamp (UTC)",
    )


class DeploymentStatusResponse(BaseModel):
    """Response for deployment status."""

    deployment_mode: str = Field(
        ..., description="Current mode: production_only, canary_active"
    )
    canary_percentage: int = Field(
        ..., description="Current canary traffic percentage (0-100)"
    )
    champion_models: Optional[List[str]] = Field(
        None, description="List of champion model URIs"
    )
    challenger_models: Optional[List[str]] = Field(
        None, description="List of challenger model URIs"
    )
    last_update: Optional[str] = Field(None, description="Last configuration update")
    config: Optional[Dict[str, Any]] = Field(
        None, description="Full traffic routing configuration"
    )


# ============================================================================
# Helper Functions
# ============================================================================


def get_traffic_config_path() -> Path:
    """Get path to traffic routing configuration file."""
    config_dir = Path("/app/config")
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "traffic_routing.json"


def load_traffic_config() -> Dict[str, Any]:
    """Load current traffic routing configuration."""
    config_path = get_traffic_config_path()

    if not config_path.exists():
        # Default configuration: 100% champion, canary disabled
        return {
            "canary_enabled": False,
            "canary_traffic_pct": 0,
            "champion_traffic_pct": 100,
            "canary_model_uris": [],
            "champion_model_uris": [],
            "ensemble_weights": {
                "xgboost": 0.50,
                "random_forest": 0.30,
                "neural_network": 0.15,
                "isolation_forest": 0.05,
            },
            "last_update": datetime.utcnow().isoformat(),
        }

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading traffic config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load traffic configuration: {str(e)}",
        )


def save_traffic_config(config: Dict[str, Any]) -> None:
    """Save traffic routing configuration."""
    config_path = get_traffic_config_path()

    try:
        config["last_update"] = datetime.utcnow().isoformat()
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Traffic config saved: {config}")
    except Exception as e:
        logger.error(f"Error saving traffic config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save traffic configuration: {str(e)}",
        )


# ============================================================================
# Endpoints
# ============================================================================


@router.post(
    "/deploy-canary",
    response_model=DeploymentResponse,
    status_code=status.HTTP_200_OK,
    summary="Deploy Canary Models",
    description="""
    Deploy challenger models in canary mode with specified traffic percentage.
    
    This endpoint:
    1. Validates model URIs exist in MLflow
    2. Updates traffic routing configuration
    3. Triggers model loading in the prediction service
    
    Traffic is split between champion (production) and challenger (staging) models.
    """,
)
async def deploy_canary(
    request: CanaryDeploymentRequest,
    current_user: Dict = Depends(get_current_admin_user),
):
    """Deploy models in canary mode."""
    try:
        logger.info(
            f"Admin {current_user['username']} deploying canary: "
            f"{request.traffic_percentage}% traffic to {len(request.model_uris)} models"
        )

        # Import here to avoid circular dependencies
        from scripts.deploy_canary import deploy_canary_models

        # Execute deployment
        result = deploy_canary_models(
            model_uris=request.model_uris,
            traffic_pct=request.traffic_percentage,
            mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        )

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Canary deployment failed"),
            )

        return DeploymentResponse(
            status="success",
            message=f"Canary deployed with {request.traffic_percentage}% traffic",
            details={
                "model_uris": request.model_uris,
                "traffic_percentage": request.traffic_percentage,
                "models_loaded": result.get("models_loaded", len(request.model_uris)),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Canary deployment error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Canary deployment failed: {str(e)}",
        )


@router.post(
    "/promote-to-production",
    response_model=DeploymentResponse,
    status_code=status.HTTP_200_OK,
    summary="Promote Models to Production",
    description="""
    Promote challenger models to production (100% traffic).
    
    This endpoint:
    1. Transitions models from Staging → Production in MLflow
    2. Updates traffic routing to 100% new models
    3. Archives old production models
    4. Reloads prediction service with new models
    """,
)
async def promote_to_production(
    request: PromoteToProductionRequest,
    current_user: Dict = Depends(get_current_admin_user),
):
    """Promote staged models to production."""
    try:
        logger.info(
            f"Admin {current_user['username']} promoting {len(request.model_uris)} models to production"
        )

        # Import here to avoid circular dependencies
        from scripts.promote_to_production import promote_models

        # Execute promotion
        result = promote_models(
            model_uris=request.model_uris,
            mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        )

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Promotion to production failed"),
            )

        return DeploymentResponse(
            status="success",
            message="Models promoted to production",
            details={
                "model_uris": request.model_uris,
                "models_promoted": result.get(
                    "models_promoted", len(request.model_uris)
                ),
                "production_versions": result.get("production_versions", {}),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Promotion error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Promotion to production failed: {str(e)}",
        )


@router.post(
    "/rollback-deployment",
    response_model=DeploymentResponse,
    status_code=status.HTTP_200_OK,
    summary="Rollback to Champion Models",
    description="""
    Rollback deployment to champion models (100% traffic).
    
    This endpoint:
    1. Reverts traffic routing to 100% champion models
    2. Removes challenger models from routing
    3. Reloads prediction service with champion models
    
    Use this when canary metrics are unhealthy.
    """,
)
async def rollback_deployment(
    current_user: Dict = Depends(get_current_admin_user),
):
    """Rollback to champion models."""
    try:
        logger.info(f"Admin {current_user['username']} triggering deployment rollback")

        # Import here to avoid circular dependencies
        from scripts.rollback_deployment import rollback_to_champion

        # Execute rollback
        result = rollback_to_champion()

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Rollback failed"),
            )

        return DeploymentResponse(
            status="success",
            message="Rolled back to champion models",
            details={
                "champion_models": result.get("champion_models", []),
                "traffic_percentage": 100,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rollback error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rollback failed: {str(e)}",
        )


@router.get(
    "/deployment-status",
    response_model=DeploymentStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Deployment Status",
    description="""
    Get current deployment configuration and status.
    
    Returns:
    - Current deployment mode (production_only or canary_active)
    - Traffic split percentages
    - Champion and challenger model lists
    - Last configuration update timestamp
    """,
)
async def get_deployment_status(
    current_user: Dict = Depends(get_current_admin_user),
):
    """Get current deployment configuration."""
    try:
        config = load_traffic_config()

        canary_pct = config.get("canary_percentage", 0)
        deployment_mode = "canary_active" if canary_pct > 0 else "production_only"

        return DeploymentStatusResponse(
            deployment_mode=deployment_mode,
            canary_percentage=canary_pct,
            champion_models=config.get("champion_models", []),
            challenger_models=config.get("challenger_models", []),
            last_update=config.get("last_update"),
            config=config,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deployment status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get deployment status: {str(e)}",
        )
