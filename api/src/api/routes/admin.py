"""
Admin API routes.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from ...api.dependencies import get_cache_service, get_prediction_service
from ...config import get_logger, settings
from ...models import ModelVersionResponse
from ...services import CacheService, PredictionService
from ...utils import UnauthorizedException
from fastapi.responses import HTMLResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


def verify_admin_token(x_admin_token: Optional[str] = Header(None)) -> str:
    """
    Verify admin token from header.

    Args:
        x_admin_token: Admin token from header

    Returns:
        Validated token

    Raises:
        HTTPException: If token is invalid
    """
    if not x_admin_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin token required"
        )

    # In production, validate against secure token storage
    if x_admin_token != settings.admin_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token"
        )

    return x_admin_token


@router.post(
    "/reload-model",
    status_code=status.HTTP_200_OK,
    summary="Reload ML model",
    description="Reload the ML model from disk (admin only)",
)
async def reload_model(
    prediction_service: PredictionService = Depends(get_prediction_service),
    admin_token: str = Depends(verify_admin_token),
):
    """
    Reload the ML model.

    Args:
        prediction_service: Prediction service dependency
        admin_token: Validated admin token

    Returns:
        Reload status
    """
    try:
        logger.info("Reloading model (admin request)")

        # Reload model
        prediction_service.model.load_models()

        # Get new model info
        model_info = prediction_service.get_model_info()

        logger.info("Model reloaded successfully")

        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_info": model_info,
        }

    except Exception as e:
        logger.error(f"Failed to reload model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "E002",
                "message": "Failed to reload model",
                "details": {"error": str(e)},
            },
        )


@router.get(
    "/model-version",
    response_model=ModelVersionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get model version",
    description="Get information about the loaded model (admin only)",
)
async def get_model_version(
    prediction_service: PredictionService = Depends(get_prediction_service),
    admin_token: str = Depends(verify_admin_token),
):
    """
    Get model version and information.

    Args:
        prediction_service: Prediction service dependency
        admin_token: Validated admin token

    Returns:
        Model version information
    """
    try:
        model_info = prediction_service.get_model_info()

        return ModelVersionResponse(
            version=settings.model_version,
            models=model_info.get("models", {}),
            loaded_at=model_info.get("loaded_at", ""),
        )

    except Exception as e:
        logger.error(f"Failed to get model version: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "E999",
                "message": "Failed to get model version",
                "details": {"error": str(e)},
            },
        )


@router.post(
    "/clear-cache",
    status_code=status.HTTP_200_OK,
    summary="Clear prediction cache",
    description="Clear all cached predictions (admin only)",
)
async def clear_cache(
    cache_service: CacheService = Depends(get_cache_service),
    admin_token: str = Depends(verify_admin_token),
):
    """
    Clear all cached predictions.

    Args:
        cache_service: Cache service dependency
        admin_token: Validated admin token

    Returns:
        Clear cache status
    """
    try:
        logger.info("Clearing cache (admin request)")

        success = await cache_service.clear_all_cache()

        if success:
            logger.info("Cache cleared successfully")
            return {"status": "success", "message": "Cache cleared successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error_code": "E005", "message": "Failed to clear cache"},
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "E005",
                "message": "Failed to clear cache",
                "details": {"error": str(e)},
            },
        )


# Endpoint for documentation redoc of FastAPI Admin routes
@router.get("/docs", include_in_schema=False)
async def get_admin_docs(
    request: Request,
    token: str,  # Query parameter
):
    """
    Access admin documentation with token in URL.

    Usage: GET /admin/docs?token=YOUR_ADMIN_TOKEN

    This returns the native FastAPI Redoc documentation.
    """
    from fastapi.openapi.docs import get_redoc_html

    # Verify token
    if token != settings.admin_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token"
        )

    # Get OpenAPI URL
    openapi_url = request.app.openapi_url
    if not openapi_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="OpenAPI schema is disabled"
        )

    # Return native FastAPI Redoc HTML
    return get_redoc_html(
        openapi_url=openapi_url,
        title="Admin API Documentation - Fraud Detection",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )


@router.get("/swagger", include_in_schema=False)
async def get_admin_swagger(
    request: Request,
    token: str,  # Query parameter
):
    """
    Access admin Swagger UI documentation with token in URL.

    Usage: GET /admin/swagger?token=YOUR_ADMIN_TOKEN

    This returns the native FastAPI Swagger UI for interactive API testing.
    """
    from fastapi.openapi.docs import get_swagger_ui_html

    # Verify token
    if token != settings.admin_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token"
        )

    # Get OpenAPI URL
    openapi_url = request.app.openapi_url
    if not openapi_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="OpenAPI schema is disabled"
        )

    # Return native FastAPI Swagger UI HTML
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title="Admin API Documentation - Fraud Detection (Swagger UI)",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )
