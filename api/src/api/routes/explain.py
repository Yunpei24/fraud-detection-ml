"""
Model Explainability API routes for SHAP explanations and feature importance.
"""
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from ...api.routes.auth import get_current_analyst_user
from ...config import get_logger
from ...models import (ErrorResponse, ExplanationRequest,
                       FeatureImportanceResponse, SHAPExplanationResponse)
from ...services import PredictionService
from ...utils import InvalidInputException

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/explain", tags=["explainability"])


@router.post(
    "/shap",
    response_model=SHAPExplanationResponse,
    status_code=status.HTTP_200_OK,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Get SHAP explanation",
    description="Generate SHAP explanation for a transaction prediction",
    dependencies=[Depends(get_current_analyst_user)],
)
async def get_shap_explanation(
    request: ExplanationRequest, prediction_service: PredictionService = Depends()
):
    """
    Generate SHAP explanation for a transaction.

    Requires analyst or admin role.

    Args:
        request: Explanation request with transaction data
        prediction_service: Prediction service dependency

    Returns:
        SHAP explanation with feature contributions
    """
    start_time = time.time()
    transaction_id = request.transaction_id

    try:
        # Generate SHAP explanation
        explanation = await prediction_service.explain_prediction_shap(
            transaction_id=transaction_id,
            features=request.features,
            model_type=request.model_type or "ensemble",
            metadata=request.metadata,
        )

        # Add processing time
        explanation["processing_time"] = time.time() - start_time
        explanation["timestamp"] = time.time()

        logger.info(f"Generated SHAP explanation for transaction {transaction_id}")

        return SHAPExplanationResponse(**explanation)

    except InvalidInputException as e:
        logger.warning(
            f"Invalid input for SHAP explanation {transaction_id}: {e.message}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": e.error_code,
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error(
            f"SHAP explanation failed for transaction {transaction_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "E800",
                "message": "SHAP explanation generation failed",
                "details": {"error": str(e)},
            },
        )


@router.get(
    "/feature-importance/{model_type}",
    response_model=FeatureImportanceResponse,
    status_code=status.HTTP_200_OK,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Get feature importance",
    description="Get global feature importance for a specific model",
    dependencies=[Depends(get_current_analyst_user)],
)
async def get_feature_importance(
    model_type: str, prediction_service: PredictionService = Depends()
):
    """
    Get global feature importance for a model.

    Requires analyst or admin role.

    Args:
        model_type: Model type ('xgboost', 'neural_network', 'isolation_forest')
        prediction_service: Prediction service dependency

    Returns:
        Feature importance rankings
    """
    start_time = time.time()

    # Validate model type
    valid_types = ["xgboost", "neural_network", "isolation_forest"]
    if model_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "E801",
                "message": f"Invalid model type. Must be one of: {valid_types}",
                "details": {"provided": model_type, "valid_types": valid_types},
            },
        )

    try:
        # Get feature importance
        importance_data = await prediction_service.get_feature_importance(model_type)

        # Add processing time and timestamp
        importance_data["processing_time"] = time.time() - start_time
        importance_data["timestamp"] = time.time()

        logger.info(f"Retrieved feature importance for model {model_type}")

        return FeatureImportanceResponse(**importance_data)

    except Exception as e:
        logger.error(
            f"Feature importance retrieval failed for model {model_type}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "E802",
                "message": "Feature importance retrieval failed",
                "details": {"error": str(e), "model_type": model_type},
            },
        )


@router.get(
    "/models",
    response_model=List[str],
    status_code=status.HTTP_200_OK,
    summary="Get available models",
    description="Get list of models available for explanation",
    dependencies=[Depends(get_current_analyst_user)],
)
async def get_available_models():
    """
    Get list of models available for explanation.

    Requires analyst or admin role.

    Returns:
        List of available model types
    """
    return ["xgboost", "neural_network", "isolation_forest", "ensemble"]
