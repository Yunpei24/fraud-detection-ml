"""
Prediction API routes.
"""
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from ...api.dependencies import (get_cache_service, get_database_service,
                                 get_prediction_service, verify_api_key)
from ...config import get_logger
from ...models import (BatchPredictionResponse, BatchTransactionRequest,
                       ErrorResponse, PredictionResponse, TransactionRequest)
from ...services import CacheService, DatabaseService, PredictionService
from ...utils import (InvalidInputException, PredictionFailedException,
                      validate_batch_request)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Make fraud prediction",
    description="Predict if a transaction is fraudulent (requires API key authentication)",
)
async def predict(
    request: TransactionRequest,
    api_key: str = Depends(verify_api_key),
    prediction_service: PredictionService = Depends(get_prediction_service),
    cache_service: CacheService = Depends(get_cache_service),
    database_service: DatabaseService = Depends(get_database_service),
):
    """
    Make a fraud prediction for a single transaction.

    Requires valid API key authentication.

    Args:
        request: Transaction data
        api_key: Validated API key
        prediction_service: Prediction service dependency
        cache_service: Cache service dependency
        database_service: Database service dependency

    Returns:
        Prediction result
    """
    start_time = time.time()
    transaction_id = request.transaction_id

    try:
        # Check cache first
        cached_prediction = await cache_service.get_cached_prediction(transaction_id)
        if cached_prediction:
            logger.info(f"Returning cached prediction for {transaction_id}")
            return PredictionResponse(**cached_prediction)

        # Make prediction
        prediction = await prediction_service.predict_single(
            transaction_id=transaction_id,
            features=request.features,
            metadata=request.metadata,
        )

        # Cache the result
        await cache_service.set_prediction_cache(transaction_id, prediction)

        # Save to database (graceful degradation if unavailable)
        try:
            await database_service.save_prediction(transaction_id, prediction)
        except Exception as e:
            logger.warning(f"Failed to save prediction to database: {e}")

        # Log audit (graceful degradation if unavailable)
        try:
            await database_service.save_audit_log(
                transaction_id=transaction_id,
                action="prediction",
                details={
                    "processing_time": time.time() - start_time,
                    "prediction": prediction["prediction"],
                },
            )
        except Exception as e:
            logger.warning(f"Failed to save audit log: {e}")

        return PredictionResponse(**prediction)

    except InvalidInputException as e:
        logger.warning(f"Invalid input for transaction {transaction_id}: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": e.error_code,
                "message": e.message,
                "details": e.details,
            },
        )
    except PredictionFailedException as e:
        logger.error(f"Prediction failed for transaction {transaction_id}: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": e.error_code,
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error(
            f"Unexpected error for transaction {transaction_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "E999",
                "message": "Internal server error",
                "details": {"error": str(e)},
            },
        )


@router.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Batch fraud prediction",
    description="Predict fraud for multiple transactions (requires API key authentication)",
)
async def batch_predict(
    request: BatchTransactionRequest,
    api_key: str = Depends(verify_api_key),
    prediction_service: PredictionService = Depends(get_prediction_service),
    database_service: DatabaseService = Depends(get_database_service),
):
    """
    Make fraud predictions for multiple transactions.

    Requires valid API key authentication.

    Args:
        request: Batch transaction data
        api_key: Validated API key
        prediction_service: Prediction service dependency
        database_service: Database service dependency

    Returns:
        Batch prediction results
    """
    start_time = time.time()

    try:
        # Validate batch request
        validate_batch_request([t.model_dump() for t in request.transactions])

        # Make batch predictions
        results = await prediction_service.predict_batch(
            [t.model_dump() for t in request.transactions]
        )

        # Save predictions to database
        for prediction in results["predictions"]:
            try:
                await database_service.save_prediction(
                    prediction["transaction_id"], prediction
                )
            except Exception as e:
                logger.error(
                    f"Failed to save prediction for {prediction['transaction_id']}: {e}"
                )

        # Calculate summary
        processing_time = time.time() - start_time

        response_data = {
            "total_transactions": results["total_transactions"],
            "successful_predictions": results["successful_predictions"],
            "failed_predictions": results["failed_predictions"],
            "fraud_detected": results["fraud_detected"],
            "fraud_rate": results["fraud_rate"],
            "predictions": results["predictions"],
            "processing_time": processing_time,
            "avg_processing_time": results["avg_processing_time"],
        }

        return BatchPredictionResponse(**response_data)

    except InvalidInputException as e:
        logger.warning(f"Invalid batch request: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": e.error_code,
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "E999",
                "message": "Batch prediction failed",
                "details": {"error": str(e)},
            },
        )
