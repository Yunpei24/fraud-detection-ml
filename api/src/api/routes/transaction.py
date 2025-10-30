"""
Transaction update and analyst feedback endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from datetime import datetime

from ...models.schemas import (
    TransactionUpdateRequest,
    TransactionUpdateResponse,
    TransactionLabelHistoryResponse
)
from ...services.database_service import DatabaseService
from ...services.auth_service import AuthService
from ..dependencies import get_database_service
from ...services.auth_service import auth_service
from .auth import get_current_analyst_user

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.put(
    "/{transaction_id}/prediction",
    response_model=TransactionUpdateResponse,
    summary="Update transaction prediction with analyst feedback",
    description="Allows analysts to update transaction predictions with their expert judgment"
)
async def update_transaction_prediction(
    transaction_id: str,
    update_request: TransactionUpdateRequest,
    current_user: dict = Depends(get_current_analyst_user),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Update a transaction's prediction with analyst feedback.

    This endpoint allows analysts to override model predictions with their expert judgment.
    The original prediction is preserved in the audit trail.

    Args:
        transaction_id: The transaction identifier
        update_request: Update request containing analyst label and metadata
        current_user: Current authenticated analyst user

    Returns:
        TransactionUpdateResponse: Confirmation of the update

    Raises:
        HTTPException: If transaction not found or update fails
    """
    try:
        # Get current prediction for audit trail
        current_prediction = await db_service.get_prediction(transaction_id)
        if not current_prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction {transaction_id} not found"
            )

        # Save analyst label
        success = await db_service.save_analyst_label(
            transaction_id=transaction_id,
            predicted_label=current_prediction["prediction"],
            analyst_label=update_request.analyst_label,
            analyst_id=current_user["user_id"],
            confidence=update_request.confidence,
            notes=update_request.notes
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save analyst feedback"
            )

        # Log the action for audit
        await auth_service.log_audit_event(
            transaction_id=transaction_id,
            action="prediction_update",
            user_id=current_user["user_id"],
            details={
                "original_prediction": current_prediction["prediction"],
                "analyst_label": update_request.analyst_label,
                "confidence": update_request.confidence,
                "notes": update_request.notes
            }
        )

        return TransactionUpdateResponse(
            transaction_id=transaction_id,
            analyst_label=update_request.analyst_label,
            analyst_id=current_user["user_id"],
            updated_at=datetime.utcnow(),
            message="Transaction prediction updated successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update transaction prediction: {str(e)}"
        )


@router.get(
    "/{transaction_id}/labels",
    response_model=List[TransactionLabelHistoryResponse],
    summary="Get analyst label history for a transaction",
    description="Retrieve the history of analyst feedback labels for a specific transaction"
)
async def get_transaction_label_history(
    transaction_id: str,
    current_user: dict = Depends(get_current_analyst_user),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get the history of analyst labels for a transaction.

    This endpoint provides transparency into how analysts have labeled this transaction
    over time, including confidence levels and notes.

    Args:
        transaction_id: The transaction identifier
        current_user: Current authenticated user (analyst or admin)

    Returns:
        List[TransactionLabelHistoryResponse]: History of analyst labels

    Raises:
        HTTPException: If access denied or retrieval fails
    """
    try:
        # Get analyst labels for this transaction
        labels = await db_service.get_analyst_labels(
            transaction_id=transaction_id,
            limit=50  # Reasonable limit for history
        )

        # Convert to response format
        history = []
        for label in labels:
            history.append(TransactionLabelHistoryResponse(
                id=label["id"],
                transaction_id=label["transaction_id"],
                predicted_label=label["predicted_label"],
                analyst_label=label["analyst_label"],
                analyst_id=label["analyst_id"],
                confidence=label["confidence"],
                notes=label["notes"],
                created_at=datetime.fromisoformat(label["created_at"])
            ))

        return history

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve label history: {str(e)}"
        )


@router.get(
    "/labels",
    response_model=List[TransactionLabelHistoryResponse],
    summary="Get analyst labels with filtering",
    description="Retrieve analyst labels with optional filtering by transaction or analyst"
)
async def get_analyst_labels(
    transaction_id: Optional[str] = None,
    analyst_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: dict = Depends(get_current_analyst_user),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get analyst labels with optional filtering.

    This endpoint allows analysts and admins to query labels for analysis and reporting.

    Args:
        transaction_id: Filter by transaction ID
        analyst_id: Filter by analyst ID
        limit: Maximum number of records to return
        offset: Number of records to skip
        current_user: Current authenticated user

    Returns:
        List[TransactionLabelHistoryResponse]: Filtered analyst labels

    Raises:
        HTTPException: If access denied or retrieval fails
    """
    try:
        # Get analyst labels with filters
        labels = await db_service.get_analyst_labels(
            transaction_id=transaction_id,
            analyst_id=analyst_id,
            limit=limit,
            offset=offset
        )

        # Convert to response format
        result = []
        for label in labels:
            result.append(TransactionLabelHistoryResponse(
                id=label["id"],
                transaction_id=label["transaction_id"],
                predicted_label=label["predicted_label"],
                analyst_label=label["analyst_label"],
                analyst_id=label["analyst_id"],
                confidence=label["confidence"],
                notes=label["notes"],
                created_at=datetime.fromisoformat(label["created_at"])
            ))

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analyst labels: {str(e)}"
        )