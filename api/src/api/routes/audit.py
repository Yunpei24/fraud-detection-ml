"""
Audit trails API routes for compliance reporting.
"""
import time
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...api.routes.auth import get_current_admin_user
from ...config import get_logger
from ...models import (
    AuditLogEntry,
    AuditLogsResponse,
    AuditLogSummaryResponse,
    AuditQueryRequest,
    ErrorResponse,
)
from ...services import DatabaseService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/audit", tags=["audit-trails"])


@router.post(
    "/logs",
    response_model=AuditLogsResponse,
    status_code=status.HTTP_200_OK,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Query audit logs",
    description="Query audit logs with optional filtering for compliance reporting",
    dependencies=[Depends(get_current_admin_user)],
)
async def query_audit_logs(
    request: AuditQueryRequest, database_service: DatabaseService = Depends()
):
    """
    Query audit logs with filtering options.

    Requires admin role.

    Args:
        request: Audit query parameters
        database_service: Database service dependency

    Returns:
        Filtered audit logs
    """
    start_time = time.time()

    try:
        # Parse dates if provided
        start_date = None
        end_date = None
        if request.start_date:
            try:
                start_date = datetime.fromisoformat(
                    request.start_date.replace("Z", "+00:00")
                )
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error_code": "E901",
                        "message": "Invalid start_date format. Use ISO format (e.g., 2025-01-01T00:00:00Z)",
                        "details": {"provided": request.start_date},
                    },
                )

        if request.end_date:
            try:
                end_date = datetime.fromisoformat(
                    request.end_date.replace("Z", "+00:00")
                )
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error_code": "E902",
                        "message": "Invalid end_date format. Use ISO format (e.g., 2025-01-31T23:59:59Z)",
                        "details": {"provided": request.end_date},
                    },
                )

        # Query audit logs
        logs = await database_service.get_audit_logs(
            transaction_id=request.transaction_id,
            user_id=request.user_id,
            action=request.action,
            limit=request.limit,
            offset=request.offset,
            start_date=start_date,
            end_date=end_date,
        )

        # Get total count (simplified - in production, you'd want a separate count query)
        total_count = (
            len(logs) + request.offset
            if len(logs) == request.limit
            else len(logs) + request.offset
        )

        # Convert to response format
        log_entries = [AuditLogEntry(**log) for log in logs]

        logger.info(f"Retrieved {len(logs)} audit log entries")

        return AuditLogsResponse(
            logs=log_entries,
            total_count=total_count,
            limit=request.limit,
            offset=request.offset,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to query audit logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "E900",
                "message": "Failed to query audit logs",
                "details": {"error": str(e)},
            },
        )


@router.get(
    "/summary",
    response_model=AuditLogSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get audit summary",
    description="Get audit log summary statistics for compliance reporting",
    dependencies=[Depends(get_current_admin_user)],
)
async def get_audit_summary(
    days: int = Query(
        default=30, ge=1, le=365, description="Number of days to look back"
    ),
    database_service: DatabaseService = Depends(),
):
    """
    Get audit log summary statistics.

    Requires admin role.

    Args:
        days: Number of days to look back
        database_service: Database service dependency

    Returns:
        Audit summary statistics
    """
    try:
        summary = await database_service.get_audit_log_summary(days=days)

        logger.info(f"Retrieved audit summary for last {days} days")

        return AuditLogSummaryResponse(**summary)

    except Exception as e:
        logger.error(f"Failed to get audit summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "E903",
                "message": "Failed to get audit summary",
                "details": {"error": str(e)},
            },
        )


@router.get(
    "/logs/{transaction_id}",
    response_model=AuditLogsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get transaction audit logs",
    description="Get all audit logs for a specific transaction",
    dependencies=[Depends(get_current_admin_user)],
)
async def get_transaction_audit_logs(
    transaction_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    database_service: DatabaseService = Depends(),
):
    """
    Get audit logs for a specific transaction.

    Requires admin role.

    Args:
        transaction_id: Transaction identifier
        limit: Maximum number of records to return
        database_service: Database service dependency

    Returns:
        Transaction audit logs
    """
    try:
        logs = await database_service.get_audit_logs(
            transaction_id=transaction_id, limit=limit, offset=0
        )

        log_entries = [AuditLogEntry(**log) for log in logs]

        logger.info(
            f"Retrieved {len(logs)} audit log entries for transaction {transaction_id}"
        )

        return AuditLogsResponse(
            logs=log_entries, total_count=len(logs), limit=limit, offset=0
        )

    except Exception as e:
        logger.error(f"Failed to get transaction audit logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "E904",
                "message": "Failed to get transaction audit logs",
                "details": {"transaction_id": transaction_id, "error": str(e)},
            },
        )
