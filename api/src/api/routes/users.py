"""
User Management API routes (Admin only).

This module provides comprehensive user management endpoints for administrators,
including CRUD operations, role management, and user activation/deactivation.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...config import get_logger
from ...models.schemas import (
    UserActivationRequest,
    UserCreateRequest,
    UserDetailResponse,
    UserListResponse,
    UserOperationResponse,
    UserPasswordResetRequest,
    UserRoleUpdateRequest,
    UserUpdateRequest,
)
from ...services.auth_service import auth_service
from ...services.user_database_service import user_db_service
from .auth import get_current_admin_user

logger = get_logger(__name__)

router = APIRouter(prefix="/admin/users", tags=["user-management"])


@router.post(
    "",
    response_model=UserOperationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description="Create a new user account. Requires admin privileges.",
)
async def create_user(
    user_data: UserCreateRequest,
    current_user: dict = Depends(get_current_admin_user),
):
    """
    Create a new user account.

    **Permissions**: Admin only

    **Request Body**:
    - username: Unique username (3-50 chars, alphanumeric + underscore/hyphen)
    - email: Valid email address
    - password: Strong password (min 8 chars)
    - first_name: Optional first name
    - last_name: Optional last name
    - role: User role (admin, analyst, viewer) - default: analyst
    - department: Optional department
    - is_active: Active status - default: True
    - is_verified: Email verified status - default: False

    **Returns**: Created user details and success message

    **Errors**:
    - 400: User already exists or validation error
    - 401: Not authenticated
    - 403: Not admin user
    - 500: Database error
    """
    try:
        # Check if username already exists
        existing_user = user_db_service.get_user_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Username '{user_data.username}' already exists",
            )

        # Check if email already exists
        existing_email = user_db_service.get_user_by_email(user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Email '{user_data.email}' already exists",
            )

        # Hash password
        password_hash = auth_service.get_password_hash(user_data.password)

        # Create user in database
        user = user_db_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password_hash=password_hash,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            role=user_data.role,
            department=user_data.department,
            is_active=user_data.is_active,
            is_verified=user_data.is_verified,
        )

        logger.info(
            f"User '{user_data.username}' created by admin '{current_user['username']}'"
        )

        return UserOperationResponse(
            success=True,
            message=f"User '{user_data.username}' created successfully",
            user=UserDetailResponse(**user),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}",
        )


@router.get(
    "",
    response_model=UserListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all users",
    description="Get a paginated list of all users. Requires admin privileges.",
)
async def list_users(
    current_user: dict = Depends(get_current_admin_user),
    limit: int = Query(
        default=50, ge=1, le=100, description="Number of users per page"
    ),
    offset: int = Query(default=0, ge=0, description="Number of users to skip"),
    role: Optional[str] = Query(
        default=None, description="Filter by role (admin, analyst, viewer)"
    ),
    is_active: Optional[bool] = Query(
        default=None, description="Filter by active status"
    ),
):
    """
    Get a paginated list of all users.

    **Permissions**: Admin only

    **Query Parameters**:
    - limit: Number of users per page (1-100, default: 50)
    - offset: Number of users to skip (default: 0)
    - role: Filter by role (optional)
    - is_active: Filter by active status (optional)

    **Returns**: List of users with pagination info

    **Errors**:
    - 401: Not authenticated
    - 403: Not admin user
    - 500: Database error
    """
    try:
        users, total_count = user_db_service.list_users(
            limit=limit, offset=offset, role=role, is_active=is_active
        )

        logger.info(
            f"Admin '{current_user['username']}' listed {len(users)} users "
            f"(total: {total_count})"
        )

        return UserListResponse(
            users=[UserDetailResponse(**user) for user in users],
            total_count=total_count,
            limit=limit,
            offset=offset,
        )

    except Exception as e:
        logger.error(f"Error listing users: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {str(e)}",
        )


@router.get(
    "/{user_id}",
    response_model=UserDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get user by ID",
    description="Get detailed information about a specific user. Requires admin privileges.",
)
async def get_user(
    user_id: int,
    current_user: dict = Depends(get_current_admin_user),
):
    """
    Get detailed information about a specific user.

    **Permissions**: Admin only

    **Path Parameters**:
    - user_id: ID of the user to retrieve

    **Returns**: Detailed user information

    **Errors**:
    - 401: Not authenticated
    - 403: Not admin user
    - 404: User not found
    - 500: Database error
    """
    try:
        user = user_db_service.get_user_by_id(user_id)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )

        logger.info(
            f"Admin '{current_user['username']}' retrieved user '{user['username']}'"
        )

        return UserDetailResponse(**user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user: {str(e)}",
        )


@router.put(
    "/{user_id}",
    response_model=UserOperationResponse,
    status_code=status.HTTP_200_OK,
    summary="Update user",
    description="Update user information. Requires admin privileges.",
)
async def update_user(
    user_id: int,
    user_data: UserUpdateRequest,
    current_user: dict = Depends(get_current_admin_user),
):
    """
    Update user information.

    **Permissions**: Admin only

    **Path Parameters**:
    - user_id: ID of the user to update

    **Request Body**: Fields to update (all optional)
    - email: New email address
    - first_name: New first name
    - last_name: New last name
    - role: New role (admin, analyst, viewer)
    - department: New department
    - is_active: New active status
    - is_verified: New verified status

    **Returns**: Updated user details and success message

    **Errors**:
    - 400: Email already exists or validation error
    - 401: Not authenticated
    - 403: Not admin user
    - 404: User not found
    - 500: Database error
    """
    try:
        # Check if user exists
        existing_user = user_db_service.get_user_by_id(user_id)
        if not existing_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )

        # Check if email is being updated and already exists
        if user_data.email and user_data.email != existing_user["email"]:
            email_exists = user_db_service.get_user_by_email(user_data.email)
            if email_exists:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Email '{user_data.email}' already exists",
                )

        # Update user
        updated_user = user_db_service.update_user(
            user_id, user_data.model_dump(exclude_unset=True)
        )

        logger.info(
            f"User '{existing_user['username']}' updated by admin '{current_user['username']}'"
        )

        return UserOperationResponse(
            success=True,
            message=f"User '{existing_user['username']}' updated successfully",
            user=UserDetailResponse(**updated_user),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}",
        )


@router.delete(
    "/{user_id}",
    response_model=UserOperationResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete user",
    description="Delete a user account. Requires admin privileges.",
)
async def delete_user(
    user_id: int,
    current_user: dict = Depends(get_current_admin_user),
):
    """
    Delete a user account.

    **Permissions**: Admin only

    **Path Parameters**:
    - user_id: ID of the user to delete

    **Returns**: Success message

    **Errors**:
    - 400: Cannot delete own account or last admin
    - 401: Not authenticated
    - 403: Not admin user
    - 404: User not found
    - 500: Database error
    """
    try:
        # Check if user exists
        user = user_db_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )

        # Prevent deleting own account
        if user["username"] == current_user["username"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account",
            )

        # Prevent deleting the last admin
        if user["role"] == "admin":
            admin_count = len(
                [
                    u
                    for u, _ in user_db_service.list_users(role="admin")
                    if isinstance(u, list)
                ]
            )
            if admin_count <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete the last admin user",
                )

        # Delete user
        user_db_service.delete_user(user_id)

        logger.warning(
            f"User '{user['username']}' (ID: {user_id}) deleted by admin '{current_user['username']}'"
        )

        return UserOperationResponse(
            success=True,
            message=f"User '{user['username']}' deleted successfully",
            user=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}",
        )


@router.patch(
    "/{user_id}/activate",
    response_model=UserOperationResponse,
    status_code=status.HTTP_200_OK,
    summary="Activate/deactivate user",
    description="Activate or deactivate a user account. Requires admin privileges.",
)
async def toggle_user_activation(
    user_id: int,
    activation_data: UserActivationRequest,
    current_user: dict = Depends(get_current_admin_user),
):
    """
    Activate or deactivate a user account.

    **Permissions**: Admin only

    **Path Parameters**:
    - user_id: ID of the user to activate/deactivate

    **Request Body**:
    - is_active: True to activate, False to deactivate

    **Returns**: Updated user details and success message

    **Errors**:
    - 400: Cannot deactivate own account
    - 401: Not authenticated
    - 403: Not admin user
    - 404: User not found
    - 500: Database error
    """
    try:
        # Check if user exists
        user = user_db_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )

        # Prevent deactivating own account
        if (
            user["username"] == current_user["username"]
            and not activation_data.is_active
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate your own account",
            )

        # Update user activation status
        updated_user = user_db_service.update_user(
            user_id, {"is_active": activation_data.is_active}
        )

        action = "activated" if activation_data.is_active else "deactivated"
        logger.info(
            f"User '{user['username']}' {action} by admin '{current_user['username']}'"
        )

        return UserOperationResponse(
            success=True,
            message=f"User '{user['username']}' {action} successfully",
            user=UserDetailResponse(**updated_user),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error toggling activation for user {user_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user activation: {str(e)}",
        )


@router.patch(
    "/{user_id}/role",
    response_model=UserOperationResponse,
    status_code=status.HTTP_200_OK,
    summary="Change user role",
    description="Change a user's role. Requires admin privileges.",
)
async def update_user_role(
    user_id: int,
    role_data: UserRoleUpdateRequest,
    current_user: dict = Depends(get_current_admin_user),
):
    """
    Change a user's role.

    **Permissions**: Admin only

    **Path Parameters**:
    - user_id: ID of the user

    **Request Body**:
    - role: New role (admin, analyst, viewer)

    **Returns**: Updated user details and success message

    **Errors**:
    - 400: Cannot change own role or remove last admin
    - 401: Not authenticated
    - 403: Not admin user
    - 404: User not found
    - 500: Database error
    """
    try:
        # Check if user exists
        user = user_db_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )

        # Prevent changing own role
        if user["username"] == current_user["username"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot change your own role",
            )

        # Prevent removing the last admin
        if user["role"] == "admin" and role_data.role != "admin":
            admin_users, _ = user_db_service.list_users(role="admin")
            if len(admin_users) <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot remove admin role from the last admin user",
                )

        # Update user role
        updated_user = user_db_service.update_user(user_id, {"role": role_data.role})

        logger.info(
            f"User '{user['username']}' role changed to '{role_data.role}' "
            f"by admin '{current_user['username']}'"
        )

        return UserOperationResponse(
            success=True,
            message=f"User '{user['username']}' role changed to '{role_data.role}' successfully",
            user=UserDetailResponse(**updated_user),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating role for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user role: {str(e)}",
        )


@router.patch(
    "/{user_id}/password",
    response_model=UserOperationResponse,
    status_code=status.HTTP_200_OK,
    summary="Reset user password",
    description="Reset a user's password. Requires admin privileges.",
)
async def reset_user_password(
    user_id: int,
    password_data: UserPasswordResetRequest,
    current_user: dict = Depends(get_current_admin_user),
):
    """
    Reset a user's password.

    **Permissions**: Admin only

    **Path Parameters**:
    - user_id: ID of the user

    **Request Body**:
    - new_password: New password (min 8 characters)

    **Returns**: Success message

    **Errors**:
    - 401: Not authenticated
    - 403: Not admin user
    - 404: User not found
    - 500: Database error
    """
    try:
        # Check if user exists
        user = user_db_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )

        # Hash new password
        password_hash = auth_service.get_password_hash(password_data.new_password)

        # Update password in database
        user_db_service.update_user(user_id, {"password_hash": password_hash})

        logger.warning(
            f"Password reset for user '{user['username']}' by admin '{current_user['username']}'"
        )

        return UserOperationResponse(
            success=True,
            message=f"Password reset successfully for user '{user['username']}'",
            user=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting password for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset password: {str(e)}",
        )
