"""
Unit tests for user management API routes.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException, status

from src.api.routes.users import (
    create_user,
    list_users,
    get_user,
    update_user,
    delete_user,
    toggle_user_activation,
    update_user_role,
    reset_user_password,
)
from src.models.schemas import (
    UserCreateRequest,
    UserUpdateRequest,
    UserActivationRequest,
    UserRoleUpdateRequest,
    UserPasswordResetRequest,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def mock_admin_user():
    """Mock admin user for authentication."""
    return {
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
        "password_hash": "$2b$12$hashed_password",
    }


@pytest.fixture
def mock_regular_user():
    """Mock regular user."""
    return {
        "id": 2,
        "username": "john_analyst",
        "email": "john@company.com",
        "first_name": "John",
        "last_name": "Doe",
        "role": "analyst",
        "department": "Fraud Detection",
        "is_active": True,
        "is_verified": True,
        "created_at": "2025-01-15T10:30:00Z",
        "last_login": "2025-01-20T14:25:00Z",
        "password_hash": "$2b$12$hashed_password",
    }


@pytest.fixture
def valid_user_create_data():
    """Valid user creation data."""
    return UserCreateRequest(
        username="new_analyst",
        email="new@company.com",
        password="SecurePass123!",
        first_name="New",
        last_name="User",
        role="analyst",
        department="Security",
        is_active=True,
        is_verified=False,
    )


# ==============================================================================
# CREATE USER TESTS
# ==============================================================================


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
@patch("src.api.routes.users.auth_service")
async def test_create_user_success(
    mock_auth_service,
    mock_user_db_service,
    valid_user_create_data,
    mock_admin_user,
    mock_regular_user,
):
    """Test successful user creation."""
    # Setup mocks
    mock_user_db_service.get_user_by_username.return_value = None
    mock_user_db_service.get_user_by_email.return_value = None
    mock_auth_service.get_password_hash.return_value = "$2b$12$hashed_password"
    mock_user_db_service.create_user.return_value = mock_regular_user

    # Call endpoint
    response = await create_user(valid_user_create_data, mock_admin_user)

    # Assertions
    assert response.success is True
    assert "created successfully" in response.message
    assert response.user is not None
    assert response.user.username == mock_regular_user["username"]
    mock_user_db_service.create_user.assert_called_once()


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_create_user_duplicate_username(
    mock_user_db_service,
    valid_user_create_data,
    mock_admin_user,
    mock_regular_user,
):
    """Test user creation with duplicate username."""
    # Setup mock - username exists
    mock_user_db_service.get_user_by_username.return_value = mock_regular_user

    # Call endpoint and expect exception
    with pytest.raises(HTTPException) as exc_info:
        await create_user(valid_user_create_data, mock_admin_user)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "already exists" in exc_info.value.detail


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_create_user_duplicate_email(
    mock_user_db_service,
    valid_user_create_data,
    mock_admin_user,
    mock_regular_user,
):
    """Test user creation with duplicate email."""
    # Setup mock - email exists
    mock_user_db_service.get_user_by_username.return_value = None
    mock_user_db_service.get_user_by_email.return_value = mock_regular_user

    # Call endpoint and expect exception
    with pytest.raises(HTTPException) as exc_info:
        await create_user(valid_user_create_data, mock_admin_user)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "already exists" in exc_info.value.detail


# ==============================================================================
# LIST USERS TESTS
# ==============================================================================


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_list_users_success(
    mock_user_db_service, mock_admin_user, mock_regular_user
):
    """Test successful user listing."""
    # Setup mock
    users_list = [mock_admin_user, mock_regular_user]
    mock_user_db_service.list_users.return_value = (users_list, 2)

    # Call endpoint
    response = await list_users(mock_admin_user, limit=50, offset=0)

    # Assertions
    assert response.total_count == 2
    assert len(response.users) == 2
    assert response.limit == 50
    assert response.offset == 0


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_list_users_with_filters(
    mock_user_db_service, mock_admin_user, mock_regular_user
):
    """Test user listing with role filter."""
    # Setup mock
    mock_user_db_service.list_users.return_value = ([mock_regular_user], 1)

    # Call endpoint with role filter
    response = await list_users(
        mock_admin_user, limit=50, offset=0, role="analyst", is_active=True
    )

    # Assertions
    assert response.total_count == 1
    assert len(response.users) == 1
    mock_user_db_service.list_users.assert_called_once_with(
        limit=50, offset=0, role="analyst", is_active=True
    )


# ==============================================================================
# GET USER TESTS
# ==============================================================================


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_get_user_success(
    mock_user_db_service, mock_admin_user, mock_regular_user
):
    """Test successful user retrieval."""
    # Setup mock
    mock_user_db_service.get_user_by_id.return_value = mock_regular_user

    # Call endpoint
    response = await get_user(2, mock_admin_user)

    # Assertions
    assert response.id == mock_regular_user["id"]
    assert response.username == mock_regular_user["username"]
    mock_user_db_service.get_user_by_id.assert_called_once_with(2)


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_get_user_not_found(mock_user_db_service, mock_admin_user):
    """Test user retrieval when user doesn't exist."""
    # Setup mock - user not found
    mock_user_db_service.get_user_by_id.return_value = None

    # Call endpoint and expect exception
    with pytest.raises(HTTPException) as exc_info:
        await get_user(999, mock_admin_user)

    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in exc_info.value.detail


# ==============================================================================
# UPDATE USER TESTS
# ==============================================================================


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_update_user_success(
    mock_user_db_service, mock_admin_user, mock_regular_user
):
    """Test successful user update."""
    # Setup mock
    mock_user_db_service.get_user_by_id.return_value = mock_regular_user
    mock_user_db_service.get_user_by_email.return_value = None
    updated_user = mock_regular_user.copy()
    updated_user["department"] = "Risk Management"
    mock_user_db_service.update_user.return_value = updated_user

    # Create update request
    update_data = UserUpdateRequest(department="Risk Management")

    # Call endpoint
    response = await update_user(2, update_data, mock_admin_user)

    # Assertions
    assert response.success is True
    assert "updated successfully" in response.message
    assert response.user.department == "Risk Management"


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_update_user_duplicate_email(
    mock_user_db_service, mock_admin_user, mock_regular_user
):
    """Test user update with duplicate email."""
    # Setup mock
    mock_user_db_service.get_user_by_id.return_value = mock_regular_user
    other_user = mock_regular_user.copy()
    other_user["id"] = 3
    other_user["email"] = "existing@company.com"
    mock_user_db_service.get_user_by_email.return_value = other_user

    # Create update request
    update_data = UserUpdateRequest(email="existing@company.com")

    # Call endpoint and expect exception
    with pytest.raises(HTTPException) as exc_info:
        await update_user(2, update_data, mock_admin_user)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "already exists" in exc_info.value.detail


# ==============================================================================
# DELETE USER TESTS
# ==============================================================================


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_delete_user_success(
    mock_user_db_service, mock_admin_user, mock_regular_user
):
    """Test successful user deletion."""
    # Setup mock
    mock_user_db_service.get_user_by_id.return_value = mock_regular_user
    mock_user_db_service.delete_user.return_value = True

    # Call endpoint
    response = await delete_user(2, mock_admin_user)

    # Assertions
    assert response.success is True
    assert "deleted successfully" in response.message
    assert response.user is None
    mock_user_db_service.delete_user.assert_called_once_with(2)


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_delete_user_own_account(mock_user_db_service, mock_admin_user):
    """Test that admin cannot delete their own account."""
    # Setup mock - trying to delete own account
    mock_user_db_service.get_user_by_id.return_value = mock_admin_user

    # Call endpoint and expect exception
    with pytest.raises(HTTPException) as exc_info:
        await delete_user(1, mock_admin_user)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "Cannot delete your own account" in exc_info.value.detail


# ==============================================================================
# ACTIVATION TESTS
# ==============================================================================


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_toggle_user_activation_success(
    mock_user_db_service, mock_admin_user, mock_regular_user
):
    """Test successful user activation/deactivation."""
    # Setup mock
    mock_user_db_service.get_user_by_id.return_value = mock_regular_user
    deactivated_user = mock_regular_user.copy()
    deactivated_user["is_active"] = False
    mock_user_db_service.update_user.return_value = deactivated_user

    # Create activation request
    activation_data = UserActivationRequest(is_active=False)

    # Call endpoint
    response = await toggle_user_activation(2, activation_data, mock_admin_user)

    # Assertions
    assert response.success is True
    assert "deactivated successfully" in response.message
    assert response.user.is_active is False


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_toggle_user_activation_own_account(
    mock_user_db_service, mock_admin_user
):
    """Test that admin cannot deactivate their own account."""
    # Setup mock - trying to deactivate own account
    mock_user_db_service.get_user_by_id.return_value = mock_admin_user

    # Create deactivation request
    activation_data = UserActivationRequest(is_active=False)

    # Call endpoint and expect exception
    with pytest.raises(HTTPException) as exc_info:
        await toggle_user_activation(1, activation_data, mock_admin_user)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "Cannot deactivate your own account" in exc_info.value.detail


# ==============================================================================
# ROLE UPDATE TESTS
# ==============================================================================


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_update_user_role_success(
    mock_user_db_service, mock_admin_user, mock_regular_user
):
    """Test successful user role update."""
    # Setup mock
    mock_user_db_service.get_user_by_id.return_value = mock_regular_user
    mock_user_db_service.list_users.return_value = ([mock_admin_user], 1)
    updated_user = mock_regular_user.copy()
    updated_user["role"] = "admin"
    mock_user_db_service.update_user.return_value = updated_user

    # Create role update request
    role_data = UserRoleUpdateRequest(role="admin")

    # Call endpoint
    response = await update_user_role(2, role_data, mock_admin_user)

    # Assertions
    assert response.success is True
    assert "role changed to 'admin'" in response.message
    assert response.user.role == "admin"


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_update_user_role_own_account(mock_user_db_service, mock_admin_user):
    """Test that admin cannot change their own role."""
    # Setup mock - trying to change own role
    mock_user_db_service.get_user_by_id.return_value = mock_admin_user

    # Create role update request
    role_data = UserRoleUpdateRequest(role="analyst")

    # Call endpoint and expect exception
    with pytest.raises(HTTPException) as exc_info:
        await update_user_role(1, role_data, mock_admin_user)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "Cannot change your own role" in exc_info.value.detail


# ==============================================================================
# PASSWORD RESET TESTS
# ==============================================================================


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
@patch("src.api.routes.users.auth_service")
async def test_reset_user_password_success(
    mock_auth_service,
    mock_user_db_service,
    mock_admin_user,
    mock_regular_user,
):
    """Test successful password reset."""
    # Setup mock
    mock_user_db_service.get_user_by_id.return_value = mock_regular_user
    mock_auth_service.get_password_hash.return_value = "$2b$12$new_hashed_password"
    mock_user_db_service.update_user.return_value = mock_regular_user

    # Create password reset request
    password_data = UserPasswordResetRequest(new_password="NewSecurePass456!")

    # Call endpoint
    response = await reset_user_password(2, password_data, mock_admin_user)

    # Assertions
    assert response.success is True
    assert "Password reset successfully" in response.message
    mock_auth_service.get_password_hash.assert_called_once_with("NewSecurePass456!")


@pytest.mark.asyncio
@patch("src.api.routes.users.user_db_service")
async def test_reset_password_user_not_found(mock_user_db_service, mock_admin_user):
    """Test password reset for non-existent user."""
    # Setup mock - user not found
    mock_user_db_service.get_user_by_id.return_value = None

    # Create password reset request
    password_data = UserPasswordResetRequest(new_password="NewSecurePass456!")

    # Call endpoint and expect exception
    with pytest.raises(HTTPException) as exc_info:
        await reset_user_password(999, password_data, mock_admin_user)

    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in exc_info.value.detail
