"""
Integration tests for user management API endpoints.

These tests verify end-to-end functionality of user management endpoints
using the TestClient to simulate real HTTP requests.

NOTE: These tests require PostgreSQL to be running.
"""

import os
import uuid
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.auth_service import auth_service

client = TestClient(app)

# Skip all tests in this file if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION_TESTS", "false").lower() == "true",
    reason="Integration tests require PostgreSQL - skipped in CI without database",
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(scope="module")
def admin_token():
    """Get admin authentication token for testing."""
    # Login as admin user (created in schema.sql)
    response = client.post(
        "/auth/login",
        data={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture(scope="module")
def auth_headers(admin_token):
    """Get headers with admin authentication."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def test_user_data():
    """Test user data for creation with unique username."""
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
    return {
        "username": f"test_analyst_{unique_id}",
        "email": f"test.analyst.{unique_id}@company.com",
        "password": "TestPass123!",
        "first_name": "Test",
        "last_name": "Analyst",
        "role": "analyst",
        "department": "Testing",
        "is_active": True,
        "is_verified": False,
    }


# ==============================================================================
# AUTHENTICATION TESTS
# ==============================================================================


def test_auth_login_success():
    """Test successful admin login."""
    response = client.post(
        "/auth/login",
        data={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "user" in data


def test_auth_login_invalid_credentials():
    """Test login with invalid credentials."""
    response = client.post(
        "/auth/login",
        data={"username": "admin", "password": "wrong_password"},
    )
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]


def test_get_current_user(auth_headers):
    """Test getting current user information."""
    response = client.get("/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "admin"
    assert data["role"] == "admin"


# ==============================================================================
# USER CREATION TESTS
# ==============================================================================


def test_create_user_success(auth_headers, test_user_data):
    """Test successful user creation by admin."""
    response = client.post(
        "/admin/users",
        json=test_user_data,
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert "created successfully" in data["message"]
    assert data["user"]["username"] == test_user_data["username"]
    assert data["user"]["email"] == test_user_data["email"]
    assert data["user"]["role"] == test_user_data["role"]


def test_create_user_duplicate_username(auth_headers, test_user_data):
    """Test user creation with duplicate username."""
    # Create user first
    response1 = client.post(
        "/admin/users",
        json=test_user_data,
        headers=auth_headers,
    )
    assert response1.status_code == 201

    # Try to create same user again
    response2 = client.post(
        "/admin/users",
        json=test_user_data,
        headers=auth_headers,
    )
    assert response2.status_code == 400
    assert "already exists" in response2.json()["detail"]


def test_create_user_invalid_email(auth_headers):
    """Test user creation with invalid email."""
    invalid_user_data = {
        "username": "invalid_email_user",
        "email": "not_an_email",
        "password": "TestPass123!",
        "role": "analyst",
    }
    response = client.post(
        "/admin/users",
        json=invalid_user_data,
        headers=auth_headers,
    )
    assert response.status_code == 422  # Validation error


def test_create_user_without_auth():
    """Test user creation without authentication."""
    response = client.post(
        "/admin/users",
        json={"username": "test", "email": "test@test.com", "password": "pass"},
    )
    assert response.status_code == 401  # Unauthorized


# ==============================================================================
# USER LISTING TESTS
# ==============================================================================


def test_list_users_success(auth_headers):
    """Test listing all users."""
    response = client.get("/admin/users", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "users" in data
    assert "total_count" in data
    assert data["total_count"] >= 1  # At least admin user exists
    assert len(data["users"]) > 0


def test_list_users_with_pagination(auth_headers):
    """Test user listing with pagination."""
    response = client.get(
        "/admin/users?limit=5&offset=0",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["limit"] == 5
    assert data["offset"] == 0


def test_list_users_filter_by_role(auth_headers):
    """Test user listing filtered by role."""
    response = client.get(
        "/admin/users?role=admin",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    # All returned users should be admins
    for user in data["users"]:
        assert user["role"] == "admin"


# ==============================================================================
# GET USER TESTS
# ==============================================================================


def test_get_user_by_id_success(auth_headers):
    """Test getting specific user by ID."""
    # First, get list of users to get a valid ID
    list_response = client.get("/admin/users", headers=auth_headers)
    users = list_response.json()["users"]
    user_id = users[0]["id"]

    # Get specific user
    response = client.get(f"/admin/users/{user_id}", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == user_id
    assert "username" in data
    assert "email" in data


def test_get_user_not_found(auth_headers):
    """Test getting non-existent user."""
    response = client.get("/admin/users/9999", headers=auth_headers)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


# ==============================================================================
# UPDATE USER TESTS
# ==============================================================================


def test_update_user_success(auth_headers):
    """Test updating user information."""
    # Get a user to update
    list_response = client.get("/admin/users?role=analyst", headers=auth_headers)
    users = list_response.json()["users"]

    if not users:
        pytest.skip("No analyst users available for testing")

    user_id = users[0]["id"]

    # Update user
    update_data = {
        "department": "Updated Department",
        "first_name": "Updated",
    }
    response = client.put(
        f"/admin/users/{user_id}",
        json=update_data,
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user"]["department"] == "Updated Department"
    assert data["user"]["first_name"] == "Updated"


def test_update_user_not_found(auth_headers):
    """Test updating non-existent user."""
    response = client.put(
        "/admin/users/9999",
        json={"department": "Test"},
        headers=auth_headers,
    )
    assert response.status_code == 404


# ==============================================================================
# USER ACTIVATION TESTS
# ==============================================================================


def test_toggle_user_activation(auth_headers):
    """Test activating/deactivating a user."""
    # Get an analyst user
    list_response = client.get("/admin/users?role=analyst", headers=auth_headers)
    users = list_response.json()["users"]

    if not users:
        pytest.skip("No analyst users available for testing")

    user_id = users[0]["id"]

    # Deactivate user
    response = client.patch(
        f"/admin/users/{user_id}/activate",
        json={"is_active": False},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "deactivated" in data["message"]
    assert data["user"]["is_active"] is False

    # Reactivate user
    response = client.patch(
        f"/admin/users/{user_id}/activate",
        json={"is_active": True},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "activated" in data["message"]
    assert data["user"]["is_active"] is True


# ==============================================================================
# ROLE UPDATE TESTS
# ==============================================================================


def test_update_user_role(auth_headers):
    """Test updating user role."""
    # Get an analyst user
    list_response = client.get("/admin/users?role=analyst", headers=auth_headers)
    users = list_response.json()["users"]

    if not users:
        pytest.skip("No analyst users available for testing")

    user_id = users[0]["id"]

    # Change role to viewer
    response = client.patch(
        f"/admin/users/{user_id}/role",
        json={"role": "viewer"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["user"]["role"] == "viewer"

    # Change back to analyst
    response = client.patch(
        f"/admin/users/{user_id}/role",
        json={"role": "analyst"},
        headers=auth_headers,
    )
    assert response.status_code == 200


# ==============================================================================
# PASSWORD RESET TESTS
# ==============================================================================


def test_reset_user_password(auth_headers):
    """Test resetting user password."""
    # Get an analyst user
    list_response = client.get("/admin/users?role=analyst", headers=auth_headers)
    users = list_response.json()["users"]

    if not users:
        pytest.skip("No analyst users available for testing")

    user_id = users[0]["id"]

    # Reset password
    response = client.patch(
        f"/admin/users/{user_id}/password",
        json={"new_password": "NewSecurePassword123!"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Password reset successfully" in data["message"]


# ==============================================================================
# DELETE USER TESTS
# ==============================================================================


def test_delete_user_success(auth_headers, test_user_data):
    """Test deleting a user."""
    # First create a user to delete
    create_response = client.post(
        "/admin/users",
        json={
            **test_user_data,
            "username": "user_to_delete",
            "email": "delete@test.com",
        },
        headers=auth_headers,
    )

    if create_response.status_code == 201:
        user_id = create_response.json()["user"]["id"]

        # Delete the user
        response = client.delete(
            f"/admin/users/{user_id}",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted successfully" in data["message"]


def test_delete_user_not_found(auth_headers):
    """Test deleting non-existent user."""
    response = client.delete("/admin/users/9999", headers=auth_headers)
    assert response.status_code == 404


# ==============================================================================
# AUTHORIZATION TESTS
# ==============================================================================


def test_user_endpoints_require_admin(test_user_data):
    """Test that user management endpoints require admin role."""
    # Try to access without any authentication
    endpoints = [
        ("GET", "/admin/users"),
        ("POST", "/admin/users", test_user_data),
        ("GET", "/admin/users/1"),
        ("PUT", "/admin/users/1", {"department": "Test"}),
        ("DELETE", "/admin/users/1"),
        ("PATCH", "/admin/users/1/activate", {"is_active": True}),
        ("PATCH", "/admin/users/1/role", {"role": "analyst"}),
        ("PATCH", "/admin/users/1/password", {"new_password": "NewPass123!"}),
    ]

    for method, url, *data in endpoints:
        if method == "GET":
            response = client.get(url)
        elif method == "POST":
            response = client.post(url, json=data[0] if data else {})
        elif method == "PUT":
            response = client.put(url, json=data[0] if data else {})
        elif method == "PATCH":
            response = client.patch(url, json=data[0] if data else {})
        elif method == "DELETE":
            response = client.delete(url)

        assert response.status_code == 401  # Unauthorized


# ==============================================================================
# SWAGGER DOCUMENTATION TEST
# ==============================================================================


def test_swagger_docs_accessible():
    """Test that Swagger documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_user_management_in_swagger():
    """Test that user management endpoints appear in OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    # Check that user management paths exist
    assert "/admin/users" in schema["paths"]
    assert "/admin/users/{user_id}" in schema["paths"]
    assert "/admin/users/{user_id}/activate" in schema["paths"]
    assert "/admin/users/{user_id}/role" in schema["paths"]
    assert "/admin/users/{user_id}/password" in schema["paths"]
