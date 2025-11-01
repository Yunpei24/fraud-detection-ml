# User Management API Documentation

## Overview

This document provides comprehensive documentation for the User Management API endpoints. All endpoints are **admin-only** and require JWT authentication with admin role.

## Table of Contents

- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Create User](#create-user)
  - [List Users](#list-users)
  - [Get User](#get-user)
  - [Update User](#update-user)
  - [Delete User](#delete-user)
  - [Activate/Deactivate User](#activatedeactivate-user)
  - [Update User Role](#update-user-role)
  - [Reset Password](#reset-password)
- [User Roles](#user-roles)
- [Security Considerations](#security-considerations)
- [Examples](#examples)

---

## Authentication

All user management endpoints require:
1. **JWT Authentication**: Valid JWT token in Authorization header
2. **Admin Role**: User must have `admin` role

### Getting Admin Token

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "username": "admin",
    "role": "admin",
    "is_active": true
  }
}
```

### Using the Token

Include the token in subsequent requests:
```bash
Authorization: Bearer <access_token>
```

---

## Endpoints

### Create User

**POST** `/admin/users`

Create a new user account.

#### Request Body

```json
{
  "username": "john_analyst",
  "email": "john.doe@company.com",
  "password": "SecurePass123!",
  "first_name": "John",
  "last_name": "Doe",
  "role": "analyst",
  "department": "Fraud Detection",
  "is_active": true,
  "is_verified": false
}
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| username | string | Yes | Unique username (3-50 chars, alphanumeric + _ -) |
| email | string | Yes | Valid email address |
| password | string | Yes | Strong password (min 8 chars) |
| first_name | string | No | First name |
| last_name | string | No | Last name |
| role | string | No | User role (default: analyst) |
| department | string | No | Department name |
| is_active | boolean | No | Active status (default: true) |
| is_verified | boolean | No | Email verified (default: false) |

#### Response (201 Created)

```json
{
  "success": true,
  "message": "User 'john_analyst' created successfully",
  "user": {
    "id": 123,
    "username": "john_analyst",
    "email": "john.doe@company.com",
    "first_name": "John",
    "last_name": "Doe",
    "role": "analyst",
    "department": "Fraud Detection",
    "is_active": true,
    "is_verified": false,
    "created_at": "2025-01-20T10:30:00Z",
    "last_login": null
  }
}
```

#### Errors

- **400**: Username or email already exists
- **401**: Not authenticated
- **403**: Not admin user
- **422**: Validation error
- **500**: Database error

#### cURL Example

```bash
curl -X POST http://localhost:8000/admin/users \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_analyst",
    "email": "john@company.com",
    "password": "SecurePass123!",
    "role": "analyst"
  }'
```

---

### List Users

**GET** `/admin/users`

Get a paginated list of all users with optional filters.

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 50 | Users per page (1-100) |
| offset | integer | 0 | Number of users to skip |
| role | string | null | Filter by role |
| is_active | boolean | null | Filter by active status |

#### Response (200 OK)

```json
{
  "users": [
    {
      "id": 1,
      "username": "admin",
      "email": "admin@company.com",
      "first_name": "Admin",
      "last_name": "User",
      "role": "admin",
      "department": "IT",
      "is_active": true,
      "is_verified": true,
      "created_at": "2025-01-01T00:00:00Z",
      "last_login": "2025-01-20T09:15:00Z"
    },
    {
      "id": 2,
      "username": "john_analyst",
      "email": "john@company.com",
      "first_name": "John",
      "last_name": "Doe",
      "role": "analyst",
      "department": "Fraud Detection",
      "is_active": true,
      "is_verified": true,
      "created_at": "2025-01-10T00:00:00Z",
      "last_login": "2025-01-19T16:30:00Z"
    }
  ],
  "total_count": 2,
  "limit": 50,
  "offset": 0
}
```

#### cURL Examples

**List all users:**
```bash
curl -X GET http://localhost:8000/admin/users \
  -H "Authorization: Bearer <token>"
```

**Filter by role:**
```bash
curl -X GET "http://localhost:8000/admin/users?role=analyst&limit=20" \
  -H "Authorization: Bearer <token>"
```

**Pagination:**
```bash
curl -X GET "http://localhost:8000/admin/users?limit=10&offset=20" \
  -H "Authorization: Bearer <token>"
```

---

### Get User

**GET** `/admin/users/{user_id}`

Get detailed information about a specific user.

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| user_id | integer | ID of the user |

#### Response (200 OK)

```json
{
  "id": 123,
  "username": "john_analyst",
  "email": "john@company.com",
  "first_name": "John",
  "last_name": "Doe",
  "role": "analyst",
  "department": "Fraud Detection",
  "is_active": true,
  "is_verified": true,
  "created_at": "2025-01-15T10:30:00Z",
  "last_login": "2025-01-20T14:25:00Z"
}
```

#### Errors

- **401**: Not authenticated
- **403**: Not admin user
- **404**: User not found
- **500**: Database error

#### cURL Example

```bash
curl -X GET http://localhost:8000/admin/users/123 \
  -H "Authorization: Bearer <token>"
```

---

### Update User

**PUT** `/admin/users/{user_id}`

Update user information. All fields are optional.

#### Request Body

```json
{
  "email": "john.doe@newcompany.com",
  "first_name": "John",
  "last_name": "Doe",
  "role": "analyst",
  "department": "Risk Management",
  "is_active": true,
  "is_verified": true
}
```

#### Response (200 OK)

```json
{
  "success": true,
  "message": "User 'john_analyst' updated successfully",
  "user": {
    "id": 123,
    "username": "john_analyst",
    "email": "john.doe@newcompany.com",
    "first_name": "John",
    "last_name": "Doe",
    "role": "analyst",
    "department": "Risk Management",
    "is_active": true,
    "is_verified": true,
    "created_at": "2025-01-15T10:30:00Z",
    "last_login": "2025-01-20T14:25:00Z"
  }
}
```

#### cURL Example

```bash
curl -X PUT http://localhost:8000/admin/users/123 \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "department": "Risk Management",
    "is_verified": true
  }'
```

---

### Delete User

**DELETE** `/admin/users/{user_id}`

Permanently delete a user account.

**⚠️ Warning**: This action cannot be undone!

#### Restrictions

- Cannot delete your own account
- Cannot delete the last admin user

#### Response (200 OK)

```json
{
  "success": true,
  "message": "User 'john_analyst' deleted successfully",
  "user": null
}
```

#### Errors

- **400**: Cannot delete own account or last admin
- **401**: Not authenticated
- **403**: Not admin user
- **404**: User not found
- **500**: Database error

#### cURL Example

```bash
curl -X DELETE http://localhost:8000/admin/users/123 \
  -H "Authorization: Bearer <token>"
```

---

### Activate/Deactivate User

**PATCH** `/admin/users/{user_id}/activate`

Activate or deactivate a user account.

#### Request Body

```json
{
  "is_active": false
}
```

#### Response (200 OK)

```json
{
  "success": true,
  "message": "User 'john_analyst' deactivated successfully",
  "user": {
    "id": 123,
    "username": "john_analyst",
    "email": "john@company.com",
    "first_name": "John",
    "last_name": "Doe",
    "role": "analyst",
    "department": "Fraud Detection",
    "is_active": false,
    "is_verified": true,
    "created_at": "2025-01-15T10:30:00Z",
    "last_login": "2025-01-20T14:25:00Z"
  }
}
```

#### cURL Example

**Deactivate user:**
```bash
curl -X PATCH http://localhost:8000/admin/users/123/activate \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}'
```

**Reactivate user:**
```bash
curl -X PATCH http://localhost:8000/admin/users/123/activate \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"is_active": true}'
```

---

### Update User Role

**PATCH** `/admin/users/{user_id}/role`

Change a user's role.

#### Request Body

```json
{
  "role": "admin"
}
```

#### Valid Roles

- `admin`: Full system access
- `analyst`: Can analyze and label transactions
- `viewer`: Read-only access

#### Restrictions

- Cannot change your own role
- Cannot remove admin role from last admin

#### Response (200 OK)

```json
{
  "success": true,
  "message": "User 'john_analyst' role changed to 'admin' successfully",
  "user": {
    "id": 123,
    "username": "john_analyst",
    "email": "john@company.com",
    "role": "admin",
    "is_active": true,
    "created_at": "2025-01-15T10:30:00Z"
  }
}
```

#### cURL Example

```bash
curl -X PATCH http://localhost:8000/admin/users/123/role \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"role": "admin"}'
```

---

### Reset Password

**PATCH** `/admin/users/{user_id}/password`

Reset a user's password.

**Security Note**: The new password is hashed using bcrypt before storage.

#### Request Body

```json
{
  "new_password": "NewSecurePass456!"
}
```

#### Response (200 OK)

```json
{
  "success": true,
  "message": "Password reset successfully for user 'john_analyst'",
  "user": null
}
```

#### cURL Example

```bash
curl -X PATCH http://localhost:8000/admin/users/123/password \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"new_password": "NewSecurePass456!"}'
```

---

## User Roles

### Admin
- **Full system access**
- Can manage all users
- Can create, update, delete users
- Can change user roles
- Can reset passwords
- Can view all audit logs

### Analyst
- Can make fraud predictions
- Can label transactions
- Can view predictions and explanations
- Cannot manage users
- Cannot access admin endpoints

### Viewer
- Read-only access
- Can view predictions
- Cannot make predictions
- Cannot label transactions
- Cannot manage users

---

## Security Considerations

### Password Requirements

- Minimum 8 characters
- Hashed using bcrypt before storage
- Never returned in API responses

### Username Requirements

- 3-50 characters
- Alphanumeric characters plus underscore (_) and hyphen (-)
- Must be unique
- Case-insensitive

### Email Requirements

- Valid email format
- Must be unique
- Case-insensitive

### Admin Protections

1. **Cannot delete own account**: Prevents admin lockout
2. **Cannot delete last admin**: Ensures system always has an admin
3. **Cannot change own role**: Prevents privilege escalation
4. **Cannot deactivate own account**: Prevents account lockout

### Audit Logging

All user management actions are logged:
- User creation
- User updates
- User deletion
- Role changes
- Password resets
- Activation/deactivation

---

## Examples

### Complete User Management Workflow

#### 1. Login as Admin

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123" | jq -r '.access_token')
```

#### 2. Create New Analyst

```bash
curl -X POST http://localhost:8000/admin/users \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jane_analyst",
    "email": "jane@company.com",
    "password": "SecurePass123!",
    "first_name": "Jane",
    "last_name": "Smith",
    "role": "analyst",
    "department": "Fraud Detection"
  }'
```

#### 3. List All Users

```bash
curl -X GET http://localhost:8000/admin/users \
  -H "Authorization: Bearer $TOKEN" | jq
```

#### 4. Update User Department

```bash
curl -X PUT http://localhost:8000/admin/users/2 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"department": "Risk Management"}'
```

#### 5. Promote to Admin

```bash
curl -X PATCH http://localhost:8000/admin/users/2/role \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"role": "admin"}'
```

#### 6. Reset Password

```bash
curl -X PATCH http://localhost:8000/admin/users/2/password \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"new_password": "NewSecurePass456!"}'
```

#### 7. Deactivate User

```bash
curl -X PATCH http://localhost:8000/admin/users/2/activate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}'
```

---

## Testing

### Run Unit Tests

```bash
cd api
python -m pytest tests/unit/test_users_routes.py -v
```

### Run Integration Tests

```bash
python -m pytest tests/integration/test_users_api.py -v
```

### Test Coverage

```bash
python -m pytest tests/unit/test_users_routes.py --cov=src/api/routes/users --cov-report=html
```

---

## Swagger Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

All user management endpoints are documented under the **"user-management"** tag.

---

## Support

For issues or questions:
- Check the Swagger documentation
- Review the test files for examples
- Contact the development team
