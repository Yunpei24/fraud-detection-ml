"""
Authentication API routes for JWT token management.
"""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from ...config import get_logger
from ...models import TokenResponse, UserResponse
from ...services.auth_service import auth_service

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Dependency to get current authenticated user from JWT token.

    Args:
        token: JWT access token

    Returns:
        User information

    Raises:
        HTTPException: If token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = auth_service.verify_token(token)
    if payload is None:
        raise credentials_exception

    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception

    # In production, fetch user from database
    # For now, return payload data
    user = {
        "username": username,
        "role": payload.get("role", "user"),
        "is_active": True,
    }

    return user


async def get_current_admin_user(current_user: dict = Depends(get_current_user)):
    """
    Dependency to ensure current user has admin role.

    Args:
        current_user: Current authenticated user

    Returns:
        User information if admin

    Raises:
        HTTPException: If user is not admin
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )
    return current_user


async def get_current_analyst_user(current_user: dict = Depends(get_current_user)):
    """
    Dependency to ensure current user has analyst or admin role.

    Args:
        current_user: Current authenticated user

    Returns:
        User information if analyst/admin

    Raises:
        HTTPException: If user doesn't have required role
    """
    user_role = current_user.get("role")
    if user_role not in ["admin", "analyst"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analyst or admin access required",
        )
    return current_user


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="User login",
    description="Authenticate user and return JWT access token",
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return JWT token.

    Args:
        form_data: OAuth2 form with username and password

    Returns:
        Access token and token type

    Raises:
        HTTPException: If authentication fails
    """
    user = auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=auth_service.access_token_expire_minutes)
    access_token = auth_service.create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires,
    )

    logger.info(f"Successful login for user: {user['username']}")

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=auth_service.access_token_expire_minutes * 60,
        user=user,  # Pass dict directly, not UserResponse
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Refresh token",
    description="Refresh JWT access token",
)
async def refresh_token(current_user: dict = Depends(get_current_user)):
    """
    Refresh JWT token for authenticated user.

    Args:
        current_user: Current authenticated user

    Returns:
        New access token
    """
    access_token_expires = timedelta(minutes=auth_service.access_token_expire_minutes)
    access_token = auth_service.create_access_token(
        data={"sub": current_user["username"], "role": current_user["role"]},
        expires_delta=access_token_expires,
    )

    logger.info(f"Token refreshed for user: {current_user['username']}")

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=auth_service.access_token_expire_minutes * 60,
        user=UserResponse(**current_user),
    )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current user",
    description="Get information about the currently authenticated user",
)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current user information.

    Args:
        current_user: Current authenticated user

    Returns:
        User information
    """
    return UserResponse(**current_user)
