"""
Authentication service for JWT token management.
"""

import hashlib
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from ..config import get_logger, settings
from .user_database_service import user_db_service

logger = get_logger(__name__)


class AuthService:
    """Service for handling JWT authentication."""

    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = settings.auth.secret_key
        self.algorithm = settings.auth.algorithm
        self.access_token_expire_minutes = settings.auth.access_token_expire_minutes

    def _normalize_password(self, password: str) -> bytes:
        """
        Normalize password to ensure it fits bcrypt's 72-byte limit.
        Uses SHA256 to hash long passwords before bcrypt.

        Args:
            password: Plain text password

        Returns:
            Normalized password bytes (always <= 72 bytes)
        """
        # Convert to bytes
        password_bytes = password.encode("utf-8")

        # If password is longer than 72 bytes, pre-hash with SHA256
        if len(password_bytes) > 72:
            logger.debug(
                f"Password exceeds 72 bytes ({len(password_bytes)} bytes), applying SHA256 pre-hash"
            )
            return hashlib.sha256(password_bytes).hexdigest().encode("utf-8")

        return password_bytes

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        normalized_password = self._normalize_password(plain_password)
        return self.pwd_context.verify(normalized_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        normalized_password = self._normalize_password(password)
        return self.pwd_context.hash(normalized_password)

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None

    def authenticate_user(
        self, username: str, password: str
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user with username and password.

        NOW USES POSTGRESQL DATABASE! ✅
        """
        # Fetch user from database
        user = user_db_service.get_user_by_username(username)

        if not user:
            logger.warning(f"User not found: {username}")
            return None

        # Check if user is active
        if not user.get("is_active"):
            logger.warning(f"Inactive user attempted login: {username}")
            return None

        # Verify password
        if not self.verify_password(password, user["password_hash"]):
            logger.warning(f"Invalid password for user: {username}")
            return None

        # Update last_login timestamp
        user_db_service.update_last_login(username)

        logger.info(f"User authenticated successfully: {username}")

        # Return user info (without password_hash)
        return {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "first_name": user.get("first_name"),
            "last_name": user.get("last_name"),
            "department": user.get("department"),
            "is_active": user["is_active"],
            "is_verified": user.get("is_verified", False),
        }

    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        role: str = "analyst",
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        department: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Register a new user.

        Args:
            username: Unique username
            email: Unique email
            password: Plain text password (will be hashed)
            role: User role
            first_name: User's first name
            last_name: User's last name
            department: User's department

        Returns:
            Created user info or None if failed
        """
        # Hash the password
        password_hash = self.get_password_hash(password)

        # Create user in database
        user = user_db_service.create_user(
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            first_name=first_name,
            last_name=last_name,
            department=department,
        )

        if user:
            logger.info(f"New user registered: {username}")
        else:
            logger.error(f"Failed to register user: {username}")

        return user


# Global auth service instance
auth_service = AuthService()
