"""
Authentication service for JWT token management.
"""

import bcrypt
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from jose import JWTError, jwt

from ..config import get_logger, settings
from .user_database_service import user_db_service

logger = get_logger(__name__)


class AuthService:
    """Service for handling JWT authentication."""

    def __init__(self):
        # Use bcrypt directly instead of passlib to avoid 72-byte limit issues
        self.secret_key = settings.auth.secret_key
        self.algorithm = settings.auth.algorithm
        self.access_token_expire_minutes = settings.auth.access_token_expire_minutes

    def _normalize_password(self, password: str) -> bytes:
        """
        Normalize password using SHA256 to ensure it fits bcrypt's 72-byte limit.
        Always uses SHA256 for consistency, regardless of password length.

        Args:
            password: Plain text password

        Returns:
            SHA256 hash as bytes (always 64 bytes, well under bcrypt's 72-byte limit)
        """
        # Always use SHA256 pre-hash for consistency
        password_bytes = password.encode("utf-8")
        sha256_hash = hashlib.sha256(password_bytes).hexdigest()

        # Convert hex string to bytes (64 bytes, safe for bcrypt)
        return sha256_hash.encode("utf-8")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash using bcrypt directly."""
        try:
            normalized_password = self._normalize_password(plain_password)
            # bcrypt.checkpw expects bytes for both password and hash
            return bcrypt.checkpw(normalized_password, hashed_password.encode("utf-8"))
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False

    def get_password_hash(self, password: str) -> str:
        """Hash a password using bcrypt directly."""
        normalized_password = self._normalize_password(password)
        # Generate salt and hash
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(normalized_password, salt)
        return hashed.decode("utf-8")

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
