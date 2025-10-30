"""
Authentication service for JWT token management.
"""
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from ..config import get_logger, settings

logger = get_logger(__name__)


class AuthService:
    """Service for handling JWT authentication."""

    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = settings.auth.secret_key
        self.algorithm = settings.auth.algorithm
        self.access_token_expire_minutes = settings.auth.access_token_expire_minutes

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)

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

        In production, this would query the database.
        For now, using simple hardcoded users for demo.
        """
        # Demo users - in production, query from database
        demo_users = {
            "admin": {
                "username": "admin",
                "hashed_password": self.get_password_hash("admin123"),
                "role": "admin",
                "is_active": True,
            },
            "analyst": {
                "username": "analyst",
                "hashed_password": self.get_password_hash("analyst123"),
                "role": "analyst",
                "is_active": True,
            },
        }

        user = demo_users.get(username)
        if not user or not user["is_active"]:
            return None

        if not self.verify_password(password, user["hashed_password"]):
            return None

        return {
            "username": user["username"],
            "role": user["role"],
            "is_active": user["is_active"],
        }


# Global auth service instance
auth_service = AuthService()
