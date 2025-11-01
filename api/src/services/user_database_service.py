"""
User database service for authentication.
"""

from typing import Optional, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

from ..config import get_logger, settings

logger = get_logger(__name__)


class UserDatabaseService:
    """Service for user-related database operations."""

    def __init__(self):
        # Get database credentials from environment or settings
        import os

        self.connection_params = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "fraud_detection"),
            "user": os.getenv("POSTGRES_USER", "fraud_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "fraud_pass_dev_2024"),
        }

    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.connection_params)

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user by username from database.

        Args:
            username: Username to search for

        Returns:
            User dict or None if not found
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    query = """
                        SELECT id, username, email, password_hash, role, 
                               first_name, last_name, department, is_active, 
                               is_verified, created_at, last_login
                        FROM users
                        WHERE username = %s
                    """
                    cursor.execute(query, (username,))
                    result = cursor.fetchone()

                    if result:
                        return dict(result)
                    return None

        except Exception as e:
            logger.error(f"Error fetching user {username}: {e}")
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user by email from database.

        Args:
            email: Email to search for

        Returns:
            User dict or None if not found
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    query = """
                        SELECT id, username, email, password_hash, role, 
                               first_name, last_name, department, is_active, 
                               is_verified, created_at, last_login
                        FROM users
                        WHERE email = %s
                    """
                    cursor.execute(query, (email,))
                    result = cursor.fetchone()

                    if result:
                        return dict(result)
                    return None

        except Exception as e:
            logger.error(f"Error fetching user by email {email}: {e}")
            return None

    def update_last_login(self, username: str) -> bool:
        """
        Update user's last_login timestamp.

        Args:
            username: Username to update

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    query = """
                        UPDATE users
                        SET last_login = %s
                        WHERE username = %s
                    """
                    cursor.execute(query, (datetime.utcnow(), username))
                    conn.commit()
                    return True

        except Exception as e:
            logger.error(f"Error updating last_login for {username}: {e}")
            return False

    def create_user(
        self,
        username: str,
        email: str,
        password_hash: str,
        role: str = "analyst",
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        department: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new user in the database.

        Args:
            username: Unique username
            email: Unique email
            password_hash: Bcrypt hashed password
            role: User role (admin, analyst, viewer)
            first_name: User's first name
            last_name: User's last name
            department: User's department

        Returns:
            Created user dict or None if failed
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    query = """
                        INSERT INTO users 
                        (username, email, password_hash, role, first_name, 
                         last_name, department, is_active, is_verified)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, FALSE)
                        RETURNING id, username, email, role, first_name, 
                                  last_name, department, is_active, created_at
                    """
                    cursor.execute(
                        query,
                        (
                            username,
                            email,
                            password_hash,
                            role,
                            first_name,
                            last_name,
                            department,
                        ),
                    )
                    result = cursor.fetchone()
                    conn.commit()

                    if result:
                        return dict(result)
                    return None

        except psycopg2.IntegrityError as e:
            logger.warning(f"User creation failed (duplicate): {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None

    def update_user_password(self, username: str, new_password_hash: str) -> bool:
        """
        Update user's password.

        Args:
            username: Username
            new_password_hash: New bcrypt hashed password

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    query = """
                        UPDATE users
                        SET password_hash = %s, updated_at = %s
                        WHERE username = %s
                    """
                    cursor.execute(
                        query, (new_password_hash, datetime.utcnow(), username)
                    )
                    conn.commit()
                    return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error updating password for {username}: {e}")
            return False

    def deactivate_user(self, username: str) -> bool:
        """
        Deactivate a user (soft delete).

        Args:
            username: Username to deactivate

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    query = """
                        UPDATE users
                        SET is_active = FALSE, updated_at = %s
                        WHERE username = %s
                    """
                    cursor.execute(query, (datetime.utcnow(), username))
                    conn.commit()
                    return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error deactivating user {username}: {e}")
            return False


# Global user database service instance
user_db_service = UserDatabaseService()
