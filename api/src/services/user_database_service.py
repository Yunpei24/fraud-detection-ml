"""
User database service for authentication.
"""

from typing import Optional, Dict, Any, List, Tuple
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
        is_active: bool = True,
        is_verified: bool = False,
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
            is_active: Whether user is active (default: True)
            is_verified: Whether email is verified (default: False)

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
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id, username, email, role, first_name, 
                                  last_name, department, is_active, is_verified, created_at
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
                            is_active,
                            is_verified,
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

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve user by ID from database.

        Args:
            user_id: User ID to search for

        Returns:
            User dict or None if not found
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    query = """
                        SELECT id, username, email, password_hash, role, 
                               first_name, last_name, department, is_active, 
                               is_verified, created_at, last_login, updated_at
                        FROM users
                        WHERE id = %s
                    """
                    cursor.execute(query, (user_id,))
                    result = cursor.fetchone()

                    if result:
                        user = dict(result)
                        # Convert datetime objects to ISO format strings
                        if user.get("created_at"):
                            user["created_at"] = user["created_at"].isoformat()
                        if user.get("last_login"):
                            user["last_login"] = user["last_login"].isoformat()
                        if user.get("updated_at"):
                            user["updated_at"] = user["updated_at"].isoformat()
                        return user
                    return None

        except Exception as e:
            logger.error(f"Error fetching user by ID {user_id}: {e}")
            return None

    def list_users(
        self,
        limit: int = 50,
        offset: int = 0,
        role: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        List users with optional filters and pagination.

        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip
            role: Filter by role (optional)
            is_active: Filter by active status (optional)

        Returns:
            Tuple of (list of user dicts, total count)
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Build WHERE clause
                    where_clauses = []
                    params = []

                    if role is not None:
                        where_clauses.append("role = %s")
                        params.append(role)

                    if is_active is not None:
                        where_clauses.append("is_active = %s")
                        params.append(is_active)

                    where_sql = ""
                    if where_clauses:
                        where_sql = "WHERE " + " AND ".join(where_clauses)

                    # Get total count
                    count_query = f"SELECT COUNT(*) FROM users {where_sql}"
                    cursor.execute(count_query, params)
                    total_count = cursor.fetchone()["count"]

                    # Get users
                    query = f"""
                        SELECT id, username, email, role, 
                               first_name, last_name, department, is_active, 
                               is_verified, created_at, last_login, updated_at
                        FROM users
                        {where_sql}
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """
                    cursor.execute(query, params + [limit, offset])
                    results = cursor.fetchall()

                    # Convert datetime objects to ISO format strings
                    users = []
                    for row in results:
                        user = dict(row)
                        if user.get("created_at"):
                            user["created_at"] = user["created_at"].isoformat()
                        if user.get("last_login"):
                            user["last_login"] = user["last_login"].isoformat()
                        if user.get("updated_at"):
                            user["updated_at"] = user["updated_at"].isoformat()
                        users.append(user)

                    return users, total_count

        except Exception as e:
            logger.error(f"Error listing users: {e}")
            return [], 0

    def update_user(
        self, user_id: int, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update user information.

        Args:
            user_id: ID of the user to update
            updates: Dictionary of fields to update

        Returns:
            Updated user dict or None if failed
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Build UPDATE query dynamically
                    allowed_fields = [
                        "email",
                        "first_name",
                        "last_name",
                        "department",
                        "role",
                        "is_active",
                        "is_verified",
                        "password_hash",
                    ]

                    update_fields = []
                    params = []

                    for field, value in updates.items():
                        if field in allowed_fields:
                            update_fields.append(f"{field} = %s")
                            params.append(value)

                    if not update_fields:
                        logger.warning(f"No valid fields to update for user {user_id}")
                        return self.get_user_by_id(user_id)

                    # Add updated_at timestamp
                    update_fields.append("updated_at = %s")
                    params.append(datetime.utcnow())

                    # Add user_id for WHERE clause
                    params.append(user_id)

                    query = f"""
                        UPDATE users
                        SET {", ".join(update_fields)}
                        WHERE id = %s
                        RETURNING id, username, email, role, 
                                  first_name, last_name, department, 
                                  is_active, is_verified, created_at, 
                                  last_login, updated_at
                    """
                    cursor.execute(query, params)
                    result = cursor.fetchone()
                    conn.commit()

                    if result:
                        user = dict(result)
                        # Convert datetime objects to ISO format strings
                        if user.get("created_at"):
                            user["created_at"] = user["created_at"].isoformat()
                        if user.get("last_login"):
                            user["last_login"] = user["last_login"].isoformat()
                        if user.get("updated_at"):
                            user["updated_at"] = user["updated_at"].isoformat()
                        return user
                    return None

        except psycopg2.IntegrityError as e:
            logger.warning(f"User update failed (constraint violation): {e}")
            return None
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}")
            return None


# Global user database service instance
user_db_service = UserDatabaseService()
