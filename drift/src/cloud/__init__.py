"""
Cloud module initialization for Azure services.
"""

from .database import get_database_connection, execute_query

__all__ = [
    "get_database_connection",
    "execute_query",
]
