"""
Cloud module initialization for Azure services.
"""

from .database import execute_query, get_database_connection

__all__ = [
    "get_database_connection",
    "execute_query",
]
