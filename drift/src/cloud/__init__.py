"""
Cloud module initialization for Azure services.
"""

from .database import get_database_connection, execute_query
from .blob_storage import (
    upload_report_to_blob,
    download_baseline_data,
    list_reports
)

__all__ = [
    "get_database_connection",
    "execute_query",
    "upload_report_to_blob",
    "download_baseline_data",
    "list_reports",
]
