"""
Azure Database connection utilities for drift detection.
"""

from typing import Any, Optional

import pandas as pd
import structlog
from sqlalchemy import create_engine

from ..config.settings import Settings

logger = structlog.get_logger(__name__)


def get_database_connection(settings: Optional[Settings] = None):
    """
    Get database connection engine.

    Args:
        settings: Configuration settings

    Returns:
        SQLAlchemy engine
    """
    settings = settings or Settings()

    try:
        engine = create_engine(settings.database_url)
        logger.info("database_connection_established")
        return engine

    except Exception as e:
        logger.error("failed_to_connect_to_database", error=str(e))
        raise


def execute_query(query: str, settings: Optional[Settings] = None) -> pd.DataFrame:
    """
    Execute SQL query and return results as DataFrame.

    Args:
        query: SQL query string
        settings: Configuration settings

    Returns:
        DataFrame with query results
    """
    try:
        engine = get_database_connection(settings)
        df = pd.read_sql(query, engine)
        logger.info("query_executed", rows=len(df))
        return df

    except Exception as e:
        logger.error("query_execution_failed", error=str(e))
        return pd.DataFrame()
