"""
Helper functions for the data module.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize DataFrame column names.
    
    Converts to lowercase, removes spaces, and standardizes format.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with sanitized column names
    """
    df = df.copy()
    df.columns = [col.lower().replace(' ', '_').strip() for col in df.columns]
    return df


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get human-readable memory usage of DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        str: Memory usage as human-readable string (e.g., "1.5 MB")
    """
    bytes_used = df.memory_usage(deep=True).sum()
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_used < 1024.0:
            return f"{bytes_used:.1f} {unit}"
        bytes_used /= 1024.0
    
    return f"{bytes_used:.1f} TB"


def get_time_bucket(timestamp: datetime, bucket_minutes: int = 60) -> datetime:
    """
    Round timestamp down to nearest time bucket.
    
    Args:
        timestamp (datetime): Input timestamp
        bucket_minutes (int): Bucket size in minutes
        
    Returns:
        datetime: Rounded timestamp
    """
    return timestamp.replace(
        minute=(timestamp.minute // bucket_minutes) * bucket_minutes,
        second=0,
        microsecond=0
    )


def format_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format response with standard structure.
    
    Args:
        data (Dict): Response data
        
    Returns:
        Dict: Formatted response
    """
    return {
        'status': 'success',
        'timestamp': datetime.utcnow().isoformat(),
        'data': data
    }


def format_error(error_message: str, error_code: str = 'ERROR') -> Dict[str, Any]:
    """
    Format error response with standard structure.
    
    Args:
        error_message (str): Error message
        error_code (str): Error code
        
    Returns:
        Dict: Formatted error response
    """
    return {
        'status': 'error',
        'timestamp': datetime.utcnow().isoformat(),
        'error_code': error_code,
        'message': error_message
    }
