"""
Helper utilities for drift detection.
"""

from typing import Any, Dict
from datetime import datetime


def format_timestamp(dt: datetime) -> str:
    """
    Format datetime to ISO string.
    
    Args:
        dt: Datetime object
        
    Returns:
        Formatted string
    """
    return dt.isoformat()


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Old value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / abs(old_value)) * 100


def truncate_dict(d: Dict[str, Any], max_items: int = 10) -> Dict[str, Any]:
    """
    Truncate dictionary to maximum number of items.
    
    Args:
        d: Dictionary to truncate
        max_items: Maximum number of items
        
    Returns:
        Truncated dictionary
    """
    if len(d) <= max_items:
        return d
    
    items = list(d.items())[:max_items]
    return dict(items)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator
