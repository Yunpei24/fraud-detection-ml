"""
Centralized logging configuration for the data module.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name (str): Logger name (typically __name__)
        level (str): Logging level (INFO, DEBUG, WARNING, ERROR)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


# Module-level logger
logger = get_logger(__name__)
