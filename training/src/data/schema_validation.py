# training/src/data/schema_validation.py
import numpy as np
import pandas as pd
from typing import Iterable
from training.src.config.logging_config import get_logger  # Import the logger configuration

# Create the logger instance
logger = get_logger(__name__)

def validate_schema(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise if required columns are missing. Warn if extras are present."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Identify and log any extra columns
    extras = [c for c in df.columns if c not in required]
    if extras:
        logger.warning(f"Extra unexpected columns found: {extras}")
