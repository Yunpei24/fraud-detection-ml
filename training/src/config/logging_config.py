"""
Logging configuration for the Fraud Detection ML training module.

Provides a reusable logger that writes messages to both the console
and a persistent log file (training/logs/training.log).
"""

import logging
import sys
from pathlib import Path


# ------------------------------------------------------------
# 1. Log Directory and File Setup
# ------------------------------------------------------------
LOG_DIR = Path("training/logs")
LOG_FILE = LOG_DIR / "training.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Base Logger Configuration
# ------------------------------------------------------------
def get_logger(name: str = "fraud_detection_training") -> logging.Logger:
    """
    Creates and returns a configured logger.

    Args:
        name (str): Logger name (module or class specific).

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if logger already exists
    if logger.hasHandlers():
        return logger

    # Console handler (stream to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # File handler (persistent logging)
    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setLevel(logging.INFO)

    # Define log format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized: {name}")
    return logger


# ------------------------------------------------------------
# 3. Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    log = get_logger("example_training_logger")
    log.info("This is an info log from the training configuration.")
    log.warning("This is a warning log.")
    log.error("This is an error log.")
