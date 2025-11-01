# training/src/utils/logger.py
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _make_stream_handler(level: int, json_logs: bool) -> logging.Handler:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    return handler


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Create/retrieve a module logger.
    Environment variables:
      LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
      JSON_LOGS=1 (to emit JSON lines)
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    json_logs = os.environ.get("JSON_LOGS", "0") == "1"

    logger = logging.getLogger(name if name else "training")
    if logger.handlers:
        return logger  # already configured (avoid duplicate handlers)

    logger.setLevel(level)
    logger.propagate = False
    logger.addHandler(
        _make_stream_handler(getattr(logging, level, logging.INFO), json_logs)
    )
    return logger
