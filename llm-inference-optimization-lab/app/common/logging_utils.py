"""Structured JSON logging configuration for all services."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.common.config import LoggingConfig


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        return json.dumps(log_entry, default=str)


def setup_logging(service_name: str, level: str | None = None) -> logging.Logger:
    """Configure and return a logger with structured JSON output."""
    log_level = getattr(logging, (level or LoggingConfig.level).upper(), logging.INFO)

    logger = logging.getLogger(service_name)
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        if LoggingConfig.structured:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
            )
        logger.addHandler(handler)

    return logger
