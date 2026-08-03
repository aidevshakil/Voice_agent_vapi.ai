"""Logging setup with request correlation.

A ``ContextVar`` carries the request id so every log line emitted while handling
a request is traceable without threading a logger through call signatures.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def set_request_id(value: str | None = None) -> str:
    rid = value or new_request_id()
    _request_id.set(rid)
    return rid


def get_request_id() -> str | None:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Idempotently configure the root logger."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())

    if json_output:
        try:
            from pythonjsonlogger import jsonlogger

            formatter: logging.Formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
                rename_fields={"asctime": "ts", "levelname": "level"},
            )
        except ImportError:  # optional dependency
            json_output = False
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | rid=%(request_id)s | %(message)s"
            )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)-34s | rid=%(request_id)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These are chatty and rarely useful at INFO.
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "hpack", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured-ish single line: ``event key=value key=value``."""
    if not fields:
        logger.info(event)
        return
    rendered = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    logger.info("%s %s", event, rendered)
