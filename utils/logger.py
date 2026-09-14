"""Console and file logging plus a config fingerprint.

:func:`setup_logging` configures a single root logger (default name
``cvproject``); child loggers obtained via :func:`get_logger` propagate to it.
:func:`config_hash` produces a short deterministic fingerprint of a
configuration for run headers and bookkeeping.

"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_ROOT_NAME = "cvproject"

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    max_bytes: int = 1_000_000,
    backup_count: int = 3,
    name: str = _ROOT_NAME,
) -> logging.Logger:
    """Idempotently configure a logger with a console and optional file handler.

    Re-calling this function replaces the existing handlers, so it is safe to
    call from tests or when the logging level changes at runtime.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_to_level(level))
    logger.propagate = False

    _remove_handlers(logger)

    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger that inherits the configured ``cvproject`` handlers.

    A default console configuration is created lazily if ``setup_logging`` was
    never called, so logging works out of the box.
    """
    root = logging.getLogger(_ROOT_NAME)
    if not root.handlers:
        setup_logging()

    if name is None or name == _ROOT_NAME:
        return root
    return root.getChild(name)


def config_hash(config: dict[str, Any]) -> str:
    """Return a 16-hex-char SHA-256 fingerprint of a configuration."""
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def log_run_header(logger: logging.Logger, config: dict[str, Any], title: str = "RUN") -> None:
    """Emit a short run header including the config fingerprint."""
    logger.info("%s | config_hash=%s", title, config_hash(config))


def _remove_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _to_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)