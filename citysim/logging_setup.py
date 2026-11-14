"""Logging configuration shared by the CLI and dashboard processes."""

from __future__ import annotations

import logging
import os
from pathlib import Path

LOGGER_NAME = "citysim"
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MANAGED_ATTRIBUTE = "_citysim_managed"


def resolve_log_level(
    level: str | int | None = None, *, quiet: bool = False
) -> int:
    """Resolve an explicit level, ``CITYSIM_LOG_LEVEL``, or the default."""
    value: str | int | None = level
    if value is None:
        value = os.environ.get("CITYSIM_LOG_LEVEL")
    if value is None:
        return logging.WARNING if quiet else logging.INFO
    if isinstance(value, int):
        return value
    normalized = str(value).strip().upper()
    levels = logging.getLevelNamesMapping()
    if normalized not in levels:
        raise ValueError(f"unknown log level: {value!r}")
    return levels[normalized]


def configure_logging(
    level: str | int | None = None,
    log_file: str | Path | None = None,
    *,
    quiet: bool = False,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``citysim`` logger without touching host handlers.

    Repeated calls are a no-op unless ``force`` is set, which lets embedders
    and tests replace the handlers created here.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolve_log_level(level, quiet=quiet))
    logger.propagate = False

    managed = [
        handler
        for handler in logger.handlers
        if getattr(handler, _MANAGED_ATTRIBUTE, False)
    ]
    if managed and not force:
        return logger
    for handler in managed:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    stream_handler = logging.StreamHandler()
    setattr(stream_handler, _MANAGED_ATTRIBUTE, True)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    target = log_file if log_file is not None else os.environ.get("CITYSIM_LOG_FILE")
    if target:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        setattr(file_handler, _MANAGED_ATTRIBUTE, True)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
