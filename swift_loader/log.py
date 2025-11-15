"""Logging module configuration for SwiftLoader.

This module provides structured logging functionality using structlog for
consistent, JSON-formatted log output.

* Copyright: 2023 Dat Tran
* Authors: Dat Tran
* Emails: hello@dats.bio
* Date: 2023-11-12
* Version: 0.0.1

License
-------
Apache 2.0
"""

from __future__ import annotations

from typing import Any

import logging
from logging import NullHandler
from logging.handlers import RotatingFileHandler

import structlog


def get_logger(**config: dict) -> Any:
    """Get a structured logger instance.

    Creates and configures a structlog logger with optional file and/or
    stdout output. Logs are formatted as JSON for easy parsing.

    Args:
        **config: Logger configuration dictionary with optional keys:
            path: Log file path (str). If None, no file logging.
                Supports .json and .log extensions.
            suffix: Suffix to append to log file name. Defaults to None.
            name: Logger name. Defaults to None.
            stdout: Whether to log to stdout. Defaults to False.
            level: Log level (str). Defaults to "DEBUG".
            separate_sink: Whether to use separate sink for suffix.
                Defaults to True.

    Returns:
        Configured structlog logger instance.

    Example:
        >>> logger = get_logger(
        ...     path="/tmp/logs/app.log",
        ...     name="my_logger",
        ...     stdout=True,
        ...     level="INFO"
        ... )
        >>> logger.info("Application started", user="alice")
    """
    if config is None:
        logger = logging.getLogger()
        logger.setLevel("NOTSET")
        logger.addHandler(NullHandler())
        return structlog.wrap_logger(logger)

    path = config.get("path", None)
    suffix = config.get("suffix")
    stdout = config.get("stdout", False)
    name = config.get("name")
    level = config.get("level", "DEBUG").upper()

    # Determine file extension
    if isinstance(path, str):
        if path.endswith(".json"):
            path = path[:-5]
            ext = ".json"
        elif path.endswith(".log"):
            path = path[:-4]
            ext = ".log"
        else:
            ext = ".json"
    else:
        ext = ".json"

    # If separate sink is not set, then suffix is not used
    separate_sink = config.get("separate_sink", True)
    if not separate_sink:
        suffix = None

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Sink to nothingness if no output configured
    if path is None and not stdout:
        logger.addHandler(NullHandler())
        return structlog.wrap_logger(logger)

    # Add file handler if path is specified
    if path:
        if suffix:
            path += "_" + suffix
        path += ext
        file_handler = RotatingFileHandler(path, maxBytes=10485760, backupCount=2)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

    # Add stdout handler if requested
    if stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Return the structured logger
    return structlog.get_logger(name)
