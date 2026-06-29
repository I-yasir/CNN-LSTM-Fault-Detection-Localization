"""
Project logging utilities.

Features
--------
- Console logging
- File logging
- Timestamped log files
- Configurable log level
- Singleton logger (prevents duplicate handlers)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


_LOGGER_NAME = "FaultDetection"


def get_logger(
    log_directory: str | Path,
    level: str = "INFO",
) -> logging.Logger:
    """
    Create (or return) the project logger.

    Parameters
    ----------
    log_directory : str | Path
        Directory where log files will be stored.

    level : str
        Logging level.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(_LOGGER_NAME)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    log_directory = Path(log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = log_directory / f"run_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # -----------------------------
    # Console Handler
    # -----------------------------
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    console_handler.setLevel(getattr(logging, level.upper()))

    # -----------------------------
    # File Handler
    # -----------------------------
    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    file_handler.setLevel(getattr(logging, level.upper()))

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    logger.propagate = False

    logger.info("=" * 70)
    logger.info("Logger initialized")
    logger.info("Log file: %s", log_file)
    logger.info("=" * 70)

    return logger