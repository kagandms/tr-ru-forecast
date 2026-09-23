"""Central structured logging foundation for the TR-RU forecasting platform."""

import logging
import sys


class CustomFormatter(logging.Formatter):
    """Clean, consistent log formatter with timestamp, level, module, and message."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
) -> None:
    """Configure root logger with standardized formatting.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path to write logs in addition to stdout.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to prevent duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = CustomFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Convenience getter for named loggers.

    Args:
        name: Name of module or component.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
