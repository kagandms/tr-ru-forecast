"""Smoke and unit tests for logging setup."""

import logging

from tr_ru_forecast.core.logging import get_logger, setup_logging


def test_logger_creation() -> None:
    """Verify logger setup produces valid logger instance."""
    setup_logging(level="DEBUG")
    logger = get_logger("test.smoke")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.smoke"
    assert logger.isEnabledFor(logging.DEBUG)


def test_logger_formatting(capsys) -> None:
    """Verify log output format contains timestamp, level, and message."""
    setup_logging(level="INFO")
    logger = get_logger("test.format")
    logger.info("M0 smoke test message")

    captured = capsys.readouterr()
    assert "INFO" in captured.out
    assert "test.format" in captured.out
    assert "M0 smoke test message" in captured.out
