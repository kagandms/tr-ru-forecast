"""Smoke and unit tests for configuration settings."""

from pathlib import Path

from tr_ru_forecast.config import Settings
from tr_ru_forecast.core.constants import (
    DIRECTIONS,
    FORECAST_HORIZONS,
    HOLDOUT_MONTHS_COUNT,
    START_PERIOD,
)


def test_settings_defaults(settings: Settings) -> None:
    """Verify settings defaults match frozen specifications."""
    assert settings.start_period == START_PERIOD
    assert settings.start_period == "2012-01"
    assert settings.holdout_months == HOLDOUT_MONTHS_COUNT
    assert settings.holdout_months == 12
    assert settings.forecast_horizons == FORECAST_HORIZONS
    assert settings.forecast_horizons == [1, 3, 6]
    assert settings.max_sectors_per_direction == 6


def test_settings_paths_resolve(settings: Settings) -> None:
    """Verify critical paths exist or are resolved properly."""
    assert isinstance(settings.project_root, Path)
    assert settings.project_root.exists()
    assert (settings.project_root / "specification").exists()
    assert settings.data_dir.name == "data"
    assert settings.raw_data_dir.name == "raw"


def test_trade_directions_spec() -> None:
    """Verify bilateral directions match specification."""
    direction_values = [d.value for d in DIRECTIONS]
    assert "TR_RU" in direction_values
    assert "RU_TR" in direction_values
    assert len(direction_values) == 2
