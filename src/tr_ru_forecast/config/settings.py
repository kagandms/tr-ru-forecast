"""Application and pipeline configuration management."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from tr_ru_forecast.core.constants import (
    DEVELOPMENT_DATA_END,
    DEVELOPMENT_DATA_START,
    FORECAST_HORIZONS,
    HOLDOUT_DATA_END,
    HOLDOUT_DATA_START,
    HOLDOUT_MONTHS_COUNT,
    MAX_SECTORS_PER_DIRECTION,
    MODELING_DATA_END,
    MODELING_DATA_START,
    START_PERIOD,
)

# Project root directory (3 levels up from src/tr_ru_forecast/config/settings.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Central configuration settings for TR-RU forecasting platform."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application Environment
    app_env: str = Field(
        default="development", description="Environment: development, staging, production"
    )
    log_level: str = Field(default="INFO", description="Global log level")
    debug: bool = Field(default=False, description="Debug mode flag")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API listen host")
    api_port: int = Field(default=8000, description="API listen port")

    # Directory Paths
    project_root: Path = Field(default=PROJECT_ROOT, description="Project root path")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", description="Base data directory")
    raw_data_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "raw", description="Raw trade and macro data"
    )
    interim_data_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "interim", description="Interim processed datasets"
    )
    processed_data_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "processed", description="Final modeling features"
    )
    models_dir: Path = Field(default=PROJECT_ROOT / "models", description="Trained model artifacts")

    # Domain Baseline Settings (Frozen Spec v1.0 & v1.1)
    start_period: str = Field(default=START_PERIOD, description="Default starting month YYYY-MM")
    modeling_data_start: str = Field(
        default=MODELING_DATA_START, description="Full verified modeling dataset start period"
    )
    modeling_data_end: str = Field(
        default=MODELING_DATA_END,
        description="Latest complete available reporting month for modeling cutoff",
    )
    development_data_start: str = Field(
        default=DEVELOPMENT_DATA_START, description="Development dataset start period"
    )
    development_data_end: str = Field(
        default=DEVELOPMENT_DATA_END,
        description="Development dataset cutoff period (before holdout)",
    )
    holdout_data_start: str = Field(
        default=HOLDOUT_DATA_START, description="Untouched test holdout start period"
    )
    holdout_data_end: str = Field(
        default=HOLDOUT_DATA_END, description="Untouched test holdout end period"
    )
    holdout_months: int = Field(
        default=HOLDOUT_MONTHS_COUNT, description="Final holdout length in complete months"
    )
    forecast_horizons: list[int] = Field(
        default=FORECAST_HORIZONS, description="Forecasting horizons in months"
    )
    max_sectors_per_direction: int = Field(
        default=MAX_SECTORS_PER_DIRECTION,
        description="Maximum HS2 chapters per trade direction",
    )

    # External Credentials / Storage Placeholders
    database_url: str | None = Field(default=None, description="Database connection URL")
    supabase_url: str | None = Field(default=None, description="Supabase project URL")
    supabase_key: str | None = Field(default=None, description="Supabase API key")
    comtrade_api_key: str | None = Field(default=None, description="UN Comtrade subscription key")

    # UN Comtrade Ingestion Settings
    comtrade_base_url: str = Field(
        default="https://comtradeapi.un.org/public/v1/preview/C/M/HS",
        description="Default public UN Comtrade endpoint",
    )
    comtrade_subscription_url: str = Field(
        default="https://comtradeapi.un.org/data/v1/get/C/M/HS",
        description="Subscription UN Comtrade endpoint",
    )
    comtrade_timeout_seconds: float = Field(
        default=30.0, description="HTTP timeout for Comtrade requests"
    )
    comtrade_max_retries: int = Field(
        default=3, description="Maximum retries for Comtrade requests"
    )
    comtrade_rate_limit_delay: float = Field(
        default=0.5, description="Delay between requests in seconds"
    )
    comtrade_partner2_code: str = Field(
        default="0", description="2nd partner code filter (0 = Total/World)"
    )
    comtrade_customs_code: str = Field(
        default="C00", description="Customs procedure filter (C00 = Total)"
    )
    comtrade_mot_code: str = Field(
        default="0", description="Mode of transport filter (0 = All modes)"
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton getter for application settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
