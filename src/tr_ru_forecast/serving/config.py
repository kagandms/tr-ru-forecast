"""Environment-configured artifact locations and explicit browser origins."""

from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tr_ru_forecast.config.settings import PROJECT_ROOT


class ServingSettings(BaseSettings):
    """Serving-only settings; M1–M11 pipeline settings and inputs remain untouched."""

    model_config = SettingsConfigDict(env_prefix="SERVING_", extra="ignore", frozen=True)
    artifact_root: Path = PROJECT_ROOT / "data/processed"
    config_root: Path = PROJECT_ROOT / "configs"
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)

    @model_validator(mode="after")
    def validate_origins(self) -> ServingSettings:
        """Validate configured origins. Args: self. Returns: settings. Raises: ValueError."""
        if any(origin == "*" or not origin.startswith(("http://", "https://")) for origin in self.cors_origins):
            raise ValueError("CORS requires explicit HTTP(S) frontend origins")
        return self
