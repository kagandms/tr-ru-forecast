"""Macroeconomic source registry models and loader.

Validates the version-controlled macro source registry configuration according to
Specification Pack v1.1.
"""

from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from tr_ru_forecast.config.settings import get_settings

REQUIRED_INDICATOR_IDS: Final[set[str]] = {
    "usd_try",
    "rub_usd",
    "brent",
    "tr_ip",
    "ru_ip",
}


class MacroIndicatorSource(BaseModel):
    """Configuration metadata for an individual macroeconomic series."""

    indicator_id: str
    indicator_name: str
    canonical_semantic: str
    canonical_unit: str
    quotation_convention: str | None = None
    base_year: str | None = None
    seasonal_adjustment: str | None = None
    origin_provider: str | None = None
    distribution_provider: str | None = None
    underlying_national_source: str | None = None
    composite_structure: bool = False
    linkage_method: str | None = None
    linkage_factor: float | None = None
    linkage_switch_date: str | None = None
    linkage_version: str | None = None
    overlap_window: dict[str, Any] | None = None
    historical_segment: dict[str, Any] | None = None
    recent_segment: dict[str, Any] | None = None
    source_provider: str
    source_dataset_name: str
    source_series_id: str
    source_url_or_endpoint: str
    source_frequency: str = "Monthly"
    target_frequency: str = "Monthly"
    aggregation_rule: str
    reference_period_semantics: str
    release_lag_rule: str
    availability_date_rule: str
    availability_exact_or_approximate: str
    fallback_lag_months: int = Field(ge=1, description="Minimum fallback lag in months (must be >= 1)")
    transformation: str = "raw_level"
    start_period_requested: str = "2012-01"
    retrieval_method: str = "fred_csv_download"
    credential_env_var_if_any: str | None = None
    status: str = "VERIFIED"
    notes: str | None = None

    @field_validator("fallback_lag_months")
    @classmethod
    def validate_fallback_lag(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Specification invariant violated: fallback_lag_months must be >= 1 month.")
        return v

    @field_validator("canonical_unit")
    @classmethod
    def validate_unit_orientation(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Canonical unit must not be empty.")
        return v

    @model_validator(mode="after")
    def validate_indicator_specific_semantics(self) -> "MacroIndicatorSource":
        # Invariant: rub_usd must explicitly quote RUB per 1 USD
        if self.indicator_id == "rub_usd":
            if self.canonical_unit != "RUB per 1 USD":
                raise ValueError(
                    f"Orientation violation: rub_usd canonical_unit must be 'RUB per 1 USD', "
                    f"got '{self.canonical_unit}'"
                )
            if self.quotation_convention != "USD/RUB":
                raise ValueError(
                    f"Quotation convention violation: rub_usd quotation_convention must be 'USD/RUB', "
                    f"got '{self.quotation_convention}'"
                )

        # Invariant: tr_ip must be 2021=100 base year and TÜİK origin
        if self.indicator_id == "tr_ip":
            if "2021=100" not in self.canonical_unit:
                raise ValueError(
                    f"Base year violation: tr_ip canonical_unit must reference '2021=100', "
                    f"got '{self.canonical_unit}'"
                )
            if self.origin_provider and "TÜİK" not in self.origin_provider and "Turkish Statistical" not in self.origin_provider:
                raise ValueError(
                    f"Provenance violation: tr_ip origin_provider must be TÜİK, "
                    f"got '{self.origin_provider}'"
                )

        # Invariant: ru_ip linked composite validation
        if self.indicator_id == "ru_ip" and self.status == "VERIFIED_LINKED_OPTION_A":
            if self.linkage_switch_date != "2022-01":
                raise ValueError(
                    f"Linkage switch date violation: ru_ip linkage_switch_date must be '2022-01', "
                    f"got '{self.linkage_switch_date}'"
                )
            if self.linkage_factor is None or self.linkage_factor <= 0:
                raise ValueError(
                    f"Linkage factor violation: ru_ip linkage_factor must be positive float, "
                    f"got '{self.linkage_factor}'"
                )
            if not self.underlying_national_source or "Rosstat" not in self.underlying_national_source:
                raise ValueError(
                    f"Provenance violation: ru_ip underlying_national_source must specify Rosstat, "
                    f"got '{self.underlying_national_source}'"
                )

        return self


class MacroSourceRegistry(BaseModel):
    """Top-level container for all macroeconomic data sources."""

    version: str
    spec_version: str
    created_at: str
    description: str
    indicators: dict[str, MacroIndicatorSource]

    @field_validator("indicators")
    @classmethod
    def validate_required_indicators(
        cls, v: dict[str, MacroIndicatorSource]
    ) -> dict[str, MacroIndicatorSource]:
        keys = set(v.keys())
        if keys != REQUIRED_INDICATOR_IDS:
            missing = REQUIRED_INDICATOR_IDS - keys
            extra = keys - REQUIRED_INDICATOR_IDS
            raise ValueError(
                f"Source registry must contain exactly the 5 initial indicators. "
                f"Missing: {missing}, Extra: {extra}"
            )
        # Ensure indicator_id matches dict key
        for key, ind in v.items():
            if ind.indicator_id != key:
                raise ValueError(
                    f"Indicator ID '{ind.indicator_id}' does not match dictionary key '{key}'"
                )
        return v


def load_macro_registry(config_path: Path | None = None) -> MacroSourceRegistry:
    """Load and validate macroeconomic source registry from YAML configuration.

    Args:
        config_path: Optional path to YAML registry file. If omitted, uses default.

    Returns:
        Validated MacroSourceRegistry instance.
    """
    settings = get_settings()
    path = config_path or (settings.project_root / "configs" / "macro_sources_v1_1.yaml")

    if not path.exists():
        raise FileNotFoundError(f"Macro source registry configuration not found at: {path}")

    with open(path, encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)

    return MacroSourceRegistry.model_validate(raw_dict)


load_macro_source_registry = load_macro_registry
