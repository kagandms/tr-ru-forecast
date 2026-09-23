"""Macroeconomic data foundation package (Milestone M4).

Provides authoritative source registry, data client, transformation, publication-safe
as-of access contracts, and end-to-end ingestion pipeline under Specification v1.1.
"""

from tr_ru_forecast.data.macro.client import FredMacroClient
from tr_ru_forecast.data.macro.pipeline import MacroDataPipeline
from tr_ru_forecast.data.macro.publication import (
    PublicationSafeAccessor,
    compute_conservative_availability_date,
    normalize_forecast_origin,
)
from tr_ru_forecast.data.macro.registry import (
    REQUIRED_INDICATOR_IDS,
    MacroIndicatorSource,
    MacroSourceRegistry,
    load_macro_source_registry,
)
from tr_ru_forecast.data.macro.transformer import (
    CANONICAL_MACRO_COLUMNS,
    compute_indicator_coverage_audit,
    parse_raw_fred_csv,
)

__all__ = [
    "CANONICAL_MACRO_COLUMNS",
    "REQUIRED_INDICATOR_IDS",
    "FredMacroClient",
    "MacroDataPipeline",
    "MacroIndicatorSource",
    "MacroSourceRegistry",
    "PublicationSafeAccessor",
    "compute_conservative_availability_date",
    "compute_indicator_coverage_audit",
    "load_macro_source_registry",
    "normalize_forecast_origin",
    "parse_raw_fred_csv",
]
