"""Data ingestion package for UN Comtrade bilateral trade data."""

from tr_ru_forecast.data.ingestion.comtrade_client import (
    ComtradeClient,
    ComtradeConnectionError,
    ComtradeError,
    ComtradeRateLimitError,
)
from tr_ru_forecast.data.ingestion.pipeline import (
    TradeIngestionPipeline,
    generate_period_range,
    resolve_latest_complete_month,
)
from tr_ru_forecast.data.ingestion.transformer import (
    STANDARDIZED_COLUMNS,
    ConflictingDuplicateError,
    DeduplicationResult,
    deduplicate_records,
    normalize_hs2,
    normalize_period,
    transform_raw_record,
)

__all__ = [
    "ComtradeClient",
    "ComtradeConnectionError",
    "ComtradeError",
    "ComtradeRateLimitError",
    "ConflictingDuplicateError",
    "DeduplicationResult",
    "STANDARDIZED_COLUMNS",
    "TradeIngestionPipeline",
    "deduplicate_records",
    "generate_period_range",
    "normalize_hs2",
    "normalize_period",
    "resolve_latest_complete_month",
    "transform_raw_record",
]
