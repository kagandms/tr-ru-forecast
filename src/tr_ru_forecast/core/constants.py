"""Frozen constants defined in Specification Pack v1.0.

These values reflect frozen domain decisions from:
- PROJECT_SPEC.md
- DATA_SPEC.md
- EXPERIMENT_PROTOCOL.md
"""

from enum import StrEnum
from typing import Final


class TradeDirection(StrEnum):
    """Canonical bilateral trade directions."""

    TR_RU = "TR_RU"
    RU_TR = "RU_TR"


DIRECTIONS: Final[list[TradeDirection]] = [
    TradeDirection.TR_RU,
    TradeDirection.RU_TR,
]

# UN M49 Country Codes and Official Names
TURKEY_M49_CODE: Final[int] = 792
RUSSIA_M49_CODE: Final[int] = 643
TURKEY_NAME: Final[str] = "Türkiye"
RUSSIA_NAME: Final[str] = "Russian Federation"

# Flow codes
FLOW_EXPORT_CODE: Final[str] = "X"
FLOW_EXPORT_NAME: Final[str] = "Export"
FLOW_IMPORT_CODE: Final[str] = "M"
FLOW_IMPORT_NAME: Final[str] = "Import"

# Canonical trade flow mappings (Exporter-reported historical storage provenance)
CANONICAL_FLOWS: Final[dict[TradeDirection, dict[str, str | int]]] = {
    TradeDirection.TR_RU: {
        "direction": TradeDirection.TR_RU.value,
        "reporter_code": TURKEY_M49_CODE,
        "reporter_name": TURKEY_NAME,
        "partner_code": RUSSIA_M49_CODE,
        "partner_name": RUSSIA_NAME,
        "flow": FLOW_EXPORT_NAME,
        "flow_code": FLOW_EXPORT_CODE,
    },
    TradeDirection.RU_TR: {
        "direction": TradeDirection.RU_TR.value,
        "reporter_code": RUSSIA_M49_CODE,
        "reporter_name": RUSSIA_NAME,
        "partner_code": TURKEY_M49_CODE,
        "partner_name": TURKEY_NAME,
        "flow": FLOW_EXPORT_NAME,
        "flow_code": FLOW_EXPORT_CODE,
    },
}

# Specification v1.1 Forecast Target Flow Mappings (Türkiye-reported official series)
FORECAST_TARGET_FLOWS: Final[dict[TradeDirection, dict[str, str | int]]] = {
    TradeDirection.TR_RU: {
        "direction": TradeDirection.TR_RU.value,
        "reporter_code": TURKEY_M49_CODE,
        "reporter_name": TURKEY_NAME,
        "partner_code": RUSSIA_M49_CODE,
        "partner_name": RUSSIA_NAME,
        "flow": FLOW_EXPORT_NAME,
        "flow_code": FLOW_EXPORT_CODE,
        "semantic": "Türkiye exports to Russia",
    },
    TradeDirection.RU_TR: {
        "direction": TradeDirection.RU_TR.value,
        "reporter_code": TURKEY_M49_CODE,
        "reporter_name": TURKEY_NAME,
        "partner_code": RUSSIA_M49_CODE,
        "partner_name": RUSSIA_NAME,
        "flow": FLOW_IMPORT_NAME,
        "flow_code": FLOW_IMPORT_CODE,
        "semantic": "Türkiye imports from Russia",
    },
}

# Specification v1.1 Reference / Mirror Flows (for quality/research layer, not forecast targets)
REFERENCE_FLOWS: Final[dict[TradeDirection, dict[str, str | int]]] = {
    TradeDirection.TR_RU: {
        "direction": TradeDirection.TR_RU.value,
        "reporter_code": RUSSIA_M49_CODE,
        "reporter_name": RUSSIA_NAME,
        "partner_code": TURKEY_M49_CODE,
        "partner_name": TURKEY_NAME,
        "flow": FLOW_IMPORT_NAME,
        "flow_code": FLOW_IMPORT_CODE,
        "semantic": "Russia imports from Türkiye",
    },
    TradeDirection.RU_TR: {
        "direction": TradeDirection.RU_TR.value,
        "reporter_code": RUSSIA_M49_CODE,
        "reporter_name": RUSSIA_NAME,
        "partner_code": TURKEY_M49_CODE,
        "partner_name": TURKEY_NAME,
        "flow": FLOW_EXPORT_NAME,
        "flow_code": FLOW_EXPORT_CODE,
        "semantic": "Russia exports to Türkiye",
    },
}

# Mirror trade flow mappings (Partner importer-reported, preserving physical direction)
MIRROR_FLOWS: Final[dict[TradeDirection, dict[str, str | int]]] = {
    TradeDirection.TR_RU: {
        "direction": TradeDirection.TR_RU.value,
        "reporter_code": RUSSIA_M49_CODE,
        "reporter_name": RUSSIA_NAME,
        "partner_code": TURKEY_M49_CODE,
        "partner_name": TURKEY_NAME,
        "flow": FLOW_IMPORT_NAME,
        "flow_code": FLOW_IMPORT_CODE,
    },
    TradeDirection.RU_TR: {
        "direction": TradeDirection.RU_TR.value,
        "reporter_code": TURKEY_M49_CODE,
        "reporter_name": TURKEY_NAME,
        "partner_code": RUSSIA_M49_CODE,
        "partner_name": RUSSIA_NAME,
        "flow": FLOW_IMPORT_NAME,
        "flow_code": FLOW_IMPORT_CODE,
    },
}


class PairStatus(StrEnum):
    """Observation presence status for canonical vs mirror pairing."""

    BOTH_PRESENT = "BOTH_PRESENT"
    CANONICAL_ONLY = "CANONICAL_ONLY"
    MIRROR_ONLY = "MIRROR_ONLY"
    BOTH_MISSING = "BOTH_MISSING"


class GapFlag(StrEnum):
    """Observation pairing and discrepancy flags for mirror vs canonical values.

    Default policy uses neutral data presence states:
        - PAIRED: Both canonical and mirror data are observed.
        - UNPAIRED: Missing on canonical, mirror, or both.

    Optional threshold states (only produced if explicit thresholds are configured):
        - NONE, MODERATE, LARGE
    """

    PAIRED = "PAIRED"
    UNPAIRED = "UNPAIRED"
    NONE = "NONE"
    MODERATE = "MODERATE"
    LARGE = "LARGE"


# Standardized columns for mirror_monthly analytical dataset
MIRROR_MONTHLY_COLUMNS: Final[list[str]] = [
    "period",
    "direction",
    "hs2",
    "canonical_value_usd",
    "mirror_value_usd",
    "mirror_gap_usd",
    "mirror_gap_pct",
    "abs_mirror_gap_pct",
    "canonical_missing",
    "mirror_missing",
    "pair_status",
    "canonical_classification",
    "canonical_classification_code",
    "canonical_is_original_classification",
    "mirror_classification",
    "mirror_classification_code",
    "mirror_is_original_classification",
    "classification_mismatch",
    "canonical_retrieved_at",
    "mirror_retrieved_at",
    "gap_flag",
]

# Primary time scope: 2012-01 -> latest complete month
START_PERIOD: Final[str] = "2012-01"

# Specification v1.1 Modeling & Validation Cutoff Constants
MODELING_DATA_START: Final[str] = "2012-01"
MODELING_DATA_END: Final[str] = "2025-12"  # Latest complete available reporting month
DEVELOPMENT_DATA_START: Final[str] = "2012-01"
DEVELOPMENT_DATA_END: Final[str] = "2024-12"  # 156 development months
HOLDOUT_DATA_START: Final[str] = "2025-01"
HOLDOUT_DATA_END: Final[str] = "2025-12"  # Strictly 12 untouched holdout months

# Mandatory forecast horizons
FORECAST_HORIZONS: Final[list[int]] = [1, 3, 6]

# Final holdout length in complete months
HOLDOUT_MONTHS_COUNT: Final[int] = 12

# Minimum history required for candidate sector eligibility (months)
MIN_SECTOR_HISTORY_MONTHS: Final[int] = 96

# Max HS2 sectors per direction (excluding TOTAL)
MAX_SECTORS_PER_DIRECTION: Final[int] = 6

# Mandatory model families
MANDATORY_MODELS: Final[list[str]] = [
    "seasonal_naive",
    "sarima",
    "sarimax",
    "xgboost",
]

# Model complexity hierarchy for champion selection rule:
# Seasonal Naive < SARIMA < SARIMAX < XGBoost
MODEL_COMPLEXITY_ORDER: Final[dict[str, int]] = {
    "seasonal_naive": 1,
    "sarima": 2,
    "sarimax": 3,
    "xgboost": 4,
}

# Prediction intervals required
PREDICTION_INTERVALS: Final[list[float]] = [0.80, 0.95]
