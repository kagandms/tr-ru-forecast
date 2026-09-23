"""Unit tests for Specification v1.1 Forecast Target Source Policy.

Verifies:
1. TR_RU forecast target is Türkiye exporter-reported series (Reporter 792, Partner 643, Flow X).
2. RU_TR forecast target is Türkiye importer-reported series (Reporter 792, Partner 643, Flow M).
3. Russia-reported flows are preserved strictly as REFERENCE_FLOWS.
4. Final holdout length is strictly 12 months (2025-01 -> 2025-12).
5. Development dataset cutoff is 2024-12 (156 development months).
6. Latest complete available reporting month cutoff is 2025-12.
"""

from tr_ru_forecast.config.settings import Settings, get_settings
from tr_ru_forecast.core.constants import (
    CANONICAL_FLOWS,
    DEVELOPMENT_DATA_END,
    DEVELOPMENT_DATA_START,
    FLOW_EXPORT_CODE,
    FLOW_EXPORT_NAME,
    FLOW_IMPORT_CODE,
    FLOW_IMPORT_NAME,
    FORECAST_TARGET_FLOWS,
    HOLDOUT_DATA_END,
    HOLDOUT_DATA_START,
    HOLDOUT_MONTHS_COUNT,
    MIRROR_FLOWS,
    MODELING_DATA_END,
    MODELING_DATA_START,
    REFERENCE_FLOWS,
    RUSSIA_M49_CODE,
    START_PERIOD,
    TURKEY_M49_CODE,
    TradeDirection,
)


def test_forecast_target_tr_ru_mapping():
    """TR_RU forecast target must be Türkiye exporter-reported flow."""
    target = FORECAST_TARGET_FLOWS[TradeDirection.TR_RU]
    assert target["reporter_code"] == TURKEY_M49_CODE
    assert target["partner_code"] == RUSSIA_M49_CODE
    assert target["flow"] == FLOW_EXPORT_NAME
    assert target["flow_code"] == FLOW_EXPORT_CODE
    assert target["direction"] == "TR_RU"


def test_forecast_target_ru_tr_mapping():
    """RU_TR forecast target must be Türkiye importer-reported flow (v1.1 policy)."""
    target = FORECAST_TARGET_FLOWS[TradeDirection.RU_TR]
    assert target["reporter_code"] == TURKEY_M49_CODE
    assert target["partner_code"] == RUSSIA_M49_CODE
    assert target["flow"] == FLOW_IMPORT_NAME
    assert target["flow_code"] == FLOW_IMPORT_CODE
    assert target["direction"] == "RU_TR"


def test_both_forecast_targets_are_official_turkiye_reported():
    """Both forecast targets MUST have Türkiye (792) as reporter and Russia (643) as partner."""
    for direction in [TradeDirection.TR_RU, TradeDirection.RU_TR]:
        target = FORECAST_TARGET_FLOWS[direction]
        assert target["reporter_code"] == TURKEY_M49_CODE
        assert target["partner_code"] == RUSSIA_M49_CODE


def test_reference_flows_are_russia_reported():
    """Reference flows for research/quality must be Russia-reported (643)."""
    tr_ru_ref = REFERENCE_FLOWS[TradeDirection.TR_RU]
    assert tr_ru_ref["reporter_code"] == RUSSIA_M49_CODE
    assert tr_ru_ref["partner_code"] == TURKEY_M49_CODE
    assert tr_ru_ref["flow_code"] == FLOW_IMPORT_CODE  # Russia import from TR

    ru_tr_ref = REFERENCE_FLOWS[TradeDirection.RU_TR]
    assert ru_tr_ref["reporter_code"] == RUSSIA_M49_CODE
    assert ru_tr_ref["partner_code"] == TURKEY_M49_CODE
    assert ru_tr_ref["flow_code"] == FLOW_EXPORT_CODE  # Russia export to TR


def test_canonical_and_mirror_provenance_preserved():
    """Ingestion provenance dictionaries must remain intact for backward compatibility."""
    assert CANONICAL_FLOWS[TradeDirection.TR_RU]["reporter_code"] == TURKEY_M49_CODE
    assert CANONICAL_FLOWS[TradeDirection.RU_TR]["reporter_code"] == RUSSIA_M49_CODE
    assert MIRROR_FLOWS[TradeDirection.TR_RU]["reporter_code"] == RUSSIA_M49_CODE
    assert MIRROR_FLOWS[TradeDirection.RU_TR]["reporter_code"] == TURKEY_M49_CODE


def test_holdout_length_is_strictly_12_months():
    """Holdout length must be exactly 12 months (2025-01 -> 2025-12)."""
    assert HOLDOUT_MONTHS_COUNT == 12
    assert HOLDOUT_DATA_START == "2025-01"
    assert HOLDOUT_DATA_END == "2025-12"


def test_modeling_cutoff_and_development_ranges():
    """Verify development window and modeling cutoff dates."""
    assert START_PERIOD == "2012-01"
    assert MODELING_DATA_START == "2012-01"
    assert MODELING_DATA_END == "2025-12"
    assert DEVELOPMENT_DATA_START == "2012-01"
    assert DEVELOPMENT_DATA_END == "2024-12"


def test_settings_exposure():
    """Verify settings exposes the v1.1 modeling configuration."""
    settings = Settings()
    assert settings.holdout_months == 12
    assert settings.modeling_data_start == "2012-01"
    assert settings.modeling_data_end == "2025-12"
    assert settings.development_data_start == "2012-01"
    assert settings.development_data_end == "2024-12"
    assert settings.holdout_data_start == "2025-01"
    assert settings.holdout_data_end == "2025-12"

    getter_settings = get_settings()
    assert getter_settings.modeling_data_end == "2025-12"
