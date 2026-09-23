"""Unit tests for UN Comtrade data transformation and normalization (M1)."""

import pytest

from tr_ru_forecast.data.ingestion.transformer import (
    STANDARDIZED_COLUMNS,
    ConflictingDuplicateError,
    deduplicate_records,
    normalize_hs2,
    normalize_period,
    transform_raw_record,
)


def test_period_normalization() -> None:
    """Test period normalization to YYYY-MM format."""
    assert normalize_period("202301") == "2023-01"
    assert normalize_period("2023-01") == "2023-01"
    assert normalize_period(202301) == "2023-01"
    assert normalize_period("invalid") is None
    assert normalize_period(None) is None


def test_hs2_normalization() -> None:
    """Test HS2 chapter code and TOTAL detection."""
    # TOTAL cases
    code, is_total = normalize_hs2("TOTAL")
    assert code == "TOTAL"
    assert is_total is True

    code, is_total = normalize_hs2("TOTALS")
    assert code == "TOTAL"
    assert is_total is True

    code, is_total = normalize_hs2("999999")
    assert code == "TOTAL"
    assert is_total is True

    # HS2 numeric cases
    code, is_total = normalize_hs2("01")
    assert code == "01"
    assert is_total is False

    code, is_total = normalize_hs2("1")
    assert code == "01"
    assert is_total is False

    code, is_total = normalize_hs2("99")
    assert code == "99"
    assert is_total is False

    # Chapter 77 is reserved in HS classification
    code, is_total = normalize_hs2("77")
    assert code is None
    assert is_total is False

    # Non-HS2 cases (e.g. HS4 or HS6)
    code, is_total = normalize_hs2("0101")
    assert code is None
    assert is_total is False

    code, is_total = normalize_hs2(None)
    assert code is None
    assert is_total is False


def test_tr_ru_direction_mapping() -> None:
    """Test that Turkey -> Russia exporter-reported row maps to TR_RU."""
    raw = {
        "period": "202301",
        "reporterCode": 792,
        "reporterDesc": "Türkiye",
        "partnerCode": 643,
        "partnerDesc": "Russian Federation",
        "flowCode": "X",
        "cmdCode": "01",
        "cmdDesc": "Live animals",
        "primaryValue": 53908.0,
        "classificationCode": "H6",
    }
    rec = transform_raw_record(raw)
    assert rec is not None
    assert rec["direction"] == "TR_RU"
    assert rec["reporter_code"] == 792
    assert rec["reporter_name"] == "Türkiye"
    assert rec["partner_code"] == 643
    assert rec["partner_name"] == "Russian Federation"
    assert rec["flow"] == "Export"
    assert rec["hs2"] == "01"
    assert rec["is_total"] is False
    assert rec["trade_value_usd"] == 53908.0
    assert rec["data_status"] == "observed"


def test_ru_tr_direction_mapping() -> None:
    """Test that Russia -> Turkey exporter-reported row maps to RU_TR."""
    raw = {
        "period": "202101",
        "reporterCode": 643,
        "reporterDesc": "Russian Federation",
        "partnerCode": 792,
        "partnerDesc": "Türkiye",
        "flowCode": "X",
        "cmdCode": "TOTAL",
        "primaryValue": 1700250824.0,
        "classificationCode": "H5",
    }
    rec = transform_raw_record(raw)
    assert rec is not None
    assert rec["direction"] == "RU_TR"
    assert rec["reporter_code"] == 643
    assert rec["partner_code"] == 792
    assert rec["flow"] == "Export"
    assert rec["hs2"] == "TOTAL"
    assert rec["is_total"] is True
    assert rec["trade_value_usd"] == 1700250824.0
    assert rec["data_status"] == "observed"


def test_missing_value_not_converted_to_zero() -> None:
    """Test critical rule: Missing != Zero."""
    # 1. Truly missing value (None)
    raw_missing = {
        "period": "202301",
        "reporterCode": 792,
        "partnerCode": 643,
        "flowCode": "X",
        "cmdCode": "02",
        "primaryValue": None,
        "fobvalue": None,
        "classificationCode": "H6",
    }
    rec_missing = transform_raw_record(raw_missing)
    assert rec_missing is not None
    assert rec_missing["trade_value_usd"] is None
    assert rec_missing["data_status"] == "missing"

    # 2. Reported Zero (0.0)
    raw_zero = {
        "period": "202301",
        "reporterCode": 792,
        "partnerCode": 643,
        "flowCode": "X",
        "cmdCode": "02",
        "primaryValue": 0.0,
        "classificationCode": "H6",
    }
    rec_zero = transform_raw_record(raw_zero)
    assert rec_zero is not None
    assert rec_zero["trade_value_usd"] == 0.0
    assert rec_zero["data_status"] == "reported_zero"

    # 3. Observed positive value
    raw_obs = {
        "period": "202301",
        "reporterCode": 792,
        "partnerCode": 643,
        "flowCode": "X",
        "cmdCode": "02",
        "primaryValue": 12500.0,
        "classificationCode": "H6",
    }
    rec_obs = transform_raw_record(raw_obs)
    assert rec_obs is not None
    assert rec_obs["trade_value_usd"] == 12500.0
    assert rec_obs["data_status"] == "observed"


def test_classification_metadata_preserved() -> None:
    """Verify that HS classification code/revision and original flag are retained."""
    raw = {
        "period": "202301",
        "reporterCode": 792,
        "partnerCode": 643,
        "cmdCode": "05",
        "classificationCode": "H6",
        "classificationSearchCode": "HS",
        "isOriginalClassification": True,
        "primaryValue": 1000.0,
    }
    rec = transform_raw_record(raw)
    assert rec is not None
    assert rec["classification"] == "HS"
    assert rec["classification_code"] == "H6"
    assert rec["is_original_classification"] is True


def test_malformed_source_row_handling() -> None:
    """Verify that malformed or non-dict rows are safely skipped."""
    assert transform_raw_record(None) is None  # type: ignore[arg-type]
    assert transform_raw_record({}) is None
    assert transform_raw_record({"period": "202301"}) is None
    assert transform_raw_record({"cmdCode": "01"}) is None

    # Invalid non-integer country code
    raw_invalid_code = {
        "period": "202301",
        "reporterCode": "ABC",
        "partnerCode": 643,
        "cmdCode": "01",
    }
    assert transform_raw_record(raw_invalid_code) is None


def test_standardized_schema_validation() -> None:
    """Verify that transformed record matches exact standardized contract fields."""
    raw = {
        "period": "202301",
        "reporterCode": 792,
        "partnerCode": 643,
        "cmdCode": "TOTAL",
        "primaryValue": 1000000.0,
        "classificationCode": "H6",
        "classificationSearchCode": "HS",
        "isOriginalClassification": True,
    }
    rec = transform_raw_record(raw)
    assert rec is not None
    assert list(rec.keys()) == STANDARDIZED_COLUMNS


def test_exact_duplicate_collapse() -> None:
    """Verify that identical duplicates collapse deterministically without conflict."""
    rec1 = {
        "period": "2023-01",
        "direction": "TR_RU",
        "reporter_code": 792,
        "partner_code": 643,
        "flow": "Export",
        "hs2": "01",
        "classification": "HS",
        "classification_code": "H6",
        "trade_value_usd": 100.0,
    }
    rec2 = dict(rec1)
    rec3 = dict(rec1)

    result = deduplicate_records([rec1, rec2, rec3])
    assert len(result.clean_records) == 1
    assert result.exact_duplicate_count == 2
    assert result.conflict_count == 0
    assert result.conflict_keys == []
    assert len(result.quarantined_records) == 0
    assert result.clean_records[0]["trade_value_usd"] == 100.0


def test_conflicting_duplicate_quarantine() -> None:
    """Verify conflicting duplicates are NOT auto-resolved via max() but quarantined."""
    rec_base = {
        "period": "2023-01",
        "direction": "TR_RU",
        "reporter_code": 792,
        "partner_code": 643,
        "flow": "Export",
        "hs2": "01",
        "classification": "HS",
        "classification_code": "H6",
        "trade_value_usd": 100.0,
    }
    rec_conflict = dict(rec_base)
    rec_conflict["trade_value_usd"] = 250.0  # Conflicting value

    result = deduplicate_records([rec_base, rec_conflict])

    # Must NOT select max(250.0) or min(100.0) - must quarantine!
    assert len(result.clean_records) == 0
    assert result.conflict_count == 1
    assert len(result.conflict_keys) == 1
    assert "2023-01:792:643:Export:01:H6" in result.conflict_keys[0]
    assert len(result.quarantined_records) == 2


def test_conflicting_duplicate_fail_closed() -> None:
    """Verify fail_closed=True raises ConflictingDuplicateError immediately."""
    rec1 = {
        "period": "2023-01",
        "reporter_code": 792,
        "partner_code": 643,
        "flow": "Export",
        "hs2": "01",
        "classification": "HS",
        "classification_code": "H6",
        "trade_value_usd": 100.0,
    }
    rec2 = dict(rec1)
    rec2["trade_value_usd"] = 999.0

    with pytest.raises(ConflictingDuplicateError):
        deduplicate_records([rec1, rec2], fail_closed=True)
