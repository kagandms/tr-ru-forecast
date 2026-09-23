"""Unit and integration tests for bilateral mirror statistics and data quality layer (M2).

Verifies strict conformance to DATA_SPEC.md and PROJECT_SPEC.md:
- Direction mapping for mirror flows (preserving physical trade direction)
- Strict missing != zero policy
- Exact duplicate collapse and conflicting duplicate quarantine
- Full outer join pairing (BOTH_PRESENT, CANONICAL_ONLY, MIRROR_ONLY, BOTH_MISSING)
- Safe gap calculations and zero-division protection
- Classification mismatch detection
- Canonical target immutability
- Coverage analyzer metrics and empirical suspension audit
"""

import httpx
import pandas as pd
import pytest

from tr_ru_forecast.config import Settings
from tr_ru_forecast.core.constants import (
    FLOW_IMPORT_CODE,
    FLOW_IMPORT_NAME,
    MIRROR_FLOWS,
    MIRROR_MONTHLY_COLUMNS,
    RUSSIA_M49_CODE,
    TURKEY_M49_CODE,
    GapFlag,
    PairStatus,
    TradeDirection,
)
from tr_ru_forecast.data.ingestion.comtrade_client import ComtradeClient
from tr_ru_forecast.data.ingestion.transformer import ConflictingDuplicateError
from tr_ru_forecast.data.quality.coverage_analyzer import BilateralCoverageAnalyzer
from tr_ru_forecast.data.quality.mirror_pairing import (
    calculate_mirror_gap,
    pair_canonical_and_mirror,
)
from tr_ru_forecast.data.quality.mirror_pipeline import MirrorQualityPipeline
from tr_ru_forecast.data.quality.mirror_transformer import (
    deduplicate_mirror_records,
    transform_mirror_raw_record,
)
from tr_ru_forecast.data.storage.raw_storage import RawSnapshotStore


def test_mirror_flow_constants() -> None:
    """Verify mirror flow mappings match frozen economic specification."""
    # TR_RU mirror: Russia reports imports from Turkey
    tr_ru_mirror = MIRROR_FLOWS[TradeDirection.TR_RU]
    assert tr_ru_mirror["direction"] == "TR_RU"
    assert tr_ru_mirror["reporter_code"] == RUSSIA_M49_CODE
    assert tr_ru_mirror["partner_code"] == TURKEY_M49_CODE
    assert tr_ru_mirror["flow"] == FLOW_IMPORT_NAME
    assert tr_ru_mirror["flow_code"] == FLOW_IMPORT_CODE

    # RU_TR mirror: Turkey reports imports from Russia
    ru_tr_mirror = MIRROR_FLOWS[TradeDirection.RU_TR]
    assert ru_tr_mirror["direction"] == "RU_TR"
    assert ru_tr_mirror["reporter_code"] == TURKEY_M49_CODE
    assert ru_tr_mirror["partner_code"] == RUSSIA_M49_CODE
    assert ru_tr_mirror["flow"] == FLOW_IMPORT_NAME
    assert ru_tr_mirror["flow_code"] == FLOW_IMPORT_CODE


def test_mirror_record_transformation_tr_ru() -> None:
    """Russia reporting imports from Turkey maps to physical direction TR_RU."""
    raw = {
        "period": "202301",
        "reporterCode": RUSSIA_M49_CODE,
        "partnerCode": TURKEY_M49_CODE,
        "cmdCode": "TOTAL",
        "primaryValue": 5000000.0,
        "classificationCode": "H6",
        "isOriginalClassification": True,
    }
    rec = transform_mirror_raw_record(raw, expected_direction="TR_RU")
    assert rec is not None
    assert rec["period"] == "2023-01"
    assert rec["direction"] == "TR_RU"
    assert rec["flow"] == "Import"
    assert rec["is_total"] is True
    assert rec["trade_value_usd"] == 5000000.0
    assert rec["data_status"] == "observed"
    assert rec["classification_code"] == "H6"
    assert rec["is_original_classification"] is True


def test_mirror_record_transformation_ru_tr() -> None:
    """Turkey reporting imports from Russia maps to physical direction RU_TR."""
    raw = {
        "period": "202301",
        "reporterCode": TURKEY_M49_CODE,
        "partnerCode": RUSSIA_M49_CODE,
        "cmdCode": "27",
        "primaryValue": 1234567.89,
        "classificationCode": "H6",
        "isOriginalClassification": True,
    }
    rec = transform_mirror_raw_record(raw, expected_direction="RU_TR")
    assert rec is not None
    assert rec["period"] == "2023-01"
    assert rec["direction"] == "RU_TR"
    assert rec["flow"] == "Import"
    assert rec["hs2"] == "27"
    assert rec["is_total"] is False
    assert rec["trade_value_usd"] == 1234567.89
    assert rec["data_status"] == "observed"


def test_mirror_missing_not_converted_to_zero() -> None:
    """Null mirror values must retain status 'missing' and must not become 0.0."""
    raw = {
        "period": "202301",
        "reporterCode": TURKEY_M49_CODE,
        "partnerCode": RUSSIA_M49_CODE,
        "cmdCode": "01",
        "primaryValue": None,
        "cifvalue": None,
    }
    rec = transform_mirror_raw_record(raw)
    assert rec is not None
    assert rec["trade_value_usd"] is None
    assert rec["data_status"] == "missing"


def test_mirror_reported_zero_preserved() -> None:
    """Explicit zero must retain status 'reported_zero' and value 0.0."""
    raw = {
        "period": "202301",
        "reporterCode": TURKEY_M49_CODE,
        "partnerCode": RUSSIA_M49_CODE,
        "cmdCode": "01",
        "primaryValue": 0.0,
    }
    rec = transform_mirror_raw_record(raw)
    assert rec is not None
    assert rec["trade_value_usd"] == 0.0
    assert rec["data_status"] == "reported_zero"


def test_mirror_exact_duplicate_collapse() -> None:
    """Exact duplicate mirror records collapse deterministically."""
    rec = {
        "period": "2023-01",
        "reporter_code": TURKEY_M49_CODE,
        "partner_code": RUSSIA_M49_CODE,
        "flow": "Import",
        "hs2": "TOTAL",
        "classification": "HS",
        "classification_code": "H6",
        "trade_value_usd": 1000.0,
    }
    records = [rec.copy(), rec.copy(), rec.copy()]
    res = deduplicate_mirror_records(records)
    assert len(res.clean_records) == 1
    assert res.exact_duplicate_count == 2
    assert res.conflict_count == 0


def test_mirror_conflicting_duplicate_quarantine() -> None:
    """Conflicting mirror values are quarantined from clean output."""
    rec1 = {
        "period": "2023-01",
        "reporter_code": TURKEY_M49_CODE,
        "partner_code": RUSSIA_M49_CODE,
        "flow": "Import",
        "hs2": "TOTAL",
        "classification": "HS",
        "classification_code": "H6",
        "trade_value_usd": 1000.0,
    }
    rec2 = rec1.copy()
    rec2["trade_value_usd"] = 2000.0

    res = deduplicate_mirror_records([rec1, rec2])
    assert len(res.clean_records) == 0
    assert res.conflict_count == 1
    assert len(res.quarantined_records) == 2


def test_mirror_conflicting_duplicate_fail_closed() -> None:
    """fail_closed=True raises ConflictingDuplicateError on conflict."""
    rec1 = {
        "period": "2023-01",
        "reporter_code": TURKEY_M49_CODE,
        "partner_code": RUSSIA_M49_CODE,
        "flow": "Import",
        "hs2": "TOTAL",
        "classification": "HS",
        "classification_code": "H6",
        "trade_value_usd": 1000.0,
    }
    rec2 = rec1.copy()
    rec2["trade_value_usd"] = 2000.0

    with pytest.raises(ConflictingDuplicateError):
        deduplicate_mirror_records([rec1, rec2], fail_closed=True)


def test_mirror_pairing_both_present() -> None:
    """Both canonical and mirror present computes correct gaps."""
    c_df = pd.DataFrame(
        [
            {
                "period": "2023-01",
                "direction": "TR_RU",
                "hs2": "TOTAL",
                "trade_value_usd": 100.0,
                "classification": "HS",
                "classification_code": "H6",
                "is_original_classification": True,
                "retrieved_at": "2026-01-01T00:00:00Z",
            }
        ]
    )
    m_df = pd.DataFrame(
        [
            {
                "period": "2023-01",
                "direction": "TR_RU",
                "hs2": "TOTAL",
                "trade_value_usd": 110.0,
                "classification": "HS",
                "classification_code": "H6",
                "is_original_classification": True,
                "retrieved_at": "2026-01-01T00:00:00Z",
            }
        ]
    )

    paired = pair_canonical_and_mirror(c_df, m_df)
    assert list(paired.columns) == MIRROR_MONTHLY_COLUMNS
    assert len(paired) == 1
    row = paired.iloc[0]
    assert row["pair_status"] == PairStatus.BOTH_PRESENT.value
    assert row["canonical_value_usd"] == 100.0
    assert row["mirror_value_usd"] == 110.0
    assert row["mirror_gap_usd"] == -10.0
    assert row["mirror_gap_pct"] == pytest.approx(-10.0)
    assert row["abs_mirror_gap_pct"] == pytest.approx(10.0)
    assert not row["canonical_missing"]
    assert not row["mirror_missing"]
    assert not row["classification_mismatch"]
    assert row["gap_flag"] == GapFlag.PAIRED.value


def test_mirror_pairing_canonical_only() -> None:
    """Canonical present, mirror absent gives CANONICAL_ONLY and null gaps."""
    c_df = pd.DataFrame(
        [
            {
                "period": "2023-01",
                "direction": "TR_RU",
                "hs2": "TOTAL",
                "trade_value_usd": 100.0,
                "classification": "HS",
                "classification_code": "H6",
                "is_original_classification": True,
                "retrieved_at": "2026-01-01T00:00:00Z",
            }
        ]
    )
    m_df = pd.DataFrame()

    paired = pair_canonical_and_mirror(c_df, m_df)
    assert len(paired) == 1
    row = paired.iloc[0]
    assert row["pair_status"] == PairStatus.CANONICAL_ONLY.value
    assert row["canonical_value_usd"] == 100.0
    assert row["mirror_value_usd"] is None
    assert row["mirror_gap_usd"] is None
    assert row["mirror_gap_pct"] is None
    assert not row["canonical_missing"]
    assert row["mirror_missing"]
    assert row["gap_flag"] == GapFlag.UNPAIRED.value


def test_mirror_pairing_mirror_only() -> None:
    """Mirror present, canonical absent gives MIRROR_ONLY and null gaps (e.g. 2022+ RU_TR)."""
    c_df = pd.DataFrame()
    m_df = pd.DataFrame(
        [
            {
                "period": "2023-01",
                "direction": "RU_TR",
                "hs2": "TOTAL",
                "trade_value_usd": 500.0,
                "classification": "HS",
                "classification_code": "H6",
                "is_original_classification": True,
                "retrieved_at": "2026-01-01T00:00:00Z",
            }
        ]
    )

    paired = pair_canonical_and_mirror(c_df, m_df)
    assert len(paired) == 1
    row = paired.iloc[0]
    assert row["pair_status"] == PairStatus.MIRROR_ONLY.value
    assert row["canonical_value_usd"] is None
    assert row["mirror_value_usd"] == 500.0
    assert row["mirror_gap_usd"] is None
    assert row["mirror_gap_pct"] is None
    assert row["canonical_missing"]
    assert not row["mirror_missing"]
    assert row["gap_flag"] == GapFlag.UNPAIRED.value


def test_mirror_pairing_zero_division_safety() -> None:
    """When canonical value is 0.0, mirror_gap_pct is None (no ZeroDivisionError or Inf)."""
    c_df = pd.DataFrame(
        [
            {
                "period": "2023-01",
                "direction": "TR_RU",
                "hs2": "01",
                "trade_value_usd": 0.0,
                "classification": "HS",
                "classification_code": "H6",
                "is_original_classification": True,
                "retrieved_at": "2026-01-01T00:00:00Z",
            }
        ]
    )
    m_df = pd.DataFrame(
        [
            {
                "period": "2023-01",
                "direction": "TR_RU",
                "hs2": "01",
                "trade_value_usd": 50.0,
                "classification": "HS",
                "classification_code": "H6",
                "is_original_classification": True,
                "retrieved_at": "2026-01-01T00:00:00Z",
            }
        ]
    )

    paired = pair_canonical_and_mirror(c_df, m_df)
    row = paired.iloc[0]
    assert row["canonical_value_usd"] == 0.0
    assert row["mirror_value_usd"] == 50.0
    assert row["mirror_gap_usd"] == -50.0
    assert row["mirror_gap_pct"] is None
    assert row["abs_mirror_gap_pct"] is None
    assert row["gap_flag"] == GapFlag.PAIRED.value


def test_mirror_canonical_immutability() -> None:
    """Assert canonical DataFrame and values are NEVER mutated or overwritten by mirror data."""
    c_df = pd.DataFrame(
        [
            {
                "period": "2023-01",
                "direction": "TR_RU",
                "hs2": "TOTAL",
                "trade_value_usd": 12345.0,
                "classification": "HS",
                "classification_code": "H6",
                "is_original_classification": True,
                "retrieved_at": "2026-01-01T00:00:00Z",
            }
        ]
    )
    original_c_val = c_df["trade_value_usd"].iloc[0]

    m_df = pd.DataFrame(
        [
            {
                "period": "2023-01",
                "direction": "TR_RU",
                "hs2": "TOTAL",
                "trade_value_usd": 999999.0,
                "classification": "HS",
                "classification_code": "H6",
                "is_original_classification": True,
                "retrieved_at": "2026-01-01T00:00:00Z",
            }
        ]
    )

    paired = pair_canonical_and_mirror(c_df, m_df)

    # 1. Canonical source DataFrame unchanged
    assert c_df["trade_value_usd"].iloc[0] == original_c_val
    # 2. Paired table preserves original canonical value
    assert paired["canonical_value_usd"].iloc[0] == original_c_val
    assert paired["canonical_value_usd"].iloc[0] != m_df["trade_value_usd"].iloc[0]


def test_mirror_classification_mismatch() -> None:
    """Mismatch in classification codes (e.g. H4 vs H5) is flagged accurately."""
    c_df = pd.DataFrame(
        [
            {
                "period": "2023-01",
                "direction": "TR_RU",
                "hs2": "TOTAL",
                "trade_value_usd": 100.0,
                "classification": "HS",
                "classification_code": "H4",
                "is_original_classification": False,
                "retrieved_at": "2026-01-01T00:00:00Z",
            }
        ]
    )
    m_df = pd.DataFrame(
        [
            {
                "period": "2023-01",
                "direction": "TR_RU",
                "hs2": "TOTAL",
                "trade_value_usd": 100.0,
                "classification": "HS",
                "classification_code": "H5",
                "is_original_classification": True,
                "retrieved_at": "2026-01-01T00:00:00Z",
            }
        ]
    )

    paired = pair_canonical_and_mirror(c_df, m_df)
    row = paired.iloc[0]
    assert row["classification_mismatch"]
    assert row["canonical_classification_code"] == "H4"
    assert row["mirror_classification_code"] == "H5"


def test_mirror_gap_flag_thresholds() -> None:
    """Explicit thresholds map to NONE, MODERATE, LARGE; default policy yields PAIRED/UNPAIRED."""
    th = (15.0, 50.0)
    # With explicit thresholds:
    # <= 15% -> NONE
    _, _, abs1, flag_low = calculate_mirror_gap(100.0, 110.0, thresholds=th)  # -10%
    assert abs1 == pytest.approx(10.0)
    assert flag_low == GapFlag.NONE

    # 15% - 50% -> MODERATE
    _, _, abs2, flag_mod = calculate_mirror_gap(100.0, 130.0, thresholds=th)  # -30%
    assert abs2 == pytest.approx(30.0)
    assert flag_mod == GapFlag.MODERATE

    # > 50% -> LARGE
    _, _, abs3, flag_large = calculate_mirror_gap(100.0, 180.0, thresholds=th)  # -80%
    assert abs3 == pytest.approx(80.0)
    assert flag_large == GapFlag.LARGE

    # Without explicit thresholds (Revision Gate default policy):
    _, _, abs_def, flag_def = calculate_mirror_gap(100.0, 180.0)
    assert abs_def == pytest.approx(80.0)
    assert flag_def == GapFlag.PAIRED


def test_coverage_analyzer_full_report() -> None:
    """BilateralCoverageAnalyzer produces valid metrics and respects verification scope."""
    data = [
        # Pre-2022 RU_TR: Both present
        {
            "period": "2021-01",
            "direction": "RU_TR",
            "hs2": "TOTAL",
            "canonical_value_usd": 1000.0,
            "mirror_value_usd": 1100.0,
            "mirror_gap_usd": -100.0,
            "mirror_gap_pct": -10.0,
            "abs_mirror_gap_pct": 10.0,
            "canonical_missing": False,
            "mirror_missing": False,
            "pair_status": PairStatus.BOTH_PRESENT.value,
            "canonical_classification": "HS",
            "canonical_classification_code": "H5",
            "canonical_is_original_classification": True,
            "mirror_classification": "HS",
            "mirror_classification_code": "H5",
            "mirror_is_original_classification": True,
            "classification_mismatch": False,
            "canonical_retrieved_at": "2026-01-01",
            "mirror_retrieved_at": "2026-01-01",
            "gap_flag": GapFlag.PAIRED.value,
        },
        # Post-2022 RU_TR: Mirror only (Russia ceased reporting)
        {
            "period": "2023-01",
            "direction": "RU_TR",
            "hs2": "TOTAL",
            "canonical_value_usd": None,
            "mirror_value_usd": 1500.0,
            "mirror_gap_usd": None,
            "mirror_gap_pct": None,
            "abs_mirror_gap_pct": None,
            "canonical_missing": True,
            "mirror_missing": False,
            "pair_status": PairStatus.MIRROR_ONLY.value,
            "canonical_classification": None,
            "canonical_classification_code": None,
            "canonical_is_original_classification": None,
            "mirror_classification": "HS",
            "mirror_classification_code": "H6",
            "mirror_is_original_classification": True,
            "classification_mismatch": False,
            "canonical_retrieved_at": None,
            "mirror_retrieved_at": "2026-01-01",
            "gap_flag": GapFlag.UNPAIRED.value,
        },
    ]
    df = pd.DataFrame(data)

    # 1. Partial coverage (default without authenticated key)
    analyzer_partial = BilateralCoverageAnalyzer(
        df, full_history_verified=False, expected_period_count=120
    )
    report_partial = analyzer_partial.generate_full_report()

    assert report_partial["total_records"] == 2
    assert report_partial["verification_scope"] == "PARTIAL"
    assert not report_partial["full_history_verified"]
    assert report_partial["expected_period_count"] == 120
    assert report_partial["observed_period_count"] == 2
    assert "RU_TR" in report_partial["directions"]

    ru_tr_audit_partial = report_partial["reporting_suspension_empirical_audit"]["RU_TR"]
    assert ru_tr_audit_partial["reporting_suspension_status"] == "UNVERIFIED"
    assert not ru_tr_audit_partial["empirical_reporting_suspension_detected"]
    assert "PRELIMINARY / PARTIAL COVERAGE" in ru_tr_audit_partial["analytical_finding"]

    # 2. Full coverage audit (authenticated key simulated)
    analyzer_full = BilateralCoverageAnalyzer(
        df, full_history_verified=True, expected_period_count=2
    )
    report_full = analyzer_full.generate_full_report()
    assert report_full["verification_scope"] == "FULL"
    assert report_full["full_history_verified"]

    ru_tr_audit_full = report_full["reporting_suspension_empirical_audit"]["RU_TR"]
    assert ru_tr_audit_full["reporting_suspension_status"] == "CONFIRMED"
    assert ru_tr_audit_full["empirical_reporting_suspension_detected"] is True
    assert "CONFIRMED FULL HISTORY AUDIT" in ru_tr_audit_full["analytical_finding"]


def test_coverage_summary_metadata_fields() -> None:
    """Verify all 12 metadata fields required by M2 Revision Gate exist in direction summary."""
    data = [
        {
            "period": "2021-01",
            "direction": "TR_RU",
            "hs2": "TOTAL",
            "canonical_value_usd": 100.0,
            "mirror_value_usd": 105.0,
            "mirror_gap_usd": -5.0,
            "mirror_gap_pct": -5.0,
            "abs_mirror_gap_pct": 5.0,
            "canonical_missing": False,
            "mirror_missing": False,
            "pair_status": PairStatus.BOTH_PRESENT.value,
            "canonical_classification": "HS",
            "canonical_classification_code": "H5",
            "canonical_is_original_classification": True,
            "mirror_classification": "HS",
            "mirror_classification_code": "H5",
            "mirror_is_original_classification": True,
            "classification_mismatch": False,
            "canonical_retrieved_at": "2026-01-01",
            "mirror_retrieved_at": "2026-01-01",
            "gap_flag": GapFlag.PAIRED.value,
        }
    ]
    df = pd.DataFrame(data)
    analyzer = BilateralCoverageAnalyzer(df, full_history_verified=False, expected_period_count=10)
    report = analyzer.generate_full_report()
    tr_ru = report["directions"]["TR_RU"]

    # All 12 required fields
    expected_fields = [
        "verification_scope",
        "full_history_verified",
        "expected_period_count",
        "observed_period_count",
        "period_min",
        "period_max",
        "canonical_observed_count",
        "mirror_observed_count",
        "both_present_count",
        "canonical_only_count",
        "mirror_only_count",
        "both_missing_count",
    ]
    for field in expected_fields:
        assert field in tr_ru, f"Missing required metadata field: {field}"

    assert tr_ru["verification_scope"] == "PARTIAL"
    assert not tr_ru["full_history_verified"]
    assert tr_ru["expected_period_count"] == 10
    assert tr_ru["observed_period_count"] == 1
    assert tr_ru["period_min"] == "2021-01"
    assert tr_ru["period_max"] == "2021-01"
    assert tr_ru["canonical_observed_count"] == 1
    assert tr_ru["mirror_observed_count"] == 1
    assert tr_ru["both_present_count"] == 1
    assert tr_ru["canonical_only_count"] == 0
    assert tr_ru["mirror_only_count"] == 0
    assert tr_ru["both_missing_count"] == 0


def test_mirror_pipeline_integration_mocked(tmp_path) -> None:
    """End-to-end integration test of MirrorQualityPipeline with mocked responses."""

    # Mock Comtrade API responses based on flowCode
    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        # Check if flowCode is M (mirror) or X (export)
        if "flowCode=M" in url_str:
            # Mirror import response
            if "reporterCode=643" in url_str:
                # RU imports from TR (mirror for TR_RU)
                data = [
                    {
                        "period": "202301",
                        "reporterCode": 643,
                        "reporterDesc": "Russian Federation",
                        "partnerCode": 792,
                        "partnerDesc": "Türkiye",
                        "flowCode": "M",
                        "cmdCode": "TOTAL",
                        "primaryValue": 105000.0,
                        "classificationCode": "H6",
                        "classificationSearchCode": "HS",
                        "isOriginalClassification": True,
                    }
                ]
            else:
                # TR imports from RU (mirror for RU_TR)
                data = [
                    {
                        "period": "202301",
                        "reporterCode": 792,
                        "reporterDesc": "Türkiye",
                        "partnerCode": 643,
                        "partnerDesc": "Russian Federation",
                        "flowCode": "M",
                        "cmdCode": "TOTAL",
                        "primaryValue": 210000.0,
                        "classificationCode": "H6",
                        "classificationSearchCode": "HS",
                        "isOriginalClassification": True,
                    }
                ]
        else:
            # Canonical export response
            if "reporterCode=792" in url_str:
                # TR exports to RU (canonical TR_RU)
                data = [
                    {
                        "period": "202301",
                        "reporterCode": 792,
                        "reporterDesc": "Türkiye",
                        "partnerCode": 643,
                        "partnerDesc": "Russian Federation",
                        "flowCode": "X",
                        "cmdCode": "TOTAL",
                        "primaryValue": 100000.0,
                        "classificationCode": "H6",
                        "classificationSearchCode": "HS",
                        "isOriginalClassification": True,
                    }
                ]
            else:
                # RU exports to TR (canonical RU_TR - e.g. empty or missing for 202301)
                data = []

        return httpx.Response(200, json={"elapsedTime": 0.01, "count": len(data), "data": data})

    mock_transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock_transport)

    test_settings = Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        interim_data_dir=tmp_path / "interim",
        comtrade_api_key=None,
    )

    client = ComtradeClient(settings=test_settings, http_client=http_client)
    snapshot_store = RawSnapshotStore(settings=test_settings)

    pipeline = MirrorQualityPipeline(
        settings=test_settings,
        client=client,
        snapshot_store=snapshot_store,
    )

    paired_df, summary = pipeline.run_all(
        start_period="2023-01",
        end_period="2023-01",
        use_cache=False,
    )

    assert not paired_df.empty
    assert len(paired_df) == 2  # One TR_RU, one RU_TR

    # Verify persisted files
    assert (test_settings.processed_data_dir / "mirror_monthly.parquet").exists()
    assert (test_settings.interim_data_dir / "mirror_monthly.csv").exists()
    assert (test_settings.processed_data_dir / "mirror_qa_summary.json").exists()

    # TR_RU row check: both present
    tr_ru_row = paired_df[paired_df["direction"] == "TR_RU"].iloc[0]
    assert tr_ru_row["pair_status"] == PairStatus.BOTH_PRESENT.value
    assert tr_ru_row["canonical_value_usd"] == 100000.0
    assert tr_ru_row["mirror_value_usd"] == 105000.0
    assert tr_ru_row["mirror_gap_usd"] == -5000.0
    assert tr_ru_row["mirror_gap_pct"] == pytest.approx(-5.0)
    assert tr_ru_row["abs_mirror_gap_pct"] == pytest.approx(5.0)
    assert tr_ru_row["gap_flag"] == GapFlag.PAIRED.value

    # RU_TR row check: canonical missing, mirror present -> MIRROR_ONLY
    ru_tr_row = paired_df[paired_df["direction"] == "RU_TR"].iloc[0]
    assert ru_tr_row["pair_status"] == PairStatus.MIRROR_ONLY.value
    assert ru_tr_row["canonical_value_usd"] is None
    assert ru_tr_row["mirror_value_usd"] == 210000.0
    assert ru_tr_row["mirror_gap_usd"] is None
    assert ru_tr_row["mirror_gap_pct"] is None
    assert ru_tr_row["abs_mirror_gap_pct"] is None

    # Summary check
    assert summary["pipeline"] == "M2_Mirror_Statistics"
    assert summary["verification_scope"] == "PARTIAL"
    assert not summary["full_history_verified"]
    assert "TR_RU" in summary["quality_audit"]["directions"]
    assert "RU_TR" in summary["quality_audit"]["directions"]


def test_mirror_pipeline_quarantine_conflicts(tmp_path) -> None:
    """Conflicting mirror duplicates trigger quarantine file creation during pipeline run."""
    mock_payload = {
        "elapsedTime": 0.01,
        "count": 2,
        "data": [
            {
                "period": "202301",
                "reporterCode": 792,
                "partnerCode": 643,
                "flowCode": "M",
                "cmdCode": "TOTAL",
                "primaryValue": 1000.0,
                "classificationCode": "H6",
            },
            {
                "period": "202301",
                "reporterCode": 792,
                "partnerCode": 643,
                "flowCode": "M",
                "cmdCode": "TOTAL",
                "primaryValue": 9999.0,
                "classificationCode": "H6",
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=mock_payload)

    mock_transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock_transport)

    test_settings = Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        interim_data_dir=tmp_path / "interim",
    )

    client = ComtradeClient(settings=test_settings, http_client=http_client)
    snapshot_store = RawSnapshotStore(settings=test_settings)

    pipeline = MirrorQualityPipeline(
        settings=test_settings,
        client=client,
        snapshot_store=snapshot_store,
    )

    df, qa = pipeline.ingest_mirror_for_direction(
        direction=TradeDirection.RU_TR,
        start_period="2023-01",
        end_period="2023-01",
    )

    assert len(df) == 0  # Conflicted records quarantined
    assert qa["conflict_count"] == 1
    assert qa["quarantined_count"] == 2
    assert (test_settings.interim_data_dir / "quarantine_mirror_RU_TR.csv").exists()
