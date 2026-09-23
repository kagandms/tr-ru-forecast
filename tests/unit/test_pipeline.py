"""Unit tests for the trade ingestion pipeline coordinator (M1)."""

import httpx
import pytest

from tr_ru_forecast.config import Settings
from tr_ru_forecast.core.constants import TradeDirection
from tr_ru_forecast.data.ingestion.comtrade_client import ComtradeClient
from tr_ru_forecast.data.ingestion.pipeline import (
    TradeIngestionPipeline,
    generate_period_range,
    resolve_latest_complete_month,
)
from tr_ru_forecast.data.ingestion.transformer import STANDARDIZED_COLUMNS
from tr_ru_forecast.data.storage.raw_storage import RawSnapshotStore


def test_generate_period_range() -> None:
    """Verify period list generation between start and end dates."""
    periods = generate_period_range("2023-01", "2023-03")
    assert periods == ["202301", "202302", "202303"]

    periods_single = generate_period_range("202301", "202301")
    assert periods_single == ["202301"]

    with pytest.raises(ValueError):
        generate_period_range("2023-05", "2023-01")


def test_resolve_latest_complete_month() -> None:
    """Verify latest complete month calculation returns YYYY-MM format."""
    month = resolve_latest_complete_month()
    assert len(month) == 7
    assert month[4] == "-"
    year_part, month_part = month.split("-")
    assert year_part.isdigit() and len(year_part) == 4
    assert 1 <= int(month_part) <= 12


def test_pipeline_ingestion_mocked(tmp_path) -> None:
    """Verify complete pipeline ingestion flow with mocked responses."""
    mock_payload = {
        "elapsedTime": 0.02,
        "count": 3,
        "data": [
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
            },
            {
                "period": "202301",
                "reporterCode": 792,
                "reporterDesc": "Türkiye",
                "partnerCode": 643,
                "partnerDesc": "Russian Federation",
                "flowCode": "X",
                "cmdCode": "01",
                "primaryValue": 40000.0,
                "classificationCode": "H6",
                "classificationSearchCode": "HS",
                "isOriginalClassification": True,
            },
            {
                "period": "202301",
                "reporterCode": 792,
                "reporterDesc": "Türkiye",
                "partnerCode": 643,
                "partnerDesc": "Russian Federation",
                "flowCode": "X",
                "cmdCode": "02",
                "primaryValue": 60000.0,
                "classificationCode": "H6",
                "classificationSearchCode": "HS",
                "isOriginalClassification": True,
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
    store = RawSnapshotStore(settings=test_settings)
    pipeline = TradeIngestionPipeline(
        settings=test_settings,
        client=client,
        snapshot_store=store,
    )

    df, qa = pipeline.run_ingestion_for_direction(
        direction=TradeDirection.TR_RU,
        start_period="2023-01",
        end_period="2023-01",
    )

    assert not df.empty
    assert len(df) == 3
    assert list(df.columns) == STANDARDIZED_COLUMNS
    assert (df["direction"] == "TR_RU").all()

    # Verify classification metadata
    assert (df["classification"] == "HS").all()
    assert (df["classification_code"] == "H6").all()
    assert (df["is_original_classification"] == True).all()  # noqa: E712

    # Verify TOTAL vs HS2
    assert (df["is_total"] == True).sum() == 1  # noqa: E712
    assert (df["is_total"] == False).sum() == 2  # noqa: E712

    # Verify snapshot stored
    saved_snapshot = store.load_snapshot("TR_RU", "202301")
    assert saved_snapshot is not None
    assert saved_snapshot["metadata"]["record_count"] == 3

    # Verify QA Summary
    assert qa["direction"] == "TR_RU"
    assert qa["raw_record_count"] == 3
    assert qa["standardized_record_count"] == 3
    assert qa["total_record_count"] == 1
    assert qa["hs2_record_count"] == 2
    assert qa["duplicate_count"] == 0
    assert qa["conflict_count"] == 0
    assert qa["conflict_keys"] == []
    assert qa["classification_values"] == ["HS"]
    assert qa["classification_codes"] == ["H6"]
    assert qa["period_min"] == "2023-01"
    assert qa["period_max"] == "2023-01"

    # Verify files on disk
    parquet_path = test_settings.processed_data_dir / "trade_monthly_TR_RU.parquet"
    csv_path = test_settings.interim_data_dir / "trade_monthly_TR_RU.csv"
    assert parquet_path.exists()
    assert csv_path.exists()


def test_pipeline_quarantine_on_conflicting_duplicates(tmp_path) -> None:
    """Verify pipeline quarantines conflicting records and writes quarantine CSV."""
    conflict_payload = {
        "count": 3,
        "data": [
            {
                "period": "202301",
                "reporterCode": 792,
                "partnerCode": 643,
                "flowCode": "X",
                "cmdCode": "TOTAL",
                "primaryValue": 500000.0,
                "classificationCode": "H6",
            },
            {
                "period": "202301",
                "reporterCode": 792,
                "partnerCode": 643,
                "flowCode": "X",
                "cmdCode": "01",
                "primaryValue": 10000.0,
                "classificationCode": "H6",
            },
            {
                "period": "202301",
                "reporterCode": 792,
                "partnerCode": 643,
                "flowCode": "X",
                "cmdCode": "01",
                "primaryValue": 25000.0,  # Conflicting value for 01!
                "classificationCode": "H6",
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=conflict_payload)

    mock_transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock_transport)

    test_settings = Settings(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        interim_data_dir=tmp_path / "interim",
    )

    pipeline = TradeIngestionPipeline(
        settings=test_settings,
        client=ComtradeClient(settings=test_settings, http_client=http_client),
        snapshot_store=RawSnapshotStore(settings=test_settings),
    )

    df, qa = pipeline.run_ingestion_for_direction(
        direction=TradeDirection.TR_RU,
        start_period="2023-01",
        end_period="2023-01",
    )

    # Clean records should ONLY contain TOTAL (the conflicting 01 must be quarantined)
    assert len(df) == 1
    assert df["hs2"].iloc[0] == "TOTAL"

    # QA metrics
    assert qa["conflict_count"] == 1
    assert len(qa["conflict_keys"]) == 1
    assert qa["quarantined_count"] == 2

    # Quarantine file must exist on disk
    quarantine_path = test_settings.interim_data_dir / "quarantine_TR_RU.csv"
    assert quarantine_path.exists()
