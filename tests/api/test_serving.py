"""Frozen artifact API contract tests, including failure policies and real artifact parity."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tr_ru_forecast.serving.config import ServingSettings


@pytest.mark.parametrize("route", ["/health", "/api/v1/health"])
def test_health_reports_verified_provenance(artifact_client: TestClient, route: str) -> None:
    """Readiness includes exact dimensions, evaluation scope and both manifest identities."""
    response = artifact_client.get(route)

    payload = response.json()
    assert response.status_code == 200
    assert payload["checksum_status"] == "verified"
    assert payload["total_tasks"] == 42
    assert payload["total_holdout_predictions"] == 1512
    assert payload["total_shap_rows"] == 9072
    assert payload["verified_file_count"] == 20
    assert payload["specification_version"] == "1.1"
    assert payload["evaluation_scope"] == "retrospective_latest_vintage_deterministic_lag"
    assert all(payload["readiness"].values())
    assert len(payload["m10"]["manifest_sha256"]) == len(payload["m11"]["source_fingerprint"]) == 64


def test_sectors_include_all_frozen_assignments(artifact_client: TestClient) -> None:
    """Both TOTAL series and twelve HS2 series expose all 42 assignments."""
    response = artifact_client.get("/api/v1/sectors")

    sectors = response.json()["sectors"]
    assert len(sectors) == 14
    assert sum(len(sector["tasks"]) for sector in sectors) == 42
    assert sum(sector["is_total"] for sector in sectors) == 2
    assert {sector["sector_id"] for sector in sectors} == {
        "RU_TR:10", "RU_TR:23", "RU_TR:27", "RU_TR:72", "RU_TR:74", "RU_TR:76", "RU_TR:TOTAL",
        "TR_RU:08", "TR_RU:39", "TR_RU:73", "TR_RU:84", "TR_RU:85", "TR_RU:87", "TR_RU:TOTAL",
    }
    assert all(sector["reporter_code"] == 792 for sector in sectors)
    assert all(sector["history_start"] == "2012-01" and sector["history_end"] == "2025-12" for sector in sectors)


def test_leaderboard_kpis_derive_from_frozen_tasks(artifact_client: TestClient) -> None:
    """Global metric values match documented holdout results and CV selection."""
    payload = artifact_client.get("/api/v1/leaderboard").json()

    summary = payload["summary"]
    assert len(payload["tasks"]) == 42
    assert sum(len(task["cv_candidates"]) for task in payload["tasks"]) == 168
    assert {task["selection_reason"] for task in payload["tasks"]} <= {
        "CLEAR_SMAPE_WINNER", "PARSIMONY_PREFERENCE"
    }
    assert summary["operational_mean_smape"] == pytest.approx(26.17040963)
    assert summary["baseline_mean_smape"] == pytest.approx(37.49645485)
    assert summary["cv_mean_skill_pct"] == pytest.approx(30.51, abs=0.01)
    assert (summary["sarima_champions"], summary["xgboost_champions"]) == (33, 9)
    assert summary["fallback_replacements"] == 1


@pytest.mark.parametrize("query,expected", [("direction=TR_RU", 21), ("horizon=6", 14),
                                          ("hs2=10", 3), ("champion_model=xgboost", 9),
                                          ("direction=RU_TR&hs2=10&horizon=3&champion_model=sarima", 1)])
def test_leaderboard_filters_preserve_global_kpis(artifact_client: TestClient, query: str, expected: int) -> None:
    """Filters change rows while keeping overall KPIs stable."""
    response = artifact_client.get("/api/v1/leaderboard?" + query)

    assert response.status_code == 200
    assert len(response.json()["tasks"]) == expected
    assert response.json()["summary"]["total_tasks"] == 42


@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_forecasts_match_every_persisted_task(
    artifact_client: TestClient, serving_settings: ServingSettings, horizon: int
) -> None:
    """All primary/raw/operational values and interval bounds match frozen artifact rows."""
    frame = pd.read_parquet(serving_settings.artifact_root / "m11_holdout_predictions.parquet")
    selected = frame.loc[frame.horizon.eq(horizon) & frame.role.eq("champion")]

    for sector_id, group in selected.groupby("sector_id"):
        response = artifact_client.get(f"/api/v1/forecasts/{sector_id}?horizon={horizon}")
        assert response.status_code == 200
        task = response.json()["tasks"][0]
        assert len(task["points"]) == 12
        assert len(response.json()["history"]) == 168
        for point, record in zip(task["points"], group.sort_values("forecast_period").itertuples(), strict=True):
            assert point["target_date"] == record.forecast_period
            assert point["champion_forecast"] == record.prediction
            assert point["operational_forecast"] == record.operational_prediction
            assert point["lower_80"] == record.operational_lower_80
            assert point["upper_95"] == record.operational_upper_95
            assert point["actual"] == record.actual


def test_negative_champion_uses_registered_fallback_with_its_intervals(artifact_client: TestClient) -> None:
    """July h3 retains a negative SARIMA raw value and serves XGBoost governed fallback."""
    response = artifact_client.get("/api/v1/forecasts/RU_TR__10?horizon=3")

    point = next(item for item in response.json()["tasks"][0]["points"] if item["target_date"] == "2025-07")
    assert point["champion_forecast"] < 0
    assert point["used_fallback"] is True
    assert point["operational_forecast"] == point["fallback_forecast"] == 76569744.0
    assert point["operational_model"] == "xgboost"
    assert point["interval_policy"] == "prequential_conformal"
    assert point["calibration_pool_size"] >= 19


def test_invalid_fallback_does_not_replace_valid_champion(artifact_client: TestClient) -> None:
    """A negative h6 fallback cannot displace its valid XGBoost champion."""
    response = artifact_client.get("/api/v1/forecasts/RU_TR:10?horizon=6")

    point = next(item for item in response.json()["tasks"][0]["points"] if item["target_date"] == "2025-07")
    assert point["fallback_forecast"] < 0
    assert point["fallback_is_valid"] is False
    assert point["used_fallback"] is False
    assert point["operational_forecast"] == point["champion_forecast"]


def test_forecast_aliases_are_equivalent(artifact_client: TestClient) -> None:
    """Colon, URL-safe path and query selections resolve to the same task."""
    colon = artifact_client.get("/api/v1/forecasts/RU_TR:10?horizon=1").json()

    assert artifact_client.get("/api/v1/forecasts/RU_TR__10?horizon=1").json() == colon
    assert artifact_client.get("/api/v1/forecasts?sector_id=RU_TR__10&horizon=1").json() == colon


def test_total_overview_uses_exact_decimal_totals_only(artifact_client: TestClient) -> None:
    """Overview does not sum sector components into TOTAL again."""
    payload = artifact_client.get("/api/v1/overview").json()

    exports = Decimal(payload["tr_ru_total_usd"])
    imports = Decimal(payload["ru_tr_total_usd"])
    assert Decimal(payload["bilateral_total_usd"]) == exports + imports
    assert Decimal(payload["turkey_trade_balance_usd"]) == exports - imports


def test_history_pagination_and_source_precision(artifact_client: TestClient) -> None:
    """Date bounds, pagination and reporter identity remain explicit."""
    response = artifact_client.get("/api/v1/trade?sector_id=TR_RU:08&start_month=2025-01&end_month=2025-12&offset=2&limit=3")

    payload = response.json()
    assert response.status_code == 200
    assert payload["total_observations"] == 12
    assert [point["period"] for point in payload["observations"]] == ["2025-03", "2025-04", "2025-05"]
    assert all(isinstance(point["trade_value_usd"], str) and point["reporter_code"] == 792 for point in payload["observations"])


def test_macro_and_methodology_keep_approximate_vintage_limits(artifact_client: TestClient) -> None:
    """Indicators expose lag metadata; methodology discloses retrospective and coverage limits."""
    indicators = artifact_client.get("/api/v1/indicators?indicator_id=brent&start_month=2025-01&end_month=2025-12").json()
    methodology = artifact_client.get("/api/v1/methodology").json()

    assert len(indicators["observations"]) == 12
    assert all(not point["availability_is_exact"] for point in indicators["observations"])
    assert any("67.59" in limitation for limitation in methodology["limitations"])
    assert methodology["forecast_kind"] == "historical_holdout"
