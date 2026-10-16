"""SHAP and mirror API tests retain role, provenance and missing-data boundaries."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tr_ru_forecast.serving.config import ServingSettings


def test_all_9072_shap_values_are_served_in_registered_roles(
    artifact_client: TestClient, serving_settings: ServingSettings
) -> None:
    """Every attributed task has 12×21 values matching frozen signed/global SHAP values."""
    local = pd.read_parquet(serving_settings.artifact_root / "m11_shap_values.parquet")
    observed_rows = 0

    for (sector_id, horizon), group in local.groupby(["sector_id", "horizon"]):
        response = artifact_client.get(f"/api/v1/explainability/{sector_id}?horizon={horizon}")
        payload = response.json()
        assert response.status_code == 200
        assert payload["status"] == "available"
        assert payload["role"] in {"champion", "fallback"}
        assert len(payload["global_importance"]) == 21
        assert len(payload["local"]) == 12
        for feature in payload["global_importance"]:
            reference = group.loc[group.feature.eq(feature["feature"]), "shap_value"]
            assert feature["mean_abs_shap"] == pytest.approx(reference.abs().mean())
            assert feature["mean_signed_shap"] == pytest.approx(reference.mean())
        for explanation in payload["local"]:
            reference = group.loc[group.forecast_period.eq(explanation["target_month"])].set_index("feature")
            observed_rows += len(explanation["contributions"])
            assert explanation["base_value"] == reference.iloc[0].base_value
            for attribution in explanation["contributions"]:
                assert attribution["shap_value"] == reference.loc[attribution["feature"], "shap_value"]
    assert observed_rows == 9072


def test_local_month_filter_keeps_global_importance(artifact_client: TestClient) -> None:
    """Global attribution uses all 12 origins even when one waterfall is requested."""
    complete = artifact_client.get("/api/v1/explainability/RU_TR:10?horizon=3").json()
    selected = artifact_client.get("/api/v1/explainability/RU_TR__10?horizon=3&target_month=2025-07").json()

    assert selected["role"] == "fallback"
    assert selected["global_importance"] == complete["global_importance"]
    assert len(selected["local"]) == 1
    assert selected["local"][0]["target_month"] == "2025-07"
    assert all(item["shap_value"] > 0 for item in selected["local"][0]["top_positive"])
    assert all(item["shap_value"] < 0 for item in selected["local"][0]["top_negative"])


def test_xgboost_champion_explanation_is_labelled_champion(artifact_client: TestClient) -> None:
    """Model Lab distinguishes XGBoost champion explanations from fallback explanations."""
    payload = artifact_client.get("/api/v1/explainability/RU_TR:10?horizon=6").json()

    assert payload["status"] == "available"
    assert payload["role"] == "champion"


def test_non_xgboost_task_returns_explicit_unavailable(artifact_client: TestClient) -> None:
    """SHAP values are not invented for SARIMA/SARIMAX-only tasks."""
    response = artifact_client.get("/api/v1/explainability/RU_TR:74?horizon=1")

    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert response.json()["reason"] == "no_xgboost_role"
    assert "local" not in response.json()


def test_mirror_preserves_unpaired_post2022_and_target_column(artifact_client: TestClient) -> None:
    """RU_TR's target is importer-reported; post-2022 missing exporter data remain missing."""
    response = artifact_client.get("/api/v1/mirror-analytics?sector_id=RU_TR__10&limit=1000")

    payload = response.json()
    assert response.status_code == 200
    assert payload["structural_shift_status"] == "unavailable_no_paired_post_2022"
    assert payload["structural_shift_pp"] is None
    assert payload["summaries"][0]["paired_count"] > 0
    assert payload["summaries"][1]["paired_count"] == 0
    assert all(point["forecast_target_column"] == "mirror_value_usd" for point in payload["observations"])
    later = [point for point in payload["observations"] if point["period"] >= "2022-01"]
    assert later and all(point["mirror_gap_usd"] is None for point in later)
    assert payload["source"] == "UN Comtrade"


def test_mirror_paging_uses_full_selection_summary(artifact_client: TestClient) -> None:
    """Page offsets do not change summary denominators or the total selected count."""
    complete = artifact_client.get("/api/v1/mirror?direction=TR_RU&hs2=08&limit=1000").json()
    selected = artifact_client.get("/api/v1/mirror-analytics?direction=TR_RU&hs2=08&offset=3&limit=2").json()

    assert selected["summaries"] == complete["summaries"]
    assert selected["total_observations"] == complete["total_observations"]
    assert selected["observations"] == complete["observations"][3:5]
    assert all(point["forecast_target_column"] == "canonical_value_usd" for point in selected["observations"])


@pytest.mark.parametrize("url", [
    "/api/v1/forecasts/UNKNOWN:10", "/api/v1/leaderboard?hs2=XX",
    "/api/v1/explainability/TR_RU:99?horizon=1", "/api/v1/mirror-analytics?hs2=XX",
    "/api/v1/mirror-analytics?sector_id=RU_TR:XX", "/api/v1/indicators?indicator_id=unknown",
])
def test_unknown_catalog_keys_are_structured_404s(artifact_client: TestClient, url: str) -> None:
    """Unknown values have a consistent domain error envelope."""
    response = artifact_client.get(url)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "catalog_not_found"


@pytest.mark.parametrize("url", [
    "/api/v1/forecasts/TR_RU:08?horizon=2", "/api/v1/leaderboard?horizon=0",
    "/api/v1/leaderboard?direction=OTHER", "/api/v1/leaderboard?champion_model=sarimax",
    "/api/v1/explainability/TR_RU:08?horizon=1&target_month=2025-13",
    "/api/v1/explainability/TR_RU:08?horizon=1&target_month=2024-01",
    "/api/v1/explainability/TR_RU:08", "/api/v1/trade?sector_id=TR_RU:08&start_month=2025-00",
    "/api/v1/trade?sector_id=TR_RU:08&start_month=2025-12&end_month=2025-01",
    "/api/v1/mirror-analytics?limit=10001", "/api/v1/mirror-analytics?offset=-1",
])
def test_invalid_query_values_are_structured_422s(artifact_client: TestClient, url: str) -> None:
    """Invalid literals, dates and bounds fail at the HTTP contract boundary."""
    response = artifact_client.get(url)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
