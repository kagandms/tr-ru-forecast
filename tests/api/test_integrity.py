"""Snapshot integrity, caching, configured CORS and safe error contract tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import create_app
from tr_ru_forecast.serving.artifacts import ArtifactError, resolve_artifact
from tr_ru_forecast.serving.config import ServingSettings
from tr_ru_forecast.serving.schemas import MetricSummary


@pytest.fixture
def writable_bundle(tmp_path: Path, serving_settings: ServingSettings) -> ServingSettings:
    """Copy only the pinned serving files into disposable roots for failure injection."""
    configured = ServingSettings(artifact_root=tmp_path / "processed", config_root=tmp_path / "configs")
    configured.config_root.mkdir()
    manifest_path = serving_settings.config_root / "serving_artifacts_v1_1.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    shutil.copy2(manifest_path, configured.config_root / manifest_path.name)
    for relative in manifest["files"]:
        source = resolve_artifact(serving_settings, relative)
        destination = resolve_artifact(configured, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return configured


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "empty_manifest", "bad_scope"])
def test_missing_or_corrupt_bundle_fails_closed(writable_bundle: ServingSettings, mutation: str) -> None:
    """Neither missing files nor a weakened manifest can publish a healthy partial snapshot."""
    _damage_bundle(writable_bundle, mutation)

    with TestClient(create_app(writable_bundle)) as client:
        health = client.get("/api/v1/health")
        response = client.get("/api/v1/sectors")

    assert health.status_code == 503
    assert health.json()["status"] == "degraded"
    assert health.json()["checksum_status"] == "failed"
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "artifact_snapshot_unavailable"
    assert str(writable_bundle.artifact_root) not in response.text


def _damage_bundle(settings: ServingSettings, mutation: str) -> None:
    artifact = settings.artifact_root / "m11_holdout_predictions.parquet"
    if mutation == "missing":
        artifact.unlink()
        return
    if mutation == "corrupt":
        artifact.write_bytes(b"corrupt")
        return
    path = settings.config_root / "serving_artifacts_v1_1.yaml"
    manifest = yaml.safe_load(path.read_text())
    manifest["files" if mutation == "empty_manifest" else "scope"] = {} if mutation == "empty_manifest" else "wrong"
    path.write_text(yaml.safe_dump(manifest))


def test_cached_snapshot_does_not_reread_changed_files(writable_bundle: ServingSettings) -> None:
    """A loaded snapshot remains coherent until restart; a new process rejects corrupted bytes."""
    with TestClient(create_app(writable_bundle)) as client:
        initial = client.get("/api/v1/leaderboard").json()
        (writable_bundle.artifact_root / "m11_holdout_predictions.parquet").unlink()

        assert client.get("/api/v1/leaderboard").json() == initial
        assert client.get("/health").json()["checksum_status"] == "verified"
    with TestClient(create_app(writable_bundle)) as restarted:
        assert restarted.get("/health").status_code == 503


def test_cors_allows_only_configured_frontend(artifact_client: TestClient) -> None:
    """CORS exposes read-only requests to exact frontend origins without credentials."""
    allowed = artifact_client.options("/api/v1/sectors", headers={
        "Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"})
    denied = artifact_client.options("/api/v1/sectors", headers={
        "Origin": "https://untrusted.example", "Access-Control-Request-Method": "GET"})

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-credentials" not in allowed.headers
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


def test_nonfinite_metrics_serialize_as_json_null() -> None:
    """NaN and infinity cannot escape typed metric responses as invalid JSON tokens."""
    metric = MetricSummary(mean_smape=float("nan"), mean_mae=float("inf"),
                           mean_rmse=float("-inf"), valid_origins=0, expected_origins=12)

    assert json.loads(metric.model_dump_json())["mean_smape"] is None
    assert metric.mean_mae is metric.mean_rmse is None


def test_manifest_path_traversal_is_rejected(serving_settings: ServingSettings) -> None:
    """Manifest keys cannot read files outside the explicitly configured roots."""
    with pytest.raises(ArtifactError):
        resolve_artifact(serving_settings, "data/processed/../../secrets")


@pytest.mark.parametrize("origins", [("*",), ("file:///tmp",)])
def test_unsafe_cors_configuration_is_rejected(origins: tuple[str, ...]) -> None:
    """Wildcard and non-web origins cannot enter the production CORS configuration."""
    with pytest.raises(ValidationError):
        ServingSettings(cors_origins=origins)


def test_openapi_exposes_typed_contracts_and_no_writes(artifact_client: TestClient) -> None:
    """Every public route is read-only and explanation availability is discriminated."""
    document = artifact_client.get("/openapi.json").json()

    assert all(set(operations).issubset({"get"}) for operations in document["paths"].values())
    explanation = document["paths"]["/api/v1/explainability/{sector_id}"]["get"]["responses"]["200"]
    assert explanation["content"]["application/json"]["schema"]["discriminator"]["propertyName"] == "status"
    assert "ForecastResponse" in document["components"]["schemas"]
    comparison = document["components"]["schemas"]["TaskComparison"]["properties"]
    assert {"selection_reason", "has_stability_flag"} <= set(comparison)


def test_environment_paths_and_cors_are_configurable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Deployments can bind custom roots/origins without editing pipeline settings."""
    monkeypatch.setenv("SERVING_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("SERVING_CORS_ORIGINS", '["https://dashboard.example"]')

    configured = ServingSettings()

    assert configured.artifact_root == tmp_path
    assert configured.cors_origins == ("https://dashboard.example",)


def test_mutating_requests_are_not_supported(artifact_client: TestClient) -> None:
    """A public client cannot start training or write through the read API."""
    response = artifact_client.post("/api/v1/forecasts/RU_TR:10", json={"prediction": 0})

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "http_error"


def test_repeated_calls_are_deterministic(artifact_client: TestClient) -> None:
    """Frozen responses have no request-time randomness or mutable model state."""
    first = artifact_client.get("/api/v1/forecasts/RU_TR:10?horizon=3")

    assert artifact_client.get("/api/v1/forecasts/RU_TR:10?horizon=3").content == first.content


def test_frozen_models_reject_mutation() -> None:
    """Cached response records are immutable at the model boundary."""
    metric = MetricSummary(mean_smape=1, mean_mae=2, mean_rmse=3, valid_origins=12, expected_origins=12)

    with pytest.raises(ValidationError):
        metric.mean_smape = 9


@pytest.mark.parametrize("name", ["m10_selection_run.json", "m11_holdout_run.json"])
def test_repinning_manifest_cannot_bypass_upstream_output_checksum(
    writable_bundle: ServingSettings, name: str
) -> None:
    """Serving pins must agree with the original model-run output identities."""
    path = writable_bundle.artifact_root / name
    document = json.loads(path.read_text())
    output = next(key for key in document["output_hashes"] if key.endswith(".parquet"))
    document["output_hashes"][output] = "0" * 64
    path.write_text(json.dumps(document))
    _repin_file(writable_bundle, path)

    with TestClient(create_app(writable_bundle)) as client:
        assert client.get("/health").status_code == 503


def _repin_file(settings: ServingSettings, path: Path) -> None:
    import hashlib

    manifest_path = settings.config_root / "serving_artifacts_v1_1.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["files"]["data/processed/" + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(yaml.safe_dump(manifest))


def test_m12_structure_contract_is_enforced() -> None:
    """The audit covers every new API, service and test function rather than a sample."""
    from scripts.verify_serving import audit_serving_structure

    assert audit_serving_structure(Path(__file__).resolve().parents[2]) == ()
