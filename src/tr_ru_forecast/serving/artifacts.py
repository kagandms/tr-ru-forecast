"""Checksum-verified loading; manifest paths cannot escape configured roots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from tr_ru_forecast.serving.config import ServingSettings
from tr_ru_forecast.serving.schemas import EVALUATION_SCOPE

REQUIRED_FILES = (
    "configs/champion_registry_v1_1.yaml", "configs/selected_sectors_v1_1.yaml",
    "data/processed/forecast_target_monthly.parquet", "data/processed/macro_monthly.parquet",
    "data/processed/mirror_monthly.parquet", "data/processed/mirror_qa_summary.json",
    "data/processed/m10_selection_run.json", "data/processed/m10_champion_matrix.parquet",
    "data/processed/m10_cv_leaderboard.parquet", "data/processed/m11_holdout_run.json",
    "data/processed/m11_original_champion_registry_v1_1.yaml",
    "data/processed/m11_holdout_predictions.parquet", "data/processed/m11_holdout_leaderboard.parquet",
    "data/processed/m11_holdout_fit_log.parquet", "data/processed/m11_shap_values.parquet",
    "data/processed/m11_shap_summary.parquet",
)



class ArtifactError(Exception):
    """Expected missing/corrupt artifact or incompatible snapshot contract."""


class ServingManifest(BaseModel):
    """Immutable serving manifest contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    spec_version: str
    scope: str
    files: dict[str, str]


@dataclass(frozen=True)
class ArtifactTables:
    """Private load-time frames, never exposed to HTTP clients or mutated."""

    candidates: pd.DataFrame
    matrix: pd.DataFrame
    holdout: pd.DataFrame
    predictions: pd.DataFrame
    shap: pd.DataFrame
    shap_summary: pd.DataFrame
    history: pd.DataFrame
    mirror: pd.DataFrame
    macro: pd.DataFrame


def hash_file(path: Path) -> str:
    """Hash bytes. Args: path. Returns: SHA-256. Raises: OSError."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_artifact(settings: ServingSettings, relative: str) -> Path:
    """Resolve a frozen manifest key. Args: settings, relative. Returns: path. Raises: ArtifactError."""
    roots = {"data/processed/": settings.artifact_root, "configs/": settings.config_root}
    for prefix, directory in roots.items():
        if relative.startswith(prefix):
            candidate = (directory / relative.removeprefix(prefix)).resolve()
            if not candidate.is_relative_to(directory.resolve()):
                raise ArtifactError("Manifest path escapes artifact root")
            return candidate
    raise ArtifactError("Unsupported manifest path")


def verify_artifacts(settings: ServingSettings) -> tuple[ServingManifest, str]:
    """Verify pinned input bytes. Args: settings. Returns: manifest/digest. Raises: ArtifactError, OSError."""
    path = settings.config_root / "serving_artifacts_v1_1.yaml"
    manifest = ServingManifest.model_validate(yaml.safe_load(path.read_text()))
    if not set(REQUIRED_FILES).issubset(manifest.files):
        raise ArtifactError("Serving manifest omits required files")
    if manifest.spec_version != "1.1" or manifest.scope != EVALUATION_SCOPE:
        raise ArtifactError("Unsupported serving specification")
    for relative, expected in manifest.files.items():
        if hash_file(resolve_artifact(settings, relative)) != expected:
            raise ArtifactError("Serving checksum mismatch")
    _verify_upstream_outputs(settings, manifest)
    _verify_upstream_inputs(settings, manifest)
    return manifest, hash_file(path)


def read_document(path: Path) -> Mapping[str, object]:
    """Read object JSON. Args: path. Returns: document. Raises: ArtifactError, OSError."""
    document: object = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ArtifactError("Manifest must be a JSON object")
    return document


def _verify_upstream_outputs(settings: ServingSettings, manifest: ServingManifest) -> None:
    for name in ("m10_selection_run.json", "m11_holdout_run.json"):
        document = read_document(settings.artifact_root / name)
        outputs = document.get("output_hashes")
        if not isinstance(outputs, dict):
            raise ArtifactError("Upstream output checksums missing")
        for filename, digest in outputs.items():
            relative = f"data/processed/{filename}"
            if filename == "champion_registry_v1_1.yaml":
                relative = "data/processed/m11_original_champion_registry_v1_1.yaml"
            if relative in manifest.files and manifest.files[relative] != digest:
                raise ArtifactError("Upstream and serving manifest disagree")


def load_tables(settings: ServingSettings) -> ArtifactTables:
    """Load verified private frames. Args: settings. Returns: tables. Raises: I/O/schema errors."""
    directory = settings.artifact_root
    return ArtifactTables(
        candidates=pd.read_parquet(directory / "m10_cv_leaderboard.parquet"),
        matrix=pd.read_parquet(directory / "m10_champion_matrix.parquet"),
        holdout=pd.read_parquet(directory / "m11_holdout_leaderboard.parquet"),
        predictions=pd.read_parquet(directory / "m11_holdout_predictions.parquet"),
        shap=pd.read_parquet(directory / "m11_shap_values.parquet"),
        shap_summary=pd.read_parquet(directory / "m11_shap_summary.parquet"),
        history=pd.read_parquet(directory / "forecast_target_monthly.parquet"),
        mirror=pd.read_parquet(directory / "mirror_monthly.parquet"),
        macro=pd.read_parquet(directory / "macro_monthly.parquet"),
    )


def _verify_upstream_inputs(settings: ServingSettings, manifest: ServingManifest) -> None:
    for name in ("m10_selection_run.json", "m11_holdout_run.json"):
        document = read_document(settings.artifact_root / name)
        provenance = document.get("provenance", document)
        if not isinstance(provenance, dict):
            raise ArtifactError("Upstream provenance missing")
        checksums = provenance.get("input_hashes")
        if not isinstance(checksums, dict):
            raise ArtifactError("Upstream input checksums missing")
        for relative, digest in checksums.items():
            if relative in manifest.files and manifest.files[relative] != digest:
                raise ArtifactError("Upstream input and serving manifest disagree")
