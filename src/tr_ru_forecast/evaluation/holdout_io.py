"""Frozen holdout contract, gate verification, source loading and immutable publication."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from tr_ru_forecast.evaluation.champion_io import (
    ChampionRegistry,
    hash_artifact,
    load_frozen_folds,
    load_task_grid,
    validate_registry,
)
from tr_ru_forecast.models.ml.provenance import compute_source_fingerprint, get_git_provenance
from tr_ru_forecast.models.ml.records import JsonObject


class HoldoutSpec(BaseModel):
    """Explicit user-approved retrospective evaluation scope and frozen paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    spec_version: str
    holdout_version: str
    scope: str
    holdout_start: str
    holdout_end: str
    origin_policy: str
    training_policy: str
    trade_availability_policy: str
    trade_lag_months: int
    macro_availability_policy: str
    calibration_policy: str
    minimum_calibration_size: int
    negative_point_policy: str
    failed_fallback_policy: str
    pre_freeze_origins: str
    champion_registry: str
    macro_sources: str
    feature_spec: str
    sarima_orders: str
    xgboost_spec: str
    cv_predictions: str


@dataclass(frozen=True)
class HoldoutInputs:
    """Validated artifacts, isolated by domain and consumed without tuning."""

    root: Path
    spec: HoldoutSpec
    registry: ChampionRegistry
    targets: pd.DataFrame
    macro: pd.DataFrame
    sectors: JsonObject
    orders: JsonObject
    parameters: JsonObject
    cv_predictions: pd.DataFrame
    input_hashes: dict[str, str]


def load_holdout_spec(path: Path) -> HoldoutSpec:
    """Read frozen scope. Args: YAML path. Returns: spec. Raises: ValueError/validation errors."""
    spec = HoldoutSpec.model_validate(yaml.safe_load(path.read_text()))
    expected = {
        "scope": "retrospective_latest_vintage_holdout_evaluation_under_deterministic_lag_rules",
        "holdout_start": "2025-01",
        "holdout_end": "2025-12",
        "trade_lag_months": 0,
        "minimum_calibration_size": 19,
        "training_policy": "expanding_origin_cutoff_no_retuning",
        "calibration_policy": "cv_seed_plus_realized_holdout_residuals_asof_origin",
    }
    if any(getattr(spec, key) != value for key, value in expected.items()):
        raise ValueError("Unsupported frozen M11 protocol")
    return spec


def validate_pre_m11_gate(root: Path, spec: HoldoutSpec) -> ChampionRegistry:
    """Verify gate and frozen content before target reads. Args: root, spec. Raises: ValueError."""
    gate = yaml.safe_load((root / "configs/m11_gate_v1_1.yaml").read_text())
    if gate["status"] != "M11_READY":
        raise ValueError("M11 gate is not ready")
    registry = ChampionRegistry.model_validate(
        yaml.safe_load((root / spec.champion_registry).read_text())
    )
    validate_registry(registry, load_task_grid(root))
    load_frozen_folds(root)
    from tr_ru_forecast.evaluation.holdout_gate import (
        validate_m8_calendar_equivalence,
        validate_stage_a_source,
    )

    validate_stage_a_source(root, gate["stage_a_evidence"])
    if validate_m8_calendar_equivalence(root, registry) != gate["m8_provenance_equivalence"]:
        raise ValueError("M8 equivalence evidence changed")
    for relative, expected in gate["frozen_input_hashes"].items():
        if hash_artifact(root / relative) != expected:
            raise ValueError(f"Frozen checksum mismatch: {relative}")
    decisions = registry.model_dump(mode="json")["champions"]
    fingerprint = hashlib.sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest()
    if fingerprint != gate["champion_decision_sha256"]:
        raise ValueError("Frozen champion decisions changed")
    if any(row.champion_model == "sarimax" for row in registry.champions):
        raise ValueError("M8 equivalence excludes SARIMAX champion use")
    return registry


def load_holdout_inputs(root: Path) -> HoldoutInputs:
    """Load sources only after gate. Args: root. Returns: inputs. Raises: validation/I/O errors."""
    spec = load_holdout_spec(root / "configs/holdout_spec_v1_1.yaml")
    registry = validate_pre_m11_gate(root, spec)
    paths = _get_input_paths(root, spec)
    hashes = {str(path.relative_to(root)): hash_artifact(path) for path in paths}
    return HoldoutInputs(
        root,
        spec,
        registry,
        pd.read_parquet(root / "data/processed/forecast_target_monthly.parquet"),
        pd.read_parquet(root / "data/processed/macro_monthly.parquet"),
        yaml.safe_load((root / "configs/selected_sectors_v1_1.yaml").read_text()),
        yaml.safe_load((root / spec.sarima_orders).read_text()),
        yaml.safe_load((root / spec.xgboost_spec).read_text()),
        pd.read_parquet(root / spec.cv_predictions),
        hashes,
    )


def _get_input_paths(root: Path, spec: HoldoutSpec) -> list[Path]:
    relative = [
        spec.champion_registry,
        spec.feature_spec,
        spec.macro_sources,
        spec.sarima_orders,
        spec.xgboost_spec,
        spec.cv_predictions,
        "configs/m11_gate_v1_1.yaml",
        "configs/holdout_spec_v1_1.yaml",
        "configs/selected_sectors_v1_1.yaml",
        "configs/validation_folds_v1_1.yaml",
        "configs/sarimax_spec_v1_1.yaml",
        "data/processed/validation_origins.parquet",
        "data/processed/forecast_target_monthly.parquet",
        "data/processed/macro_monthly.parquet",
        "specification/EXPERIMENT_PROTOCOL.md",
    ]
    return [root / name for name in relative]


def build_holdout_provenance(inputs: HoldoutInputs) -> JsonObject:
    """Capture exact software/data identities. Args: inputs. Returns: metadata. Raises: I/O errors."""
    git = get_git_provenance(inputs.root)
    sources = sorted((inputs.root / "src/tr_ru_forecast/evaluation").glob("*.py"))
    sources += sorted((inputs.root / "src/tr_ru_forecast/models/ml").glob("*.py"))
    sources += sorted((inputs.root / "src/tr_ru_forecast/models/classical").glob("*.py"))
    sources += sorted((inputs.root / "src/tr_ru_forecast/features").glob("*.py"))
    sources.append(inputs.root / "src/tr_ru_forecast/models/ml/xgboost_template.yaml")
    return {
        "git_commit": git.commit,
        "is_working_tree_dirty": git.is_dirty,
        "dirty_paths": git.dirty_paths,
        "source_fingerprint": compute_source_fingerprint(sources),
        "input_hashes": inputs.input_hashes,
        "scope": inputs.spec.scope,
        "zero_retuning": True,
        "trade_lag_months": inputs.spec.trade_lag_months,
        "historical_vintage_fidelity_proven": False,
        "random_seed": 42,
    }


def publish_holdout_outputs(
    directory: Path, tables: dict[str, pd.DataFrame], documents: dict[str, str]
) -> dict[str, str]:
    """Publish without overwriting. Args: directory, tables, documents. Returns: hashes. Raises: FileExistsError."""
    names = list(tables) + list(documents)
    if any((directory / name).exists() for name in names):
        raise FileExistsError("M11 outputs exist; use a fresh M11_OUTPUT_DIR")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m11-") as temporary:
        stage = Path(temporary)
        for name, table in tables.items():
            if name.endswith(".csv"):
                table.to_csv(stage / name, index=False)
            else:
                table.to_parquet(stage / name, index=False)
        for name, content in documents.items():
            (stage / name).write_text(content, encoding="utf-8")
        hashes = {name: hash_artifact(stage / name) for name in names}
        for name in names:
            with (stage / name).open("rb") as source, (directory / name).open("xb") as target:
                shutil.copyfileobj(source, target)
    return hashes
