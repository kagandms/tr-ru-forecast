"""Explicit pre-M11 gate closure with preserved original registry and legacy evidence."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from tr_ru_forecast.evaluation.champion_io import ChampionRegistry, hash_artifact, load_frozen_folds
from tr_ru_forecast.models.ml.provenance import compute_source_fingerprint


def validate_m8_calendar_equivalence(root: Path, registry: ChampionRegistry) -> dict[str, str]:
    """Verify the accepted non-champion M8 equivalence. Args: root, registry. Raises: ValueError."""
    if any(entry.champion_model == "sarimax" for entry in registry.champions):
        raise ValueError("M8 equivalence cannot bypass a SARIMAX champion review")
    predictions = pd.read_parquet(root / "data/processed/m8_sarimax_predictions.parquet")
    columns = ["horizon", "forecast_origin", "forecast_period"]
    expected = load_frozen_folds(root)[columns].sort_values(columns).reset_index(drop=True)
    actual = predictions[columns].drop_duplicates().sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
    keys = ["direction", "hs2", "is_total", "horizon", "forecast_origin"]
    if len(predictions) != 3934 or predictions.duplicated(keys).any():
        raise ValueError("Incomplete M8 task-origin matrix")
    _verify_m8_targets(root, predictions, keys)
    metadata = json.loads((root / "data/processed/m8_sarimax_run.json").read_text())
    return {
        "rule": "M8_ROW_LEVEL_CALENDAR_AND_TARGET_EQUIVALENCE_NON_CHAMPION",
        "recorded_yaml_sha256": metadata["validation_fold_manifest_hash"],
        "current_yaml_sha256": hash_artifact(root / "configs/validation_folds_v1_1.yaml"),
        "m8_predictions_sha256": hash_artifact(
            root / "data/processed/m8_sarimax_predictions.parquet"
        ),
        "frozen_calendar_sha256": hash_artifact(root / "data/processed/validation_origins.parquet"),
    }


def audit_m9_structure(root: Path) -> None:
    """Enforce A2 length, types and forbidden escapes. Args: root. Returns: None. Raises: ValueError."""
    for path in (root / "src/tr_ru_forecast/models/ml").glob("*.py"):
        tree = ast.parse(path.read_text())
        if (
            any(isinstance(node, ast.Name) and node.id == "Any" for node in ast.walk(tree))
            or "type: ignore" in path.read_text()
        ):
            raise ValueError(f"Forbidden M9 type escape: {path}")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                if (
                    node.end_lineno - node.lineno + 1 > 30
                    or node.returns is None
                    or any(arg.annotation is None for arg in arguments if arg.arg != "self")
                ):
                    raise ValueError(f"M9 A2 structure remains open: {path.name}/{node.name}")


def freeze_pre_m11_gate(root: Path) -> dict[str, object]:
    """Unlock only after recorded M9 validation. Args: root. Returns: gate. Raises: validation/I/O errors."""
    audit_m9_structure(root)
    evidence = json.loads((root / "docs/M11_STAGE_A_VERIFICATION.json").read_text())
    sources = sorted((root / "src/tr_ru_forecast/models/ml").glob("*.py")) + [
        root / "src/tr_ru_forecast/models/ml/xgboost_template.yaml"
    ]
    if (
        evidence["source_fingerprint"] != compute_source_fingerprint(sources)
        or evidence["warnings"] != 0
    ):
        raise ValueError("Stage A evidence does not match current M9 source")
    path = root / "configs/champion_registry_v1_1.yaml"
    registry = ChampionRegistry.model_validate(yaml.safe_load(path.read_text()))
    equivalence = validate_m8_calendar_equivalence(root, registry)
    return _persist_gate(root, registry, (equivalence, evidence))


def _persist_gate(
    root: Path, registry: ChampionRegistry, context: tuple[dict[str, str], dict[str, object]]
) -> dict[str, object]:
    equivalence, evidence = context
    payload = _unlock_registry(root, registry)
    names = _get_frozen_gate_paths()
    gate = {
        "status": "M11_READY",
        "stage_a_evidence": evidence,
        "m8_provenance_equivalence": equivalence,
        "original_m10_registry_sha256": hash_artifact(
            root / "data/processed/m11_original_champion_registry_v1_1.yaml"
        ),
        "champion_decision_sha256": hashlib.sha256(
            json.dumps(payload["champions"], sort_keys=True).encode()
        ).hexdigest(),
        "frozen_input_hashes": {name: hash_artifact(root / name) for name in names},
    }
    (root / "configs/m11_gate_v1_1.yaml").write_text(
        yaml.safe_dump(gate, sort_keys=False), encoding="utf-8"
    )
    return gate


def _verify_m8_targets(root: Path, predictions: pd.DataFrame, keys: list[str]) -> None:
    baseline = pd.read_parquet(root / "data/processed/m6_seasonal_naive_predictions.parquet")
    paired = predictions.merge(
        baseline[keys + ["actual_value"]],
        on=keys,
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    if len(paired) != 3934 or not paired.actual_value.eq(paired.actual_value_baseline).all():
        raise ValueError("M8 scoring target mismatch")


def _get_frozen_gate_paths() -> list[str]:
    return [
        "data/processed/forecast_target_monthly.parquet",
        "data/processed/macro_monthly.parquet",
        "data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_predictions.parquet",
        "configs/champion_registry_v1_1.yaml",
        "data/processed/m11_original_champion_registry_v1_1.yaml",
        "configs/sarima_orders_v1_1.yaml",
        "configs/feature_spec_v1_1.yaml",
        "configs/macro_sources_v1_1.yaml",
        "configs/holdout_spec_v1_1.yaml",
        "data/processed/m9_acceptance_2026-10-14_r2/xgboost_spec_v1_1.yaml",
    ]


def _unlock_registry(root: Path, registry: ChampionRegistry) -> dict[str, object]:
    archived = root / "data/processed/m11_original_champion_registry_v1_1.yaml"
    current = root / "configs/champion_registry_v1_1.yaml"
    if not archived.exists():
        archived.write_bytes(current.read_bytes())
    expected = json.loads((root / "data/processed/m10_selection_run.json").read_text())[
        "output_hashes"
    ]["champion_registry_v1_1.yaml"]
    if hash_artifact(archived) != expected:
        raise ValueError("Archived registry differs from M10 original checksum")
    original = ChampionRegistry.model_validate(yaml.safe_load(archived.read_text()))
    if original.champions != registry.champions:
        raise ValueError("Champion assignments changed since M10 freeze")
    payload = registry.model_dump(mode="json")
    payload["status"] = "CV_SELECTION_FROZEN_M11_READY"
    current.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def validate_stage_a_source(root: Path, evidence: dict[str, object]) -> None:
    """Verify current source against Stage A evidence. Args: root, evidence. Raises: ValueError."""
    audit_m9_structure(root)
    sources = sorted((root / "src/tr_ru_forecast/models/ml").glob("*.py")) + [
        root / "src/tr_ru_forecast/models/ml/xgboost_template.yaml"
    ]
    if (
        compute_source_fingerprint(sources) != evidence["source_fingerprint"]
        or evidence["warnings"] != 0
    ):
        raise ValueError("Stage A code or verification evidence changed")
