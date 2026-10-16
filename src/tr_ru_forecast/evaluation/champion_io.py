"""CV-only input boundary, schema validation and frozen registry persistence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, FiniteFloat

from tr_ru_forecast.evaluation.champion_selector import (
    FOLD_COLUMNS,
    MODEL_FAMILIES,
    TASK_COLUMNS,
    Candidate,
    ModelFamily,
    Selection,
    SelectionReason,
    SelectionSpec,
    validate_selection_spec,
)
from tr_ru_forecast.evaluation.splits import generate_validation_folds
from tr_ru_forecast.models.ml.provenance import compute_source_fingerprint, get_git_provenance

CHAMPION_METRICS = (
    "mean_smape",
    "std_smape",
    "median_smape",
    "max_fold_smape",
    "mean_mae",
    "mean_rmse",
    "n_valid_folds",
)
CONFIG_INPUTS = (
    "selected_sectors_v1_1.yaml",
    "validation_folds_v1_1.yaml",
    "feature_spec_v1_1.yaml",
    "sarima_orders_v1_1.yaml",
    "sarimax_spec_v1_1.yaml",
    "xgboost_spec_v1_1.yaml",
    "macro_sources_v1_1.yaml",
)
PREDICTION_NAMES = dict(
    zip(
        MODEL_FAMILIES,
        (
            "m6_seasonal_naive_predictions.parquet",
            "m7_sarima_predictions.parquet",
            "m8_sarimax_predictions.parquet",
            "m9_xgboost_predictions.parquet",
        ),
        strict=True,
    )
)


class ModelReference(BaseModel):
    """Frozen configuration address for downstream refits; never a fitted-model claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    config_path: str
    config_sha256: str
    config_locator: str
    feature_spec_path: str
    feature_spec_sha256: str
    interval_policy: str


class ChampionEntry(BaseModel):
    """Strict machine-readable record for one sector and horizon."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    sector_id: str
    direction: Literal["TR_RU", "RU_TR"]
    hs2: str
    is_total: bool
    horizon: Literal[1, 3, 6]
    champion_model: ModelFamily
    fallback_model: ModelFamily
    raw_best_model: ModelFamily
    selection_reason: SelectionReason
    mean_smape: FiniteFloat
    std_smape: FiniteFloat
    median_smape: FiniteFloat
    max_fold_smape: FiniteFloat
    mean_mae: FiniteFloat
    mean_rmse: FiniteFloat
    skill_vs_naive_pct: FiniteFloat
    n_valid_folds: int
    has_stability_flag: bool
    baseline_reference: Candidate
    champion_reference: ModelReference
    fallback_reference: ModelReference


class Provenance(BaseModel):
    """Current source and exact consumed input/output content identities."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    git_commit: str
    is_working_tree_dirty: bool
    dirty_paths: tuple[str, ...]
    created_at: str
    source_fingerprint: str
    input_hashes: dict[str, str]
    output_hashes: dict[str, str]
    legacy_metadata_discrepancies: tuple[str, ...]
    holdout_accessed: Literal[False] = False


class ChampionRegistry(BaseModel):
    """42-task registry with strict required metrics and no manual overrides."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    spec_version: Literal["1.1"] = "1.1"
    registry_version: Literal["1.1.0"] = "1.1.0"
    status: Literal["CV_SELECTION_FROZEN_M11_GATED", "CV_SELECTION_FROZEN_M11_READY"] = (
        "CV_SELECTION_FROZEN_M11_GATED"
    )
    provenance: Provenance
    champions: tuple[ChampionEntry, ...]


def load_selection_spec(path: Path) -> SelectionSpec:
    """Load YAML. Args: path. Returns: validated spec. Raises: validation/I/O errors."""
    spec = SelectionSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    validate_selection_spec(spec)
    return spec


def hash_artifact(path: Path) -> str:
    """Hash file bytes. Args: path. Returns: SHA-256. Raises: OSError for missing files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_cv_input(root: Path, relative: str, expected_name: str) -> Path:
    """Allow only named processed CV artifacts. Args: root, path, expected filename.

    Returns: resolved path. Raises: ValueError for path escapes or forbidden inputs.
    """
    artifact = (root / relative).resolve()
    processed = (root / "data/processed").resolve()
    if not artifact.is_relative_to(processed) or artifact.name != expected_name:
        raise ValueError(f"Forbidden CV input path: {relative}")
    return artifact


def load_task_grid(root: Path) -> pd.DataFrame:
    """Build 42-task universe. Args: root. Returns: grid. Raises: ValueError for invalid freeze."""
    sectors = yaml.safe_load((root / "configs/selected_sectors_v1_1.yaml").read_text())
    manifest = yaml.safe_load((root / "configs/validation_folds_v1_1.yaml").read_text())
    rows: list[dict[str, str | int | bool]] = []
    for direction in ("TR_RU", "RU_TR"):
        codes = ["TOTAL"] + [str(entry["hs2"]).zfill(2) for entry in sectors[direction]]
        for code in codes:
            for horizon in (1, 3, 6):
                rows.append(
                    {
                        "direction": direction,
                        "hs2": code,
                        "is_total": code == "TOTAL",
                        "horizon": horizon,
                        "n_expected_folds": manifest["horizons"][f"h{horizon}"][
                            "number_of_origins"
                        ],
                    }
                )
    grid = pd.DataFrame(rows)
    if len(grid) != 42 or grid.duplicated(TASK_COLUMNS).any():
        raise ValueError("Frozen sector universe must contain exactly 42 unique tasks")
    return grid


def load_frozen_folds(root: Path) -> pd.DataFrame:
    """Read only the CV calendar. Args: root. Returns: frozen folds. Raises: ValueError."""
    frozen = pd.read_parquet(root / "data/processed/validation_origins.parquet")
    expected = generate_validation_folds()
    pd.testing.assert_frame_equal(frozen.reset_index(drop=True), expected.reset_index(drop=True))
    manifest = yaml.safe_load((root / "configs/validation_folds_v1_1.yaml").read_text())
    if manifest["development_end"] != "2024-12" or manifest["total_folds"] != len(expected):
        raise ValueError("CV calendar manifest differs from frozen development protocol")
    for horizon in (1, 3, 6):
        if manifest["horizons"][f"h{horizon}"]["number_of_origins"] != int(
            expected.horizon.eq(horizon).sum()
        ):
            raise ValueError("CV manifest fold count mismatch")
    return frozen


def _validate_calendar(predictions: pd.DataFrame, folds: pd.DataFrame) -> None:
    if predictions[FOLD_COLUMNS].isna().any().any() or predictions.duplicated(FOLD_COLUMNS).any():
        raise ValueError("Null or duplicate CV keys")
    calendar = predictions.merge(
        folds[["horizon", "forecast_origin", "forecast_period"]],
        on=["horizon", "forecast_origin", "forecast_period"],
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    if not calendar["_merge"].eq("both").all():
        raise ValueError("Prediction outside frozen CV calendar; holdout/post-CV access forbidden")
    for column in (
        "train_start",
        "train_end",
        "train_feature_end",
        "train_label_end",
        "seasonal_source_period",
    ):
        if column not in predictions:
            continue
        values = predictions[column]
        if values.isna().any() or values.gt(predictions["forecast_origin"]).any():
            raise ValueError(f"Invalid training/source availability: {column}")
        if column == "train_start" and not values.eq("2012-01").all():
            raise ValueError("Training window must start at frozen development start")
        if column == "train_end" and not values.eq(predictions["forecast_origin"]).all():
            raise ValueError("Classical training end must equal forecast origin")


def normalize_cv_predictions(
    predictions: pd.DataFrame, folds: pd.DataFrame, family: ModelFamily
) -> pd.DataFrame:
    """Normalize family schemas. Args: predictions, frozen calendar, family. Returns: CV rows.

    Raises: ValueError for duplicate/out-of-CV keys. Invalid forecasts are retained and flagged.
    """
    normalized = predictions.rename(columns={"actual": "actual_value"}).copy()
    _validate_calendar(normalized, folds)
    _validate_model_metadata(normalized, family)
    expected_type = "xgboost_direct" if family == "xgboost" else family
    if not normalized["model_type"].eq(expected_type).all():
        raise ValueError("Prediction model family mismatch")
    normalized["model_family"] = family
    normalized = _attach_family_fit_status(normalized, family)
    numeric = normalized[["actual_value", "prediction"]].to_numpy(dtype=float)
    normalized["is_valid_prediction"] = (
        np.isfinite(numeric).all(axis=1)
        & (numeric >= 0).all(axis=1)
        & normalized["has_successful_fit"]
    )
    absolute = normalized["actual_value"].sub(normalized["prediction"]).abs()
    denominator = normalized["actual_value"].abs().add(normalized["prediction"].abs())
    normalized["smape_component"] = absolute.div(denominator.where(denominator != 0, 1)).mul(200)
    normalized["absolute_error"] = absolute
    normalized["squared_error"] = absolute.pow(2)
    return normalized


def _attach_xgboost_fit_status(predictions: pd.DataFrame, fit_log: pd.DataFrame) -> pd.DataFrame:
    outer = fit_log.loc[fit_log["stage"] == "outer_validation"].rename(
        columns={"origin": "forecast_origin"}
    )
    keys = ["direction", "hs2", "horizon", "forecast_origin"]
    if outer.duplicated(keys).any():
        raise ValueError("Duplicate XGBoost fit evidence")
    joined = predictions.merge(
        outer[keys + ["status", "candidate_id"]], on=keys, how="left", validate="one_to_one"
    )
    joined["has_successful_fit"] = joined["status"].eq("SUCCESS") & joined["parameter_set_id"].eq(
        joined["candidate_id"]
    )
    return joined


def load_cv_predictions(
    root: Path, spec: SelectionSpec, context: tuple[pd.DataFrame, pd.DataFrame]
) -> pd.DataFrame:
    """Read allowlisted CV predictions. Args: root, spec, (folds, tasks). Returns: audit rows.

    Raises: ValueError on family, task, actual-value or calendar mismatch. No raw targets read.
    """
    folds, tasks = context
    frames: list[pd.DataFrame] = []
    for family in MODEL_FAMILIES:
        path = resolve_cv_input(root, spec.inputs[family], PREDICTION_NAMES[family])
        calendar_columns = FOLD_COLUMNS + ["train_start"]
        _validate_calendar(pd.read_parquet(path, columns=calendar_columns), folds)
        predictions = pd.read_parquet(path)
        if family == "xgboost":
            fit_path = resolve_cv_input(root, spec.xgboost_fit_log, "m9_xgboost_fit_log.parquet")
            predictions = _attach_xgboost_fit_status(predictions, pd.read_parquet(fit_path))
            _validate_xgboost_parameter_ids(
                predictions, resolve_cv_input(root, spec.xgboost_config, "xgboost_spec_v1_1.yaml")
            )
        normalized = normalize_cv_predictions(predictions, folds, family)
        _validate_task_membership(normalized, tasks)
        frames.append(normalized)
    combined = pd.concat(frames, ignore_index=True)
    _validate_shared_actuals(combined)
    return _extract_audit_columns(combined)


def _validate_task_membership(predictions: pd.DataFrame, tasks: pd.DataFrame) -> None:
    joined = predictions.merge(tasks[TASK_COLUMNS], on=TASK_COLUMNS, how="left", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Unexpected sector, direction, total flag or horizon")


def _validate_shared_actuals(predictions: pd.DataFrame) -> None:
    baseline = predictions.loc[
        predictions["model_family"] == "seasonal_naive", FOLD_COLUMNS + ["actual_value"]
    ]
    joined = predictions.merge(
        baseline, on=FOLD_COLUMNS, suffixes=("", "_baseline"), validate="many_to_one"
    )
    if not np.array_equal(joined.actual_value, joined.actual_value_baseline, equal_nan=True):
        raise ValueError("Candidates must be scored against identical baseline actuals")


def build_model_reference(root: Path, candidate: Candidate, spec: SelectionSpec) -> ModelReference:
    """Address frozen parameters. Args: root, candidate, spec. Returns: reference. Raises: OSError."""
    paths = {
        "seasonal_naive": "configs/champion_selection_spec_v1_1.yaml",
        "sarima": "configs/sarima_orders_v1_1.yaml",
        "sarimax": "configs/sarimax_spec_v1_1.yaml",
        "xgboost": spec.xgboost_config,
    }
    locators = {
        "seasonal_naive": "seasonal_offset=12",
        "sarima": f"series_orders[direction={candidate.direction},hs2={candidate.hs2}]",
        "sarimax": f"structural_order_policy + configs/sarima_orders_v1_1.yaml:{candidate.direction}/{candidate.hs2}",
        "xgboost": f"fixed_parameters + tuning.selected_parameters.h{candidate.horizon}",
    }
    intervals = {
        "seasonal_naive": "unavailable_no_interval_method_approved",
        "sarima": "native_80_95",
        "sarimax": "native_80_95_conditioned_on_macro_persistence",
        "xgboost": "prequential_conformal_80_95_requires_19_residuals",
    }
    config_path = paths[candidate.model_family]
    feature_path = "configs/feature_spec_v1_1.yaml"
    return ModelReference(
        config_path=config_path,
        config_sha256=hash_artifact(root / config_path),
        config_locator=locators[candidate.model_family],
        feature_spec_path=feature_path,
        feature_spec_sha256=hash_artifact(root / feature_path),
        interval_policy=intervals[candidate.model_family],
    )


def build_registry_entry(root: Path, selection: Selection, spec: SelectionSpec) -> ChampionEntry:
    """Serialize selection. Args: root, selection, spec. Returns: strict entry. Raises: ValidationError."""
    champion = selection.champion
    metrics = {name: getattr(champion, name) for name in CHAMPION_METRICS}
    return ChampionEntry.model_validate(
        {
            **metrics,
            "sector_id": f"{champion.direction}:{champion.hs2}",
            "direction": champion.direction,
            "hs2": champion.hs2,
            "is_total": champion.is_total,
            "horizon": champion.horizon,
            "champion_model": champion.model_family,
            "fallback_model": selection.fallback.model_family,
            "raw_best_model": selection.raw_best_model,
            "selection_reason": selection.selection_reason,
            "skill_vs_naive_pct": champion.delta_smape_vs_naive_pct,
            "has_stability_flag": selection.has_stability_flag,
            "baseline_reference": selection.baseline_reference,
            "champion_reference": build_model_reference(root, champion, spec),
            "fallback_reference": build_model_reference(root, selection.fallback, spec),
        }
    )


def validate_registry(registry: ChampionRegistry, tasks: pd.DataFrame) -> None:
    """Validate full task coverage. Args: registry, grid. Returns: None. Raises: ValueError."""
    expected = set(zip(tasks.direction, tasks.hs2, tasks.horizon, strict=True))
    actual = [(row.direction, row.hs2, row.horizon) for row in registry.champions]
    if len(actual) != 42 or len(set(actual)) != 42 or set(actual) != expected:
        raise ValueError("Registry requires exactly the 42 frozen unique tasks")
    expected_counts = {
        (row["direction"], row["hs2"], row["horizon"]): row["n_expected_folds"]
        for row in tasks.to_dict("records")
    }
    for row in registry.champions:
        if row.n_valid_folds != expected_counts[row.direction, row.hs2, row.horizon]:
            raise ValueError("Champion must have every frozen CV origin")
        if row.sector_id != f"{row.direction}:{row.hs2}" or row.is_total != (row.hs2 == "TOTAL"):
            raise ValueError("Registry sector identity mismatch")


def collect_provenance(root: Path, spec: SelectionSpec, spec_path: Path) -> Provenance:
    """Capture consumed input identities. Args: root, spec, spec path. Returns: provenance.

    Raises: ValueError on M9 content mismatch. Legacy M8 manifest discrepancy is retained.
    """
    paths = _get_provenance_paths(root, spec, spec_path)
    hashes = {str(path.relative_to(root)): hash_artifact(path) for path in paths}
    _validate_m9_provenance(root, spec, hashes)
    git = get_git_provenance(root)
    sources = sorted((root / "src/tr_ru_forecast/evaluation").glob("*.py"))
    sources += [
        root / "src/tr_ru_forecast/models/ml/provenance.py",
        root / "specification/EXPERIMENT_PROTOCOL.md",
    ]
    return Provenance(
        git_commit=git.commit,
        is_working_tree_dirty=git.is_dirty,
        dirty_paths=git.dirty_paths,
        created_at=datetime.now(UTC).isoformat(),
        source_fingerprint=compute_source_fingerprint(sources),
        input_hashes=hashes,
        output_hashes={},
        legacy_metadata_discrepancies=_collect_legacy_discrepancies(root, hashes),
    )


def _validate_m9_provenance(root: Path, spec: SelectionSpec, hashes: dict[str, str]) -> None:
    metadata = json.loads((root / spec.xgboost_run).read_text())
    for artifact, relative in [
        ("predictions", spec.inputs["xgboost"]),
        ("fit_log", spec.xgboost_fit_log),
        ("xgboost_spec_manifest", spec.xgboost_config),
    ]:
        if metadata["output_artifact_hashes"][artifact] != hashes[relative]:
            raise ValueError(f"M9 checksum mismatch: {artifact}")
    for name, relative in [
        ("selected_sectors_manifest", "configs/selected_sectors_v1_1.yaml"),
        ("feature_spec_manifest", "configs/feature_spec_v1_1.yaml"),
        ("validation_folds_manifest", "configs/validation_folds_v1_1.yaml"),
        ("validation_folds_artifact", "data/processed/validation_origins.parquet"),
    ]:
        if metadata["input_artifacts"][f"{name}_hash"] != hashes[relative]:
            raise ValueError(f"M9 frozen input mismatch: {name}")
    manifest = yaml.safe_load((root / spec.xgboost_config).read_text())
    if metadata["selected_parameters"] != manifest["tuning"]["selected_parameters"]:
        raise ValueError("M9 selected parameters differ from frozen run manifest")


def _collect_legacy_discrepancies(root: Path, hashes: dict[str, str]) -> tuple[str, ...]:
    warnings: list[str] = []
    for prefix in ("m7_sarima", "m8_sarimax"):
        metadata = json.loads((root / f"data/processed/{prefix}_run.json").read_text())
        if (
            metadata["validation_fold_manifest_hash"]
            != hashes["configs/validation_folds_v1_1.yaml"]
        ):
            warnings.append(
                f"{prefix}: recorded validation YAML hash differs; exact row-level CV calendar independently verified"
            )
    return tuple(warnings)


def _attach_family_fit_status(normalized: pd.DataFrame, family: ModelFamily) -> pd.DataFrame:
    if family in ("sarima", "sarimax"):
        normalized["has_successful_fit"] = normalized["converged"].eq(True) & normalized[
            "fit_status"
        ].eq("SUCCESS")
    elif family == "seasonal_naive":
        normalized["has_successful_fit"] = True
    elif "has_successful_fit" not in normalized:
        raise ValueError("XGBoost fit evidence is required")
    return normalized


def _extract_audit_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = FOLD_COLUMNS + [
        "model_family",
        "actual_value",
        "prediction",
        "has_successful_fit",
        "is_valid_prediction",
        "smape_component",
        "absolute_error",
        "squared_error",
    ]
    return predictions[columns].sort_values(FOLD_COLUMNS + ["model_family"]).reset_index(drop=True)


def _get_provenance_paths(root: Path, spec: SelectionSpec, spec_path: Path) -> list[Path]:
    paths = [root / "configs" / name for name in CONFIG_INPUTS] + [spec_path]
    paths += [
        resolve_cv_input(root, spec.inputs[family], PREDICTION_NAMES[family])
        for family in MODEL_FAMILIES
    ]
    paths += [root / "data/processed/validation_origins.parquet"]
    paths += [
        resolve_cv_input(root, relative, name)
        for relative, name in (
            (spec.xgboost_fit_log, "m9_xgboost_fit_log.parquet"),
            (spec.xgboost_run, "m9_xgboost_run.json"),
            (spec.xgboost_config, "xgboost_spec_v1_1.yaml"),
        )
    ]
    paths += [
        root / f"data/processed/{name}_run.json"
        for name in ("m6_seasonal_naive", "m7_sarima", "m8_sarimax")
    ]
    return paths


def _validate_model_metadata(predictions: pd.DataFrame, family: ModelFamily) -> None:
    if family == "seasonal_naive":
        expected_source = (pd.PeriodIndex(predictions["forecast_period"], freq="M") - 12).astype(
            str
        )
        if not predictions["seasonal_source_period"].eq(expected_source).all():
            raise ValueError("Seasonal Naive source must equal target month minus 12")
        if (
            "seasonal_source_value" in predictions
            and not predictions["prediction"].eq(predictions["seasonal_source_value"]).all()
        ):
            raise ValueError("Seasonal Naive prediction differs from seasonal source value")
    if family == "xgboost":
        expected_feature_end = (
            pd.PeriodIndex(predictions["forecast_origin"], freq="M")
            - predictions["horizon"].to_numpy()
        ).astype(str)
        if (
            not predictions["train_feature_end"].eq(expected_feature_end).all()
            or not predictions["train_label_end"].eq(predictions["forecast_origin"]).all()
        ):
            raise ValueError(
                "Direct XGBoost training features/labels violate frozen origin geometry"
            )


def _validate_xgboost_parameter_ids(predictions: pd.DataFrame, config_path: Path) -> None:
    manifest = yaml.safe_load(config_path.read_text())
    expected_ids = predictions["horizon"].map(
        {
            horizon: manifest["tuning"]["selected_parameters"][f"h{horizon}"]["candidate_id"]
            for horizon in (1, 3, 6)
        }
    )
    if not predictions["parameter_set_id"].eq(expected_ids).all():
        raise ValueError("XGBoost predictions differ from frozen per-horizon parameters")
