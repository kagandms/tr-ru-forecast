"""M9 orchestration with typed fold state and unchanged numerical execution order."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml

from tr_ru_forecast import __version__
from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.evaluation.metrics import compute_smape_single
from tr_ru_forecast.models.baselines.pipeline import extract_modeling_series_list
from tr_ru_forecast.models.ml.conformal import PrequentialResidualCalibrator
from tr_ru_forecast.models.ml.constants import (
    FIXED_XGB_PARAMS,
    FORECAST_HORIZONS,
    M9_PREDICTORS,
    M9_RUN_ID,
    MODEL_TYPE,
    MODEL_VERSION,
    SPEC_VERSION,
)
from tr_ru_forecast.models.ml.provenance import compute_source_fingerprint, get_git_provenance
from tr_ru_forecast.models.ml.records import ForecastIdentity, JsonObject
from tr_ru_forecast.models.ml.row_builder import (
    FoldValues,
    build_fit_row,
    build_interval_row,
    build_prediction_row,
)
from tr_ru_forecast.models.ml.shap_explainer import (
    compute_global_shap_importance,
    explain_fold_prediction,
)
from tr_ru_forecast.models.ml.tuner import run_pre_validation_tuning
from tr_ru_forecast.models.ml.xgboost_model import (
    build_xgb_regressor,
    extract_origin_features,
    get_training_subset,
    prepare_training_xy,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class PipelinePaths:
    """Explicit input and output boundaries of a single M9 run."""

    root: Path
    input_directory: Path
    output_directory: Path
    specification_path: Path


@dataclass(frozen=True)
class FoldTask:
    """Frozen identity, parameters and sequential residual state."""

    identity: ForecastIdentity
    is_total: bool
    parameters: JsonObject
    calibrator: PrequentialResidualCalibrator


def compute_file_sha256(path: Path) -> str:
    """Hash required artifact. Args: path. Returns: SHA-256. Raises: FileNotFoundError."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        while chunk := artifact.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def generate_xgboost_manifest(
    selected_by_horizon: dict[int, JsonObject], output_path: Path
) -> JsonObject:
    """Persist identical frozen manifest fields. Args: parameters, output. Raises: I/O errors."""
    manifest = yaml.safe_load(Path(__file__).with_name("xgboost_template.yaml").read_text())
    manifest["created_at"] = datetime.now(UTC).isoformat()
    manifest["tuning"]["selected_parameters"] = {
        f"h{h}": {
            name: selected_by_horizon[h][name]
            for name in (
                "candidate_id",
                "max_depth",
                "min_child_weight",
                "learning_rate",
                "n_estimators",
            )
        }
        for h in FORECAST_HORIZONS
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest


def _compute_fold(series: pd.DataFrame, task: FoldTask) -> tuple[FoldValues, list[JsonObject]]:
    identity = task.identity
    training = get_training_subset(series, identity.forecast_origin, identity.horizon)
    if training.empty:
        raise ValueError(f"No training rows available for {identity}")
    predictors, labels = prepare_training_xy(training, identity.horizon)
    origin_features = extract_origin_features(series, identity.forecast_origin)
    model = build_xgb_regressor(task.parameters)
    model.fit(predictors, labels)
    prediction = float(model.predict(origin_features)[0])
    actual = float(
        series.loc[series.period.eq(identity.forecast_origin), f"target_h{identity.horizon}"].iloc[
            0
        ]
    )
    calibration = task.calibrator.calibrate(identity.forecast_origin, prediction, actual)
    task.calibrator.record_historical_prediction(
        identity.forecast_origin, identity.forecast_period, actual, prediction
    )
    _, _, local = explain_fold_prediction(
        model,
        origin_features,
        prediction,
        identity.direction,
        identity.hs2,
        identity.horizon,
        identity.forecast_origin,
        identity.forecast_period,
    )
    return _build_fold_values(training, task, (actual, prediction, calibration)), local


def _run_series_folds(
    series: pd.DataFrame, folds: pd.DataFrame, task: FoldTask
) -> dict[str, list[JsonObject]]:
    records: dict[str, list[JsonObject]] = {
        name: [] for name in ("predictions", "interval_audit", "shap_values", "fit_log")
    }
    for row in folds.to_dict("records"):
        identity = ForecastIdentity(
            task.identity.direction,
            task.identity.hs2,
            task.identity.horizon,
            str(row["forecast_origin"]),
            str(row["forecast_period"]),
        )
        current = FoldTask(identity, task.is_total, task.parameters, task.calibrator)
        try:
            result, local = _compute_fold(series, current)
        except Exception:
            logger.exception("Outer fit failed at %s", identity)
            raise
        records["predictions"].append(build_prediction_row(result))
        records["interval_audit"].append(build_interval_row(result))
        records["shap_values"].extend(local)
        records["fit_log"].append(build_fit_row(result))
    return records


def _run_outer_cv(
    features: pd.DataFrame, folds: pd.DataFrame, parameters: dict[int, JsonObject]
) -> dict[str, pd.DataFrame]:
    series_list = _load_frozen_series()
    records: dict[str, list[JsonObject]] = {
        name: [] for name in ("predictions", "interval_audit", "shap_values", "fit_log")
    }
    for horizon in FORECAST_HORIZONS:
        calendar = (
            folds.loc[folds.horizon.eq(horizon)]
            .sort_values("forecast_origin")
            .reset_index(drop=True)
        )
        for identity in series_list:
            series = (
                features.loc[
                    features.direction.eq(identity["direction"]) & features.hs2.eq(identity["hs2"])
                ]
                .sort_values("period")
                .reset_index(drop=True)
            )
            task = _build_fold_task(identity, horizon, parameters[horizon])
            result = _run_series_folds(series, calendar, task)
            for name, rows in result.items():
                records[name].extend(rows)
    return {name: pd.DataFrame(rows) for name, rows in records.items()}


def _build_interval_metrics(valid: pd.DataFrame) -> JsonObject:
    intervals = valid.loc[valid.interval_available]
    return {
        "interval_available_count": len(intervals),
        "coverage_80": float(intervals.covered_80.mean()) if len(intervals) else np.nan,
        "coverage_95": float(intervals.covered_95.mean()) if len(intervals) else np.nan,
        "mean_interval_width_80": float(intervals.interval_width_80.mean())
        if len(intervals)
        else np.nan,
        "mean_interval_width_95": float(intervals.interval_width_95.mean())
        if len(intervals)
        else np.nan,
    }


def _build_group_metrics(
    group: pd.DataFrame, key: tuple[str, str, bool, int], run: tuple[str, str]
) -> JsonObject:
    direction, code, is_total, horizon = key
    run_id, created_at = run
    valid = group.loc[group.prediction.notnull() & group.actual.notnull()]
    return {
        "run_id": run_id,
        "direction": str(direction),
        "hs2": str(code),
        "is_total": bool(is_total),
        "horizon": int(horizon),
        "model_type": MODEL_TYPE,
        "n_validation_origins": int(len(valid)),
        "first_origin": str(valid.forecast_origin.min()) if len(valid) else None,
        "last_origin": str(valid.forecast_origin.max()) if len(valid) else None,
        "smape": float(valid.smape_component.mean()) if len(valid) else np.nan,
        "mae": float(valid.absolute_error.mean()) if len(valid) else np.nan,
        "rmse": float(np.sqrt(valid.squared_error.mean())) if len(valid) else np.nan,
        **_build_interval_metrics(valid),
        "created_at": created_at,
    }


def aggregate_m9_group_metrics(predictions_df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Return identical M9 metrics. Args: predictions, run ID. Raises: schema errors."""
    context = (run_id, datetime.now(UTC).isoformat())
    rows = [
        _build_group_metrics(group, key, context)
        for key, group in predictions_df.groupby(["direction", "hs2", "is_total", "horizon"])
    ]
    return pd.DataFrame(rows).sort_values(["direction", "hs2", "horizon"]).reset_index(drop=True)


def _load_pipeline_inputs(
    paths: PipelinePaths, custom: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame, list[JsonObject]]:
    for name in (
        "selected_sectors_v1_1.yaml",
        "feature_spec_v1_1.yaml",
        "macro_sources_v1_1.yaml",
        "validation_folds_v1_1.yaml",
    ):
        compute_file_sha256(paths.root / "configs" / name)
    feature_path = paths.input_directory / "model_features_development.parquet"
    compute_file_sha256(feature_path)
    features = custom.copy() if custom is not None else pd.read_parquet(feature_path)
    calendar_path = paths.input_directory / "validation_origins.parquet"
    if not calendar_path.is_file():
        raise FileNotFoundError(f"Frozen validation fold artifact is missing: {calendar_path}")
    folds = pd.read_parquet(calendar_path)
    if len(folds) != 281:
        raise ValueError(f"Expected 281 validation folds, got {len(folds)}")
    sectors = yaml.safe_load((paths.root / "configs/selected_sectors_v1_1.yaml").read_text())
    series = extract_modeling_series_list(sectors)
    if len(series) != 14:
        raise ValueError(f"Expected 14 series, got {len(series)}")
    return features, folds, series


def _get_artifact_paths(paths: PipelinePaths) -> dict[str, Path]:
    names = {
        "tuning_results": "tuning_results",
        "predictions": "predictions",
        "metrics": "metrics",
        "interval_audit": "interval_audit",
        "shap_values": "shap_values",
        "shap_global": "shap_global",
        "fit_log": "fit_log",
    }
    return {
        "xgboost_spec_manifest": paths.specification_path,
        **{
            key: paths.output_directory / f"m9_xgboost_{suffix}.parquet"
            for key, suffix in names.items()
        },
    }


def _build_input_metadata(paths: PipelinePaths) -> JsonObject:
    configuration = {
        "selected_sectors_manifest": "selected_sectors_v1_1.yaml",
        "feature_spec_manifest": "feature_spec_v1_1.yaml",
        "macro_sources_manifest": "macro_sources_v1_1.yaml",
        "validation_folds_manifest": "validation_folds_v1_1.yaml",
    }
    inputs = {key: paths.root / "configs" / name for key, name in configuration.items()}
    inputs.update(
        {
            "validation_folds_artifact": paths.input_directory / "validation_origins.parquet",
            "model_features_development": paths.input_directory
            / "model_features_development.parquet",
        }
    )
    metadata: JsonObject = {}
    for name, path in inputs.items():
        metadata.update({name: str(path), f"{name}_hash": compute_file_sha256(path)})
    return metadata


def _build_run_identity(paths: PipelinePaths, run_id: str) -> JsonObject:
    git = get_git_provenance(paths.root)
    sources = sorted((paths.root / "src/tr_ru_forecast/models/ml").glob("*.py"))
    sources.append(Path(__file__).with_name("xgboost_template.yaml"))
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git.commit,
        "git_working_tree_dirty": git.is_dirty,
        "git_dirty_paths": git.dirty_paths,
        "source_fingerprint": compute_source_fingerprint(sources),
        "code_version": __version__,
        "specification_version": SPEC_VERSION,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "python_version": platform.python_version(),
        "xgboost_version": xgb.__version__,
        "shap_version": shap.__version__,
        "input_artifacts": _build_input_metadata(paths),
    }


def _build_run_counts(tables: dict[str, pd.DataFrame], folds: pd.DataFrame) -> JsonObject:
    predictions = tables["predictions"]
    return {
        "validation_fold_counts": {
            "total_folds": len(folds),
            **{f"h{h}": len(folds.loc[folds.horizon.eq(h)]) for h in FORECAST_HORIZONS},
        },
        "predictions_count": len(predictions),
        "metric_groups_count": len(tables["metrics"]),
        "interval_enabled_rows": int(predictions.interval_available.sum()),
        "interval_unavailable_rows": int((~predictions.interval_available).sum()),
        "shap_values_rows": len(tables["shap_values"]),
        "shap_global_rows": len(tables["shap_global"]),
        "fit_failures_count": int(tables["fit_log"].status.eq("FAILED").sum()),
    }


def _persist_run(
    paths: PipelinePaths,
    tables: dict[str, pd.DataFrame],
    context: tuple[str, dict[int, JsonObject], pd.DataFrame],
) -> JsonObject:
    run_id, parameters, folds = context
    artifacts = _get_artifact_paths(paths)
    for name, table in tables.items():
        table.to_parquet(artifacts[name], index=False)
    metadata = {
        **_build_run_identity(paths, run_id),
        "output_artifacts": {name: str(path) for name, path in artifacts.items()},
        "run_metadata_path": str(paths.output_directory / "m9_xgboost_run.json"),
        "output_artifact_hashes": {
            name: compute_file_sha256(path) for name, path in artifacts.items()
        },
        **_build_training_policy(parameters),
        **_build_run_counts(tables, folds),
    }
    (paths.output_directory / "m9_xgboost_run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def _run_m9_xgboost_pipeline(
    run_id: str | None = None,
    custom_dev_features_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, JsonObject]:
    settings = get_settings()
    directory = output_dir if output_dir is not None else settings.processed_data_dir
    paths = PipelinePaths(
        settings.project_root,
        settings.processed_data_dir,
        directory,
        directory / "xgboost_spec_v1_1.yaml"
        if output_dir is not None
        else settings.project_root / "configs/xgboost_spec_v1_1.yaml",
    )
    active_run_id = run_id or f"{M9_RUN_ID}_{uuid4().hex[:12]}"
    features, folds, series = _load_pipeline_inputs(paths, custom_dev_features_df)
    tuning, parameters = run_pre_validation_tuning(features, series, FORECAST_HORIZONS)
    directory.mkdir(parents=True, exist_ok=True)
    generate_xgboost_manifest(parameters, paths.specification_path)
    tables = _run_outer_cv(features, folds, parameters)
    tables["tuning_results"] = tuning
    if len(tables["predictions"]) != 3934:
        raise ValueError("Expected exactly 3,934 predictions")
    tables["shap_global"] = compute_global_shap_importance(tables["shap_values"])
    tables["metrics"] = aggregate_m9_group_metrics(tables["predictions"], active_run_id)
    if len(tables["metrics"]) != 42 or len(tables["shap_global"]) != 882:
        raise ValueError("Incomplete M9 metrics or SHAP grid")
    metadata = _persist_run(paths, tables, (active_run_id, parameters, folds))
    return tables["predictions"], tables["metrics"], metadata


def _build_fold_values(
    training: pd.DataFrame, task: FoldTask, result: tuple[float, float, JsonObject]
) -> FoldValues:
    identity = task.identity
    actual, prediction, calibration = result
    fold = FoldValues(
        identity,
        task.is_total,
        str(training.period.min()),
        str(training.period.max()),
        str(pd.Period(training.period.max(), freq="M") + identity.horizon),
        len(training),
        str(task.parameters["candidate_id"]),
        actual,
        prediction,
        abs(actual - prediction),
        (actual - prediction) ** 2,
        compute_smape_single(actual, prediction),
        calibration,
    )
    return fold


def _build_fold_task(identity: JsonObject, horizon: int, parameters: JsonObject) -> FoldTask:
    key = ForecastIdentity(str(identity["direction"]), str(identity["hs2"]), horizon, "", "")
    return FoldTask(
        key,
        bool(identity["is_total"]),
        parameters,
        PrequentialResidualCalibrator(key.direction, key.hs2, horizon),
    )


def _build_training_policy(parameters: dict[int, JsonObject]) -> JsonObject:
    return {
        "predictors": M9_PREDICTORS,
        "predictor_count": len(M9_PREDICTORS),
        "random_seed": 42,
        "fixed_parameters": FIXED_XGB_PARAMS,
        "tuning_grid": {
            "max_depth": [2, 3],
            "min_child_weight": [1, 5],
            "learning_rate": [0.03, 0.10],
            "n_estimators": [150, 300],
        },
        "selected_parameters": {f"h{h}": parameters[h] for h in FORECAST_HORIZONS},
    }


def _load_frozen_series() -> list[JsonObject]:
    sectors = yaml.safe_load(
        (get_settings().project_root / "configs/selected_sectors_v1_1.yaml").read_text()
    )
    series = extract_modeling_series_list(sectors)
    if len(series) != 14:
        raise ValueError("Expected 14 modeling series")
    return series


def run_m9_xgboost_pipeline(
    run_id: str | None = None,
    custom_dev_features_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, JsonObject]:
    """Execute M9 tuning and CV. Args: run ID, features, output. Returns: predictions, metrics, metadata. Raises: pipeline errors."""
    return _run_m9_xgboost_pipeline(run_id, custom_dev_features_df, output_dir)
