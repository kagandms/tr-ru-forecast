"""Milestone M6 Seasonal Naive Pipeline Orchestrator.

Orchestrates:
1. Generation and persistence of the shared expanding walk-forward validation fold registry.
2. Batch execution of the Seasonal Naive forecasting baseline across all 14 target series.
3. Row-level prediction error scoring (sMAPE, MAE, squared error).
4. Aggregation of the 42 canonical series x horizon metrics.
5. Serialization of prediction, metric, and reproducibility run artifacts.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.constants import (
    DEVELOPMENT_DATA_END,
    DEVELOPMENT_DATA_START,
    FORECAST_HORIZONS,
)
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.evaluation.metrics import (
    aggregate_group_metrics,
    compute_prediction_error_components,
)
from tr_ru_forecast.evaluation.splits import (
    generate_validation_folds,
    save_validation_registry,
)
from tr_ru_forecast.models.baselines.seasonal_naive import SeasonalNaivePredictor

logger = get_logger(__name__)

M6_RUN_ID = "run_m6_seasonal_naive_v1_1"
SPEC_VERSION = "1.1"
MODEL_VERSION = "1.1.0"


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum for a file."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_modeling_series_list(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract list of 14 modeling series dicts from selected sectors manifest."""
    series_list = []

    # TR_RU
    for item in manifest.get("TR_RU", []):
        series_list.append({
            "direction": "TR_RU",
            "hs2": str(item["hs2"]).zfill(2),
            "description": item.get("description", ""),
            "is_total": False,
        })
    series_list.append({
        "direction": "TR_RU",
        "hs2": "TOTAL",
        "description": "Total All Commodities",
        "is_total": True,
    })

    # RU_TR
    for item in manifest.get("RU_TR", []):
        series_list.append({
            "direction": "RU_TR",
            "hs2": str(item["hs2"]).zfill(2),
            "description": item.get("description", ""),
            "is_total": False,
        })
    series_list.append({
        "direction": "RU_TR",
        "hs2": "TOTAL",
        "description": "Total All Commodities",
        "is_total": True,
    })

    return series_list


def run_seasonal_naive_pipeline(
    run_id: str = M6_RUN_ID,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Execute end-to-end Seasonal Naive backtesting pipeline."""
    settings = get_settings()

    # Paths
    dev_features_path = settings.processed_data_dir / "model_features_development.parquet"
    sectors_yaml_path = settings.project_root / "configs" / "selected_sectors_v1_1.yaml"
    feature_spec_path = settings.project_root / "configs" / "feature_spec_v1_1.yaml"
    folds_yaml_path = settings.project_root / "configs" / "validation_folds_v1_1.yaml"
    folds_parquet_path = settings.processed_data_dir / "validation_origins.parquet"
    predictions_path = settings.processed_data_dir / "m6_seasonal_naive_predictions.parquet"
    metrics_path = settings.processed_data_dir / "m6_seasonal_naive_metrics.parquet"
    run_json_path = settings.processed_data_dir / "m6_seasonal_naive_run.json"

    logger.info("Loading development features from %s", dev_features_path)
    target_df = pd.read_parquet(dev_features_path)

    logger.info("Loading sector manifest from %s", sectors_yaml_path)
    with open(sectors_yaml_path, encoding="utf-8") as f:
        sectors_manifest = yaml.safe_load(f)

    # 1. Generate and save shared validation folds
    logger.info("Generating expanding walk-forward validation folds")
    folds_df = generate_validation_folds(
        start_period=DEVELOPMENT_DATA_START,
        end_period=DEVELOPMENT_DATA_END,
        min_history_months=60,
        horizons=FORECAST_HORIZONS,
    )
    save_validation_registry(folds_df, folds_yaml_path, folds_parquet_path)

    # 2. Extract 14 modeling series
    series_list = extract_modeling_series_list(sectors_manifest)
    if len(series_list) != 14:
        raise ValueError(f"Expected exactly 14 modeling series, got {len(series_list)}")

    # 3. Instantiate Seasonal Naive predictor and run predictions
    logger.info("Instantiating SeasonalNaivePredictor and generating predictions")
    predictor = SeasonalNaivePredictor(target_df)
    pred_raw_df = predictor.predict_batch(folds_df, series_list)

    # Add run_id and error components
    pred_raw_df["run_id"] = run_id
    pred_df = compute_prediction_error_components(pred_raw_df)

    # Arrange column order according to contract
    ordered_pred_cols = [
        "run_id",
        "fold_id",
        "direction",
        "hs2",
        "is_total",
        "horizon",
        "forecast_origin",
        "train_start",
        "train_end",
        "forecast_period",
        "seasonal_source_period",
        "seasonal_source_value",
        "actual_value",
        "prediction",
        "absolute_error",
        "squared_error",
        "smape_component",
        "model_type",
    ]
    pred_df = pred_df[ordered_pred_cols]

    # Verify no 2025 forecast periods exist
    invalid_periods = pred_df[pred_df["forecast_period"] > DEVELOPMENT_DATA_END]
    if not invalid_periods.empty:
        raise ValueError(f"CRITICAL LEAKAGE: Predictions targeting period > {DEVELOPMENT_DATA_END} found!")

    # 4. Aggregate group metrics across 42 groups
    logger.info("Aggregating metrics across 42 series x horizon groups")
    metrics_df = aggregate_group_metrics(pred_df, run_id=run_id, model_type="seasonal_naive")
    if len(metrics_df) != 42:
        raise ValueError(f"Expected 42 metric groups, got {len(metrics_df)}")

    # 5. Persist prediction and metrics artifacts
    logger.info("Saving predictions to %s (%d rows)", predictions_path, len(pred_df))
    pred_df.to_parquet(predictions_path, index=False)

    logger.info("Saving metrics to %s (%d groups)", metrics_path, len(metrics_df))
    metrics_df.to_parquet(metrics_path, index=False)

    # 6. Build reproducibility run metadata
    logger.info("Compiling M6 run reproducibility metadata")
    horizon_summary = {}
    for h, g in metrics_df.groupby("horizon"):
        horizon_summary[f"h{h}"] = {
            "mean_smape": float(g["smape"].mean()),
            "median_smape": float(g["smape"].median()),
            "mean_mae": float(g["mae"].mean()),
            "mean_rmse": float(g["rmse"].mean()),
        }

    run_meta = {
        "run_id": run_id,
        "spec_version": SPEC_VERSION,
        "code_version": MODEL_VERSION,
        "milestone": "M6",
        "model_type": "seasonal_naive",
        "created_at": datetime.now(UTC).isoformat(),
        "random_seed": None,  # Deterministic method; no pseudo-random state
        "hyperparameters": {
            "seasonal_period_months": 12,
            "rule": "Forecast(t+h) = Actual(t+h-12)",
        },
        "development_start": DEVELOPMENT_DATA_START,
        "development_end": DEVELOPMENT_DATA_END,
        "minimum_initial_history_months": 60,
        "first_origin": "2016-12",
        "horizons": FORECAST_HORIZONS,
        "total_prediction_rows": len(pred_df),
        "total_metric_groups": len(metrics_df),
        "portfolio_summary": {
            "overall_median_smape": float(metrics_df["smape"].median()),
            "overall_mean_smape": float(metrics_df["smape"].mean()),
            "horizon_metrics": horizon_summary,
        },
        "artifact_paths": {
            "validation_manifest": str(folds_yaml_path),
            "validation_origins": str(folds_parquet_path),
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "run_meta": str(run_json_path),
        },
        "checksums": {
            "selected_sectors_yaml": compute_file_sha256(sectors_yaml_path),
            "feature_spec_yaml": compute_file_sha256(feature_spec_path),
            "model_features_development": compute_file_sha256(dev_features_path),
            "validation_folds_yaml": compute_file_sha256(folds_yaml_path),
            "validation_origins_parquet": compute_file_sha256(folds_parquet_path),
        },
    }

    with open(run_json_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    logger.info("Saved run metadata to %s", run_json_path)

    return pred_df, metrics_df, run_meta
