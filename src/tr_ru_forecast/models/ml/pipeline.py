"""Milestone M9 XGBoost Direct Pipeline Orchestrator.

Orchestrates:
1. Pre-validation time-aware hyperparameter tuning (calendar year 2016 target period).
2. Selection and freezing of 3 horizon-shared parameter sets (one each for h1, h3, h6).
3. Expanding walk-forward backtest across the exact 281 folds from M6/M7/M8 registry (3,934 predictions).
4. Critical training label-availability embargo: training_period + h <= forecast_origin.
5. Prequential residual conformal uncertainty calibration (80% and 95% intervals, n >= 19).
6. Local TreeSHAP attributions with strict additivity verification for all 3,934 forecasts (82,614 rows).
7. Aggregation of global feature importance per 42 series/horizon groups (882 rows).
8. Prediction error evaluation (sMAPE, MAE, RMSE) and 42 group metrics with interval coverage.
9. Serialization of all 8 data/config artifacts and reproducibility metadata.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
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
    MIN_CALIBRATION_SIZE,
    MODEL_TYPE,
    MODEL_VERSION,
    SPEC_VERSION,
)
from tr_ru_forecast.models.ml.provenance import (
    compute_source_fingerprint,
    get_git_provenance,
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


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum for a file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_xgboost_manifest(
    selected_by_horizon: dict[int, dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Generate and persist the authoritative configs/xgboost_spec_v1_1.yaml manifest."""
    manifest = {
        "spec_version": SPEC_VERSION,
        "xgboost_version": MODEL_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "model_family": "xgboost",
        "strategy": "direct",
        "horizons": FORECAST_HORIZONS,
        "predictor_columns": M9_PREDICTORS,
        "random_seed": 42,
        "fixed_parameters": FIXED_XGB_PARAMS,
        "tuning": {
            "topology": "one_selected_parameter_set_per_horizon_shared_across_series",
            "tuning_block_target_period": "2016-01 to 2016-12",
            "tuning_origins": {
                "h1": "2015-12 to 2016-11 (12 origins)",
                "h3": "2015-10 to 2016-09 (12 origins)",
                "h6": "2015-07 to 2016-06 (12 origins)",
            },
            "candidate_grid": {
                "max_depth": [2, 3],
                "min_child_weight": [1, 5],
                "learning_rate": [0.03, 0.10],
                "n_estimators": [150, 300],
            },
            "candidate_count_per_horizon": 16,
            "total_candidate_evaluations": 48,
            "selection_metric": "arithmetic_mean_of_14_series_smape",
            "tie_breaks": [
                "lowest_median_of_14_series_smape",
                "candidate_id_ascending",
            ],
            "selected_parameters": {
                f"h{h}": {
                    "candidate_id": selected_by_horizon[h]["candidate_id"],
                    "max_depth": selected_by_horizon[h]["max_depth"],
                    "min_child_weight": selected_by_horizon[h]["min_child_weight"],
                    "learning_rate": selected_by_horizon[h]["learning_rate"],
                    "n_estimators": selected_by_horizon[h]["n_estimators"],
                }
                for h in FORECAST_HORIZONS
            },
        },
        "training_label_availability_rule": "training_period + horizon <= forecast_origin",
        "outer_validation_registry": {
            "source": "configs/validation_folds_v1_1.yaml",
            "total_folds": 281,
            "horizons": {
                "h1": {"origins": 96, "range": "2016-12 to 2024-11"},
                "h3": {"origins": 94, "range": "2016-12 to 2024-09"},
                "h6": {"origins": 91, "range": "2016-12 to 2024-06"},
            },
        },
        "missing_value_policy": "native_xgboost_nan_handling_no_imputation_no_scaling",
        "early_stopping": False,
        "interval": {
            "method": "prequential_conformal_absolute_residual",
            "series_specific": True,
            "horizon_specific": True,
            "prequential": True,
            "residual_eligibility_rule": "residual_target_period <= forecast_origin",
            "levels": [0.80, 0.95],
            "minimum_calibration_size": MIN_CALIBRATION_SIZE,
            "order_statistic_rule": "ceil((n + 1) * (1 - alpha))",
            "expected_interval_origins_per_series": {
                "h1": 77,
                "h3": 73,
                "h6": 67,
            },
            "expected_interval_enabled_rows": 3038,
        },
        "shap": {
            "explainer": "shap.TreeExplainer",
            "model_output": "raw",
            "feature_perturbation": "tree_path_dependent",
            "additivity_tolerance": "rtol=1e-4, atol=1.0",
            "global_semantics": "mean_abs_shap_per_42_groups",
            "local_semantics": "long_form_3934_rows_by_21_features",
        },
        "holdout_policy": "strict_2025_quarantine_no_tuning_no_features_no_calibration",
        "failure_policy": "fail_explicitly_no_hidden_fallbacks",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    logger.info("Saved M9 XGBoost specification manifest to %s", output_path)
    return manifest


def aggregate_m9_group_metrics(
    predictions_df: pd.DataFrame,
    run_id: str = M9_RUN_ID,
) -> pd.DataFrame:
    """Aggregate row-level predictions into the canonical 42 series x horizon metrics table with interval coverage."""
    records: list[dict[str, Any]] = []
    now_iso = datetime.now(UTC).isoformat()
    group_cols = ["direction", "hs2", "is_total", "horizon"]

    for (dir_name, hs2_code, is_tot, h), group in predictions_df.groupby(group_cols):
        valid_group = group[group["prediction"].notnull() & group["actual"].notnull()]

        smape_val = float(valid_group["smape_component"].mean()) if not valid_group.empty else np.nan
        mae_val = float(valid_group["absolute_error"].mean()) if not valid_group.empty else np.nan
        rmse_val = float(np.sqrt(valid_group["squared_error"].mean())) if not valid_group.empty else np.nan

        first_orig = str(valid_group["forecast_origin"].min()) if not valid_group.empty else None
        last_orig = str(valid_group["forecast_origin"].max()) if not valid_group.empty else None

        # Interval metrics computed over interval-enabled rows only
        interval_rows = valid_group[valid_group["interval_available"]]
        n_interval = len(interval_rows)

        if n_interval > 0:
            cov_80 = float(interval_rows["covered_80"].mean())
            cov_95 = float(interval_rows["covered_95"].mean())
            mean_w80 = float(interval_rows["interval_width_80"].mean())
            mean_w95 = float(interval_rows["interval_width_95"].mean())
        else:
            cov_80 = np.nan
            cov_95 = np.nan
            mean_w80 = np.nan
            mean_w95 = np.nan

        records.append({
            "run_id": run_id,
            "direction": str(dir_name),
            "hs2": str(hs2_code),
            "is_total": bool(is_tot),
            "horizon": int(h),
            "model_type": MODEL_TYPE,
            "n_validation_origins": int(len(valid_group)),
            "first_origin": first_orig,
            "last_origin": last_orig,
            "smape": smape_val,
            "mae": mae_val,
            "rmse": rmse_val,
            "interval_available_count": n_interval,
            "coverage_80": cov_80,
            "coverage_95": cov_95,
            "mean_interval_width_80": mean_w80,
            "mean_interval_width_95": mean_w95,
            "created_at": now_iso,
        })

    metrics_df = pd.DataFrame(records)
    metrics_df = metrics_df.sort_values(by=["direction", "hs2", "horizon"]).reset_index(drop=True)
    return metrics_df


def run_m9_xgboost_pipeline(
    run_id: str | None = None,
    custom_dev_features_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Execute end-to-end M9 XGBoost Direct forecasting pipeline.

    Returns:
        (predictions_df, metrics_df, run_metadata)
    """
    settings = get_settings()
    active_run_id = run_id or f"{M9_RUN_ID}_{uuid4().hex[:12]}"
    data_dir = output_dir if output_dir is not None else settings.processed_data_dir
    config_dir = settings.project_root / "configs"

    # Input paths
    sectors_yaml_path = config_dir / "selected_sectors_v1_1.yaml"
    feature_spec_path = config_dir / "feature_spec_v1_1.yaml"
    macro_sources_path = config_dir / "macro_sources_v1_1.yaml"
    folds_yaml_path = config_dir / "validation_folds_v1_1.yaml"
    folds_parquet_path = settings.processed_data_dir / "validation_origins.parquet"

    # Output paths
    m9_spec_yaml_path = (
        data_dir / "xgboost_spec_v1_1.yaml"
        if output_dir is not None
        else config_dir / "xgboost_spec_v1_1.yaml"
    )
    tuning_parquet_path = data_dir / "m9_xgboost_tuning_results.parquet"
    predictions_parquet_path = data_dir / "m9_xgboost_predictions.parquet"
    metrics_parquet_path = data_dir / "m9_xgboost_metrics.parquet"
    interval_audit_parquet_path = data_dir / "m9_xgboost_interval_audit.parquet"
    shap_values_parquet_path = data_dir / "m9_xgboost_shap_values.parquet"
    shap_global_parquet_path = data_dir / "m9_xgboost_shap_global.parquet"
    fit_log_parquet_path = data_dir / "m9_xgboost_fit_log.parquet"
    run_json_path = data_dir / "m9_xgboost_run.json"

    # 1. Load inputs
    canonical_dev_features_path = settings.processed_data_dir / "model_features_development.parquet"
    required_inputs = [
        sectors_yaml_path,
        feature_spec_path,
        macro_sources_path,
        folds_yaml_path,
        canonical_dev_features_path,
    ]
    missing_inputs = [path for path in required_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing required M9 inputs: {missing_inputs}")
    if custom_dev_features_df is not None:
        dev_features_df = custom_dev_features_df.copy()
    else:
        logger.info("Loading development features from %s", canonical_dev_features_path)
        dev_features_df = pd.read_parquet(canonical_dev_features_path)

    with open(sectors_yaml_path, encoding="utf-8") as f:
        sectors_manifest = yaml.safe_load(f)
    series_list = extract_modeling_series_list(sectors_manifest)
    if len(series_list) != 14:
        raise ValueError(f"Expected 14 series, got {len(series_list)}")

    # Load outer validation folds
    if not folds_parquet_path.is_file():
        raise FileNotFoundError(f"Frozen validation fold artifact is missing: {folds_parquet_path}")
    folds_df = pd.read_parquet(folds_parquet_path)


    if len(folds_df) != 281:
        raise ValueError(f"Expected 281 validation folds, got {len(folds_df)}")

    # 2. Stage 1: Pre-validation Time-Aware Tuning Block
    logger.info("Executing Stage 1: Pre-validation time-aware tuning block")
    tuning_df, selected_by_horizon = run_pre_validation_tuning(
        dev_features_df=dev_features_df,
        series_list=series_list,
        horizons=FORECAST_HORIZONS,
    )

    # Persist tuning results
    tuning_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tuning_df.to_parquet(tuning_parquet_path, index=False)
    logger.info("Saved tuning results to %s (%d rows)", tuning_parquet_path, len(tuning_df))

    # Persist M9 specification manifest
    generate_xgboost_manifest(selected_by_horizon, m9_spec_yaml_path)

    # 3. Stage 2: Outer Expanding Walk-Forward Backtesting
    logger.info("Executing Stage 2: Outer expanding walk-forward backtest (281 folds x 14 series)")
    predictions_records: list[dict[str, Any]] = []
    interval_audit_records: list[dict[str, Any]] = []
    local_shap_records: list[dict[str, Any]] = []
    fit_log_records: list[dict[str, Any]] = []

    # Execute per horizon and per series to maintain sequential prequential calibration
    for h in FORECAST_HORIZONS:
        h_folds = folds_df[folds_df["horizon"] == h].sort_values("forecast_origin").reset_index(drop=True)
        h_params = selected_by_horizon[h]
        cand_id = h_params["candidate_id"]

        for s_info in series_list:
            dir_name = s_info["direction"]
            hs2_code = s_info["hs2"]
            is_tot = s_info["is_total"]


            series_df = dev_features_df[
                (dev_features_df["direction"] == dir_name)
                & (dev_features_df["hs2"] == hs2_code)
            ].sort_values("period").reset_index(drop=True)

            calibrator = PrequentialResidualCalibrator(
                direction=dir_name,
                hs2=hs2_code,
                horizon=h,
            )

            for _, fold in h_folds.iterrows():
                f_origin = str(fold["forecast_origin"])

                f_period = str(fold["forecast_period"])

                try:
                    # A. Enforce label-availability embargo
                    train_sub = get_training_subset(series_df, f_origin, h)
                    if train_sub.empty:
                        raise ValueError(f"No training rows available for origin {f_origin}, h={h}")

                    train_feat_start = str(train_sub["period"].min())
                    train_feat_end = str(train_sub["period"].max())
                    train_label_end = str(pd.Period(train_feat_end, freq="M") + h)
                    n_train_rows = len(train_sub)

                    X_train, y_train = prepare_training_xy(train_sub, h)
                    X_origin = extract_origin_features(series_df, f_origin)

                    # B. Fit model
                    model = build_xgb_regressor(h_params)
                    model.fit(X_train, y_train)

                    # C. Predict
                    y_pred = float(model.predict(X_origin)[0])

                    # D. Actual value
                    label_col = f"target_h{h}"
                    actual_series_rows = series_df[series_df["period"] == f_origin]
                    actual_val = float(actual_series_rows[label_col].iloc[0])

                    abs_err = abs(actual_val - y_pred)
                    sq_err = (actual_val - y_pred) ** 2
                    smape_comp = compute_smape_single(actual_val, y_pred)

                    # E. Prequential conformal calibration
                    cal_dict = calibrator.calibrate(f_origin, y_pred, actual_val)

                    # F. Record in calibrator for future origins
                    calibrator.record_historical_prediction(f_origin, f_period, actual_val, y_pred)

                    # G. TreeSHAP attribution
                    base_val, feat_shaps, fold_local_shaps = explain_fold_prediction(
                        model=model,
                        X_origin=X_origin,
                        point_prediction=y_pred,
                        direction=dir_name,
                        hs2=hs2_code,
                        horizon=h,
                        forecast_origin=f_origin,
                        forecast_period=f_period,
                    )
                    local_shap_records.extend(fold_local_shaps)

                    # Append prediction record
                    predictions_records.append({
                        "direction": dir_name,
                        "hs2": hs2_code,
                        "is_total": is_tot,
                        "horizon": h,
                        "forecast_origin": f_origin,
                        "forecast_period": f_period,
                        "train_start": train_feat_start,
                        "train_feature_end": train_feat_end,
                        "train_label_end": train_label_end,
                        "training_row_count": n_train_rows,
                        "model_type": MODEL_TYPE,
                        "parameter_set_id": cand_id,
                        "actual": actual_val,
                        "prediction": y_pred,
                        "absolute_error": abs_err,
                        "squared_error": sq_err,
                        "smape_component": smape_comp,
                        "interval_available": cal_dict["interval_available"],
                        "calibration_count": cal_dict["calibration_count"],
                        "lower_80": cal_dict["lower_80"],
                        "upper_80": cal_dict["upper_80"],
                        "lower_95": cal_dict["lower_95"],
                        "upper_95": cal_dict["upper_95"],
                        "interval_width_80": cal_dict["interval_width_80"],
                        "interval_width_95": cal_dict["interval_width_95"],
                        "covered_80": cal_dict["covered_80"],
                        "covered_95": cal_dict["covered_95"],
                    })

                    # Append interval audit record
                    interval_audit_records.append({
                        "direction": dir_name,
                        "hs2": hs2_code,
                        "horizon": h,
                        "forecast_origin": f_origin,
                        "forecast_period": f_period,
                        "interval_available": cal_dict["interval_available"],
                        "calibration_count": cal_dict["calibration_count"],
                        "max_calibration_origin": cal_dict["max_calibration_origin"],
                        "max_calibration_target_period": cal_dict["max_calibration_target_period"],
                        "q80": cal_dict["q80"],
                        "q95": cal_dict["q95"],
                        "lower_80": cal_dict["lower_80"],
                        "upper_80": cal_dict["upper_80"],
                        "lower_95": cal_dict["lower_95"],
                        "upper_95": cal_dict["upper_95"],
                        "actual": actual_val,
                        "prediction": y_pred,
                        "covered_80": cal_dict["covered_80"],
                        "covered_95": cal_dict["covered_95"],
                    })

                    # Log successful fit
                    fit_log_records.append({
                        "stage": "outer_validation",
                        "direction": dir_name,
                        "hs2": hs2_code,
                        "horizon": h,
                        "origin": f_origin,
                        "candidate_id": cand_id,
                        "status": "SUCCESS",
                        "exception_type": None,
                        "exception_message": None,
                    })

                except Exception as e:
                    logger.error(
                        "Outer fit failed for (%s, %s, h=%d) at %s: %s",
                        dir_name, hs2_code, h, f_origin, e,
                    )
                    fit_log_records.append({
                        "stage": "outer_validation",
                        "direction": dir_name,
                        "hs2": hs2_code,
                        "horizon": h,
                        "origin": f_origin,
                        "candidate_id": cand_id,
                        "status": "FAILED",
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                    })
                    # Fail loudly per specification
                    raise

    # 4. Convert to DataFrames
    predictions_df = pd.DataFrame(predictions_records)
    if len(predictions_df) != 3934:
        raise ValueError(f"Expected exactly 3,934 predictions, got {len(predictions_df)}")

    interval_audit_df = pd.DataFrame(interval_audit_records)
    local_shap_df = pd.DataFrame(local_shap_records)
    fit_log_df = pd.DataFrame(fit_log_records)

    # 5. Compute global SHAP importance
    logger.info("Computing global feature importance across 42 groups")
    global_shap_df = compute_global_shap_importance(local_shap_df)
    if len(global_shap_df) != 882:
        raise ValueError(f"Expected 882 global SHAP rows (42 groups x 21 features), got {len(global_shap_df)}")

    # 6. Aggregate 42 group metrics
    logger.info("Aggregating 42 series x horizon group metrics")
    metrics_df = aggregate_m9_group_metrics(predictions_df, run_id=active_run_id)
    if len(metrics_df) != 42:
        raise ValueError(f"Expected 42 metric groups, got {len(metrics_df)}")

    # 7. Persist all parquet artifacts
    predictions_df.to_parquet(predictions_parquet_path, index=False)
    metrics_df.to_parquet(metrics_parquet_path, index=False)
    interval_audit_df.to_parquet(interval_audit_parquet_path, index=False)
    local_shap_df.to_parquet(shap_values_parquet_path, index=False)
    global_shap_df.to_parquet(shap_global_parquet_path, index=False)
    fit_log_df.to_parquet(fit_log_parquet_path, index=False)

    logger.info("Successfully persisted all M9 Parquet artifacts.")

    # 8. Compile run metadata JSON
    git_provenance = get_git_provenance(settings.project_root)
    source_paths = sorted((settings.project_root / "src" / "tr_ru_forecast" / "models" / "ml").glob("*.py"))
    output_artifact_paths = {
        "xgboost_spec_manifest": m9_spec_yaml_path,
        "tuning_results": tuning_parquet_path,
        "predictions": predictions_parquet_path,
        "metrics": metrics_parquet_path,
        "interval_audit": interval_audit_parquet_path,
        "shap_values": shap_values_parquet_path,
        "shap_global": shap_global_parquet_path,
        "fit_log": fit_log_parquet_path,
    }
    run_meta = {
        "run_id": active_run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_provenance.commit,
        "git_working_tree_dirty": git_provenance.is_dirty,
        "git_dirty_paths": git_provenance.dirty_paths,
        "source_fingerprint": compute_source_fingerprint(source_paths),
        "code_version": __version__,
        "specification_version": SPEC_VERSION,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "python_version": platform.python_version(),
        "xgboost_version": xgb.__version__,
        "shap_version": shap.__version__,
        "input_artifacts": {
            "selected_sectors_manifest": str(sectors_yaml_path),
            "selected_sectors_manifest_hash": compute_file_sha256(sectors_yaml_path),
            "feature_spec_manifest": str(feature_spec_path),
            "feature_spec_manifest_hash": compute_file_sha256(feature_spec_path),
            "macro_sources_manifest": str(macro_sources_path),
            "macro_sources_manifest_hash": compute_file_sha256(macro_sources_path),
            "validation_folds_manifest": str(folds_yaml_path),
            "validation_folds_manifest_hash": compute_file_sha256(folds_yaml_path),
            "validation_folds_artifact": str(folds_parquet_path),
            "validation_folds_artifact_hash": compute_file_sha256(folds_parquet_path),
            "model_features_development": str(canonical_dev_features_path),
            "model_features_development_hash": compute_file_sha256(canonical_dev_features_path),
        },
        "output_artifacts": {name: str(path) for name, path in output_artifact_paths.items()},
        "run_metadata_path": str(run_json_path),
        "output_artifact_hashes": {
            name: compute_file_sha256(path) for name, path in output_artifact_paths.items()
        },
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
        "selected_parameters": {
            f"h{h}": {
                "candidate_id": selected_by_horizon[h]["candidate_id"],
                "max_depth": selected_by_horizon[h]["max_depth"],
                "min_child_weight": selected_by_horizon[h]["min_child_weight"],
                "learning_rate": selected_by_horizon[h]["learning_rate"],
                "n_estimators": selected_by_horizon[h]["n_estimators"],
            }
            for h in FORECAST_HORIZONS
        },
        "validation_fold_counts": {
            "total_folds": len(folds_df),
            "h1": len(folds_df[folds_df["horizon"] == 1]),
            "h3": len(folds_df[folds_df["horizon"] == 3]),
            "h6": len(folds_df[folds_df["horizon"] == 6]),
        },
        "predictions_count": len(predictions_df),
        "metric_groups_count": len(metrics_df),
        "interval_enabled_rows": int(predictions_df["interval_available"].sum()),
        "interval_unavailable_rows": int((~predictions_df["interval_available"]).sum()),
        "shap_values_rows": len(local_shap_df),
        "shap_global_rows": len(global_shap_df),
        "fit_failures_count": int((fit_log_df["status"] == "FAILED").sum()),
    }

    with open(run_json_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    logger.info("Saved M9 run metadata to %s", run_json_path)

    return predictions_df, metrics_df, run_meta
