"""Milestone M7 SARIMA Pipeline Orchestrator.

Orchestrates:
1. Deterministic SARIMA order selection on the initial 60-month training window (2012-01 to 2016-12).
2. Expanding walk-forward validation across the exact 281 folds from M6 fold registry.
3. Multi-step native point and prediction interval forecasting (h=1, 3, 6; 80% and 95% intervals).
4. Prediction error calculation matching M6 metric definitions.
5. Aggregation of the 42 canonical series x horizon metrics (with coverage and interval widths).
6. Recording of fold convergence/failure logs.
7. Full-development residual diagnostics (2012-01 to 2024-12, Ljung-Box test at lag 12).
8. Serialization of order manifest, audit, predictions, metrics, failures, diagnostics, and run metadata.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.evaluation.metrics import compute_smape_single
from tr_ru_forecast.models.classical.order_selector import run_order_selection
from tr_ru_forecast.models.classical.sarima import (
    SARIMAPredictor,
    fit_full_development_diagnostic,
)

logger = get_logger(__name__)

M7_RUN_ID = "run_m7_sarima_v1_1"
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


def aggregate_sarima_group_metrics(
    pred_df: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    """Aggregate row-level SARIMA predictions into the canonical 42 series x horizon metrics table."""
    now_iso = datetime.now(UTC).isoformat()
    group_cols = ["direction", "hs2", "is_total", "horizon"]
    group_records: list[dict[str, Any]] = []

    # Map horizon to expected origins: h1 -> 96, h3 -> 94, h6 -> 91
    expected_origins_map = {1: 96, 3: 94, 6: 91}

    for (dir_name, hs2_code, is_tot, h), group in pred_df.groupby(group_cols):
        exp_origins = expected_origins_map.get(int(h), len(group))
        valid_mask = group["prediction"].notnull() & np.isfinite(group["prediction"])
        valid_group = group[valid_mask]

        n_valid = int(len(valid_group))
        n_failed = exp_origins - n_valid

        smape_val = (
            float(valid_group["smape_component"].mean()) if not valid_group.empty else np.nan
        )
        mae_val = float(valid_group["absolute_error"].mean()) if not valid_group.empty else np.nan
        rmse_val = (
            float(np.sqrt(valid_group["squared_error"].mean()))
            if not valid_group.empty
            else np.nan
        )

        first_orig = str(group["forecast_origin"].min()) if not group.empty else None
        last_orig = str(group["forecast_origin"].max()) if not group.empty else None

        # Interval diagnostics on valid rows
        cov_80, cov_95 = np.nan, np.nan
        width_80, width_95 = np.nan, np.nan
        if not valid_group.empty:
            act = valid_group["actual_value"]
            l80 = valid_group["lower_80"]
            u80 = valid_group["upper_80"]
            l95 = valid_group["lower_95"]
            u95 = valid_group["upper_95"]

            cov_80 = float(((act >= l80) & (act <= u80)).mean())
            cov_95 = float(((act >= l95) & (act <= u95)).mean())
            width_80 = float((u80 - l80).mean())
            width_95 = float((u95 - l95).mean())

        group_records.append({
            "run_id": run_id,
            "direction": str(dir_name),
            "hs2": str(hs2_code),
            "is_total": bool(is_tot),
            "horizon": int(h),
            "model_type": "sarima",
            "expected_origins": exp_origins,
            "valid_predictions": n_valid,
            "failed_predictions": n_failed,
            "first_origin": first_orig,
            "last_origin": last_orig,
            "smape": smape_val,
            "mae": mae_val,
            "rmse": rmse_val,
            "interval_80_coverage": cov_80,
            "interval_95_coverage": cov_95,
            "mean_interval_80_width": width_80,
            "mean_interval_95_width": width_95,
            "created_at": now_iso,
        })

    metrics_df = pd.DataFrame(group_records)
    metrics_df = metrics_df.sort_values(["direction", "hs2", "horizon"]).reset_index(drop=True)
    return metrics_df


def run_sarima_pipeline(
    run_id: str = M7_RUN_ID,
    force_order_selection: bool = False,
    feature_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    sarima_orders_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Execute end-to-end M7 SARIMA forecasting pipeline."""
    settings = get_settings()

    # Paths
    out_dir = output_dir or settings.processed_data_dir
    dev_features_path = settings.processed_data_dir / "model_features_development.parquet"
    sectors_yaml_path = settings.project_root / "configs" / "selected_sectors_v1_1.yaml"
    feature_spec_path = settings.project_root / "configs" / "feature_spec_v1_1.yaml"
    folds_yaml_path = settings.project_root / "configs" / "validation_folds_v1_1.yaml"
    folds_parquet_path = settings.processed_data_dir / "validation_origins.parquet"
    sarima_orders_yaml_path = sarima_orders_path or (
        settings.project_root / "configs" / "sarima_orders_v1_1.yaml"
    )
    order_search_parquet_path = out_dir / "m7_sarima_order_search.parquet"
    predictions_path = out_dir / "m7_sarima_predictions.parquet"
    metrics_path = out_dir / "m7_sarima_metrics.parquet"
    failures_path = out_dir / "m7_sarima_failures.parquet"
    diagnostics_path = out_dir / "m7_sarima_diagnostics.parquet"
    run_json_path = out_dir / "m7_sarima_run.json"

    if feature_df is not None:
        target_df = feature_df.copy()
    else:
        logger.info("Loading development features from %s", dev_features_path)
        target_df = pd.read_parquet(dev_features_path)

    # 1. Order selection on 2012-01 to 2016-12
    if not sarima_orders_yaml_path.exists() or force_order_selection:
        logger.info("Executing SARIMA order selection on initial 60-month training window")
        sarima_manifest, _ = run_order_selection(
            target_df,
            sectors_manifest_path=sectors_yaml_path,
            output_yaml_path=sarima_orders_yaml_path,
            output_parquet_path=order_search_parquet_path,
        )
    else:
        logger.info("Loading existing frozen SARIMA order manifest from %s", sarima_orders_yaml_path)
        with open(sarima_orders_yaml_path) as f:
            sarima_manifest = yaml.safe_load(f)

    # Check for order selection failure
    orders_by_key = {}
    for entry in sarima_manifest.get("series_orders", []):
        key = (entry["direction"], entry["hs2"])
        if entry.get("status") == "ORDER_SELECTION_FAILED":
            raise RuntimeError(f"Order selection failed for series {key}")
        orders_by_key[key] = entry

    # 2. Validation fold registry
    logger.info("Loading frozen validation fold registry from %s", folds_parquet_path)
    if not folds_parquet_path.exists():
        raise FileNotFoundError(f"Missing required fold registry {folds_parquet_path}")
    folds_df = pd.read_parquet(folds_parquet_path)
    if len(folds_df) != 281:
        raise ValueError(f"Expected 281 validation folds, found {len(folds_df)}")

    # 3. 14 modeling series
    with open(sectors_yaml_path, encoding="utf-8") as f:
        sectors_manifest = yaml.safe_load(f)
    series_list = extract_modeling_series_list(sectors_manifest)
    if len(series_list) != 14:
        raise ValueError(f"Expected 14 series, found {len(series_list)}")

    # Pre-index target data for O(1) lookup
    target_indexed = target_df.set_index(["direction", "hs2", "period"])["target"]

    # 4. Multi-step backtesting
    # We iterate series by series, fitting once per unique forecast origin
    unique_origins = sorted(folds_df["forecast_origin"].unique())
    logger.info(
        "Starting walk-forward SARIMA backtest across %d series and %d unique origins",
        len(series_list),
        len(unique_origins),
    )

    predictions_records: list[dict[str, Any]] = []
    failures_records: list[dict[str, Any]] = []

    # Map fold registry by (origin, horizon)
    folds_by_origin: dict[str, list[dict[str, Any]]] = {}
    for _, fold_row in folds_df.iterrows():
        orig = fold_row["forecast_origin"]
        folds_by_origin.setdefault(orig, []).append(fold_row.to_dict())

    for series_info in series_list:
        direction = series_info["direction"]
        hs2 = series_info["hs2"]
        is_total = series_info["is_total"]
        series_order_spec = orders_by_key[(direction, hs2)]

        order = series_order_spec["order"]
        seasonal_order = series_order_spec["seasonal_order"]
        trend = series_order_spec["trend"]

        predictor = SARIMAPredictor(
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            primary_maxiter=200,
            retry_maxiter=500,
        )

        # Full historical series for this target
        s_target = target_df[
            (target_df["direction"] == direction) & (target_df["hs2"] == hs2)
        ].sort_values("period")

        for orig in unique_origins:
            active_folds = folds_by_origin.get(orig, [])
            if not active_folds:
                continue

            active_horizons = sorted([f["horizon"] for f in active_folds])
            train_series = s_target[s_target["period"] <= orig]
            y_train = train_series["target"].values.astype(float)

            # Fit once per series/origin and extract forecasts for all active horizons
            fit_res = predictor.fit_and_forecast(y_train, active_horizons)
            fit_status = fit_res["fit_status"]
            opt_used = fit_res["optimizer_used"]
            retry_cnt = fit_res["retry_count"]
            converged = fit_res["converged"]
            fail_type = fit_res["failure_type"]

            for fold in active_folds:
                h = fold["horizon"]
                fc_period = fold["forecast_period"]
                f_id = fold["fold_id"]
                tr_start = fold["train_start"]
                tr_end = fold["train_end"]

                fc_h = fit_res["forecasts"].get(h, {})
                pred_val = fc_h.get("prediction", np.nan)
                l80 = fc_h.get("lower_80", np.nan)
                u80 = fc_h.get("upper_80", np.nan)
                l95 = fc_h.get("lower_95", np.nan)
                u95 = fc_h.get("upper_95", np.nan)

                # Look up actual value
                actual_val = target_indexed.get((direction, hs2, fc_period), np.nan)
                if pd.notnull(actual_val):
                    actual_val = float(actual_val)

                # Compute errors
                abs_err = np.nan
                sq_err = np.nan
                smape_comp = np.nan
                if pd.notnull(actual_val) and pd.notnull(pred_val):
                    abs_err = abs(actual_val - pred_val)
                    sq_err = (actual_val - pred_val) ** 2
                    smape_comp = compute_smape_single(actual_val, pred_val)

                rec = {
                    "run_id": run_id,
                    "fold_id": f_id,
                    "direction": direction,
                    "hs2": hs2,
                    "is_total": is_total,
                    "horizon": int(h),
                    "forecast_origin": orig,
                    "forecast_period": fc_period,
                    "train_start": tr_start,
                    "train_end": tr_end,
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "trend": trend,
                    "prediction": pred_val,
                    "lower_80": l80,
                    "upper_80": u80,
                    "lower_95": l95,
                    "upper_95": u95,
                    "actual_value": actual_val,
                    "absolute_error": abs_err,
                    "squared_error": sq_err,
                    "smape_component": smape_comp,
                    "fit_status": fit_status,
                    "optimizer_used": opt_used,
                    "retry_count": int(retry_cnt),
                    "converged": bool(converged),
                    "failure_type": fail_type,
                    "model_type": "sarima",
                }
                predictions_records.append(rec)

                if fit_status != "SUCCESS" or not converged:
                    failures_records.append({
                        "direction": direction,
                        "hs2": hs2,
                        "horizon": int(h),
                        "forecast_origin": orig,
                        "order": str(order),
                        "seasonal_order": str(seasonal_order),
                        "optimizer": opt_used,
                        "retry_count": int(retry_cnt),
                        "failure_type": fail_type,
                        "warning_type": fit_res["warning_type"],
                        "message": fit_res["failure_message"],
                        "timestamp": datetime.now(UTC).isoformat(),
                    })

    # 5. Format and persist predictions parquet
    pred_df = pd.DataFrame(predictions_records)
    pred_cols = [
        "run_id",
        "fold_id",
        "direction",
        "hs2",
        "is_total",
        "horizon",
        "forecast_origin",
        "forecast_period",
        "train_start",
        "train_end",
        "order",
        "seasonal_order",
        "trend",
        "prediction",
        "lower_80",
        "upper_80",
        "lower_95",
        "upper_95",
        "actual_value",
        "absolute_error",
        "squared_error",
        "smape_component",
        "fit_status",
        "optimizer_used",
        "retry_count",
        "converged",
        "failure_type",
        "model_type",
    ]
    pred_df = pred_df[pred_cols]
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(predictions_path, index=False)
    logger.info("Saved %d SARIMA prediction rows to %s", len(pred_df), predictions_path)

    # 6. Aggregate metrics across 42 groups
    metrics_df = aggregate_sarima_group_metrics(pred_df, run_id=run_id)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_parquet(metrics_path, index=False)
    logger.info("Saved %d SARIMA metric groups to %s", len(metrics_df), metrics_path)

    # 7. Failures parquet
    failure_cols = [
        "direction",
        "hs2",
        "horizon",
        "forecast_origin",
        "order",
        "seasonal_order",
        "optimizer",
        "retry_count",
        "failure_type",
        "warning_type",
        "message",
        "timestamp",
    ]
    if failures_records:
        failures_df = pd.DataFrame(failures_records)[failure_cols]
    else:
        failures_df = pd.DataFrame(columns=failure_cols)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_df.to_parquet(failures_path, index=False)
    logger.info("Saved %d fold failures to %s", len(failures_df), failures_path)

    # 8. Full-development residual diagnostics (2012-01 to 2024-12, 156 obs)
    logger.info("Fitting full-development residual diagnostics for all 14 series")
    diagnostic_records: list[dict[str, Any]] = []
    for series_info in series_list:
        direction = series_info["direction"]
        hs2 = series_info["hs2"]
        series_order_spec = orders_by_key[(direction, hs2)]

        s_target = target_df[
            (target_df["direction"] == direction) & (target_df["hs2"] == hs2)
        ].sort_values("period")
        y_full = s_target["target"].values.astype(float)

        diag_rec = fit_full_development_diagnostic(
            y_full,
            direction=direction,
            hs2=hs2,
            order=series_order_spec["order"],
            seasonal_order=series_order_spec["seasonal_order"],
            trend=series_order_spec["trend"],
            ljung_box_lag=12,
        )
        diagnostic_records.append(diag_rec)

    diag_cols = [
        "direction",
        "hs2",
        "order",
        "seasonal_order",
        "trend",
        "nobs",
        "converged",
        "aic",
        "bic",
        "residual_mean",
        "residual_std",
        "ljung_box_lag",
        "ljung_box_stat",
        "ljung_box_pvalue",
        "warning_count",
    ]
    diagnostics_df = pd.DataFrame(diagnostic_records)[diag_cols]
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_df.to_parquet(diagnostics_path, index=False)
    logger.info("Saved %d diagnostics rows to %s", len(diagnostics_df), diagnostics_path)

    # 9. Reproducibility metadata JSON
    run_meta: dict[str, Any] = {
        "run_id": run_id,
        "spec_version": SPEC_VERSION,
        "model_version": MODEL_VERSION,
        "model_type": "sarima",
        "code_version": "1.1.0",
        "data_snapshot_id": f"dev_features_{compute_file_sha256(dev_features_path)[:12]}",
        "selected_sector_manifest_hash": compute_file_sha256(sectors_yaml_path),
        "feature_manifest_hash": compute_file_sha256(feature_spec_path),
        "validation_fold_manifest_hash": compute_file_sha256(folds_yaml_path),
        "sarima_order_manifest_hash": compute_file_sha256(sarima_orders_yaml_path),
        "order_selection_window": {
            "start": "2012-01",
            "end": "2016-12",
        },
        "search_grid": sarima_manifest.get("search_grid", {}),
        "selection_criterion": sarima_manifest.get("selection_criterion", {}),
        "optimizer_policy": sarima_manifest.get("optimizer_policy", {}),
        "development_start": "2012-01",
        "development_end": "2024-12",
        "horizons": [1, 3, 6],
        "validation_origins": {
            "total_folds": len(folds_df),
            "h1_count": int((folds_df["horizon"] == 1).sum()),
            "h3_count": int((folds_df["horizon"] == 3).sum()),
            "h6_count": int((folds_df["horizon"] == 6).sum()),
        },
        "predictions_summary": {
            "total_prediction_rows": len(pred_df),
            "valid_predictions": int(pred_df["prediction"].notnull().sum()),
            "failed_predictions": int(pred_df["prediction"].isnull().sum()),
            "negative_point_forecasts": int((pred_df["prediction"] < 0).sum()),
            "negative_lower_80": int((pred_df["lower_80"] < 0).sum()),
            "negative_lower_95": int((pred_df["lower_95"] < 0).sum()),
        },
        "metrics_artifact": (
            str(metrics_path.relative_to(settings.project_root))
            if metrics_path.is_relative_to(settings.project_root)
            else str(metrics_path)
        ),
        "prediction_artifact": (
            str(predictions_path.relative_to(settings.project_root))
            if predictions_path.is_relative_to(settings.project_root)
            else str(predictions_path)
        ),
        "diagnostics_artifact": (
            str(diagnostics_path.relative_to(settings.project_root))
            if diagnostics_path.is_relative_to(settings.project_root)
            else str(diagnostics_path)
        ),
        "failure_artifact": (
            str(failures_path.relative_to(settings.project_root))
            if failures_path.is_relative_to(settings.project_root)
            else str(failures_path)
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "random_seed": None,
    }

    run_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(run_json_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    logger.info("Saved reproducibility run metadata to %s", run_json_path)

    return pred_df, metrics_df, failures_df, diagnostics_df, run_meta
