"""Milestone M8 SARIMAX Pipeline Orchestrator.

Orchestrates:
1. Strict reuse of frozen M7 SARIMA structural orders from configs/sarima_orders_v1_1.yaml.
2. Consumption of the exact five publication-safe exogenous variables:
   usd_try_lagged, rub_usd_lagged, brent_lagged, tr_ip_lagged, ru_ip_lagged.
3. Strict enforcement of the origin_known_persistence future-exogenous strategy:
   origin-known 5-dimensional vector held constant across all forecast steps (h=1,3,6).
4. Compilation of an auditable origin-level exogenous provenance artifact.
5. Deterministic missing-exog handling: fail closed loudly without data fabrication or fallbacks.
6. Expanding walk-forward validation across the exact 281 folds from M6/M7 fold registry.
7. Native multi-step point and prediction interval forecasting (h=1, 3, 6; 80% and 95% intervals).
8. Prediction error calculation matching M6/M7 metric definitions (sMAPE, MAE, RMSE).
9. Aggregation of the 42 canonical series x horizon metrics (with coverage and interval widths).
10. Full-development residual diagnostics (2012-01 to 2024-12, Ljung-Box test at lag 12).
11. Serialization of predictions, metrics, diagnostics, fit logs, audit, and run metadata.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import yaml

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.data.macro.publication import PublicationSafeAccessor
from tr_ru_forecast.data.macro.registry import REQUIRED_INDICATOR_IDS
from tr_ru_forecast.evaluation.metrics import compute_smape_single
from tr_ru_forecast.features.macro_aligner import MACRO_FEATURE_COLUMN_MAP
from tr_ru_forecast.models.classical.pipeline import extract_modeling_series_list
from tr_ru_forecast.models.classical.sarimax import (
    SARIMAXPredictor,
    fit_full_development_sarimax_diagnostic,
)

logger = get_logger(__name__)

M8_RUN_ID: Final[str] = "run_m8_sarimax_v1_1"
SPEC_VERSION: Final[str] = "1.1"
MODEL_VERSION: Final[str] = "1.1.0"
FUTURE_EXOG_STRATEGY: Final[str] = "origin_known_persistence"

EXOG_COLUMNS: Final[list[str]] = [
    "usd_try_lagged",
    "rub_usd_lagged",
    "brent_lagged",
    "tr_ip_lagged",
    "ru_ip_lagged",
]


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum for a file."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_exog_provenance_audit(
    macro_df: pd.DataFrame,
    unique_origins: list[str],
    dev_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build origin-level macroeconomic provenance and publication-safety audit artifact.

    For each validation origin and each of the 5 macro indicators, records:
    - forecast origin period
    - canonical origin snapshot date
    - indicator id
    - feature column name
    - value used
    - underlying macro reference period
    - availability date or approximation
    - release-lag rule
    - future-exog strategy = origin_known_persistence
    - is_publication_safe (availability <= snapshot date)
    """
    records: list[dict[str, Any]] = []
    now_iso = datetime.now(UTC).isoformat()

    # Pre-map features by (period, feature_col) and extract stored feature_snapshot_date if present
    cols_to_extract = ["period"] + EXOG_COLUMNS
    has_stored_snapshot = "feature_snapshot_date" in dev_features_df.columns
    if has_stored_snapshot:
        cols_to_extract.append("feature_snapshot_date")
    feat_subset = dev_features_df[cols_to_extract].drop_duplicates("period")
    feat_indexed = feat_subset.set_index("period")

    for orig in sorted(unique_origins):
        if (
            has_stored_snapshot
            and orig in feat_indexed.index
            and pd.notnull(feat_indexed.loc[orig, "feature_snapshot_date"])
        ):
            snapshot_date = str(feat_indexed.loc[orig, "feature_snapshot_date"])
        else:
            orig_p = pd.Period(orig, freq="M")
            snapshot_date = str(orig_p.end_time.date())

        safe_macro = PublicationSafeAccessor.filter_as_of(
            macro_df,
            forecast_origin=snapshot_date,
        )

        for ind_id in sorted(REQUIRED_INDICATOR_IDS):
            feat_col = MACRO_FEATURE_COLUMN_MAP[ind_id]
            ind_sub = safe_macro[safe_macro["indicator_id"] == ind_id]

            val_used = np.nan
            if orig in feat_indexed.index and feat_col in feat_indexed.columns:
                val_used = float(feat_indexed.loc[orig, feat_col])

            ref_period = None
            avail_date = None
            release_rule = None
            is_pub_safe = False

            if not ind_sub.empty:
                latest_row = ind_sub.sort_values("period").iloc[-1]
                ref_period = str(latest_row["period"])
                avail_date = str(latest_row["availability_date_or_approximation"])
                release_rule = str(latest_row["release_lag_rule"])
                is_pub_safe = bool(avail_date <= snapshot_date)

            records.append({
                "forecast_origin": orig,
                "canonical_origin_snapshot_date": snapshot_date,
                "indicator_id": ind_id,
                "feature_column": feat_col,
                "value_used": val_used,
                "underlying_macro_reference_period": ref_period,
                "availability_date_or_approximation": avail_date,
                "release_lag_rule": release_rule,
                "future_exog_strategy": FUTURE_EXOG_STRATEGY,
                "is_publication_safe": is_pub_safe,
                "created_at": now_iso,
            })

    audit_df = pd.DataFrame(records)
    audit_df = audit_df.sort_values(["forecast_origin", "indicator_id"]).reset_index(drop=True)
    return audit_df


def aggregate_sarimax_group_metrics(
    pred_df: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    """Aggregate row-level SARIMAX predictions into the canonical 42 series x horizon metrics table."""
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
            "model_type": "sarimax",
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


def run_sarimax_pipeline(
    run_id: str = M8_RUN_ID,
    feature_df: pd.DataFrame | None = None,
    macro_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Execute end-to-end Milestone M8 SARIMAX forecasting pipeline.

    Returns:
        tuple of (predictions_df, metrics_df, diagnostics_df, fit_log_df, exog_audit_df, run_meta)
    """
    settings = get_settings()

    # Paths
    out_dir = output_dir or settings.processed_data_dir
    dev_features_path = settings.processed_data_dir / "model_features_development.parquet"
    macro_path = settings.processed_data_dir / "macro_monthly.parquet"
    sectors_yaml_path = settings.project_root / "configs" / "selected_sectors_v1_1.yaml"
    feature_spec_path = settings.project_root / "configs" / "feature_spec_v1_1.yaml"
    macro_sources_path = settings.project_root / "configs" / "macro_sources_v1_1.yaml"
    folds_yaml_path = settings.project_root / "configs" / "validation_folds_v1_1.yaml"
    folds_parquet_path = settings.processed_data_dir / "validation_origins.parquet"
    sarima_orders_yaml_path = settings.project_root / "configs" / "sarima_orders_v1_1.yaml"
    sarimax_spec_yaml_path = settings.project_root / "configs" / "sarimax_spec_v1_1.yaml"

    predictions_path = out_dir / "m8_sarimax_predictions.parquet"
    metrics_path = out_dir / "m8_sarimax_metrics.parquet"
    fit_log_path = out_dir / "m8_sarimax_fit_log.parquet"
    diagnostics_path = out_dir / "m8_sarimax_residual_diagnostics.parquet"
    exog_audit_path = out_dir / "m8_sarimax_exog_audit.parquet"
    run_json_path = out_dir / "m8_sarimax_run.json"

    # 1. Load target and macro data
    if feature_df is not None:
        target_df = feature_df.copy()
    else:
        logger.info("Loading development features from %s", dev_features_path)
        target_df = pd.read_parquet(dev_features_path)

    if macro_df is not None:
        macro_data = macro_df.copy()
    else:
        logger.info("Loading macroeconomic data from %s", macro_path)
        macro_data = pd.read_parquet(macro_path)

    # 2. Load frozen SARIMA orders (Strict structural order reuse)
    logger.info("Loading frozen SARIMA orders from %s", sarima_orders_yaml_path)
    if not sarima_orders_yaml_path.exists():
        raise FileNotFoundError(f"Missing required M7 order manifest {sarima_orders_yaml_path}")
    with open(sarima_orders_yaml_path) as f:
        sarima_manifest = yaml.safe_load(f)

    orders_by_key = {}
    for entry in sarima_manifest.get("series_orders", []):
        key = (entry["direction"], entry["hs2"])
        if entry.get("status") == "ORDER_SELECTION_FAILED":
            raise RuntimeError(f"Cannot reuse failed order for series {key}")
        orders_by_key[key] = entry

    # 3. Load frozen validation folds
    logger.info("Loading frozen validation fold registry from %s", folds_parquet_path)
    if not folds_parquet_path.exists():
        raise FileNotFoundError(f"Missing required fold registry {folds_parquet_path}")
    folds_df = pd.read_parquet(folds_parquet_path)
    if len(folds_df) != 281:
        raise ValueError(f"Expected 281 validation folds, found {len(folds_df)}")

    # 4. Extract 14 modeling series
    with open(sectors_yaml_path, encoding="utf-8") as f:
        sectors_manifest = yaml.safe_load(f)
    series_list = extract_modeling_series_list(sectors_manifest)
    if len(series_list) != 14:
        raise ValueError(f"Expected 14 series, found {len(series_list)}")

    # Verify that all 14 series have frozen orders
    for s_info in series_list:
        k = (s_info["direction"], s_info["hs2"])
        if k not in orders_by_key:
            raise KeyError(f"Series {k} missing from frozen SARIMA order manifest.")

    # 5. Build Exogenous Provenance Audit Artifact
    unique_origins = sorted(folds_df["forecast_origin"].unique())
    logger.info("Building exogenous provenance audit artifact across %d origins", len(unique_origins))
    exog_audit_df = build_exog_provenance_audit(macro_data, unique_origins, target_df)

    # Validate publication-safety invariant across all audit rows
    leakage_violations = exog_audit_df[~exog_audit_df["is_publication_safe"]]
    if not leakage_violations.empty:
        raise ValueError(
            f"CRITICAL LEAKAGE: {len(leakage_violations)} audit records violate publication safety "
            f"(availability_date > origin_snapshot_date)."
        )

    # 6. Verify missing-exog policy on development features
    # Check that the 5 exog columns exist in target_df
    for col in EXOG_COLUMNS:
        if col not in target_df.columns:
            raise ValueError(f"Required exogenous column '{col}' missing from feature dataset.")

    # Pre-index target data for O(1) lookup
    target_indexed = target_df.set_index(["direction", "hs2", "period"])["target"]

    # 7. Walk-forward SARIMAX backtesting
    # Iterate series by series, fitting once per unique forecast origin
    logger.info(
        "Starting walk-forward SARIMAX backtest across %d series and %d unique origins",
        len(series_list),
        len(unique_origins),
    )

    predictions_records: list[dict[str, Any]] = []
    fit_log_records: list[dict[str, Any]] = []

    # Map fold registry by origin
    folds_by_origin: dict[str, list[dict[str, Any]]] = {}
    for _, fold_row in folds_df.iterrows():
        orig = fold_row["forecast_origin"]
        folds_by_origin.setdefault(orig, []).append(fold_row.to_dict())

    now_iso = datetime.now(UTC).isoformat()

    for series_info in series_list:
        direction = series_info["direction"]
        hs2 = series_info["hs2"]
        is_total = series_info["is_total"]
        series_order_spec = orders_by_key[(direction, hs2)]

        order = series_order_spec["order"]
        seasonal_order = series_order_spec["seasonal_order"]
        trend = series_order_spec["trend"]

        predictor = SARIMAXPredictor(
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            primary_maxiter=200,
            retry_maxiter=500,
        )

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
            X_train = train_series[EXOG_COLUMNS].values.astype(float)

            # Check missing-exog policy: fail closed loudly without data fabrication
            missing_exog_detected = False
            missing_exog_reason = ""
            if not np.isfinite(X_train).all():
                missing_exog_detected = True
                missing_exog_reason = "Historical X_train contains non-finite/NaN values."

            # Origin-known persistence vector
            x_origin = None
            if not missing_exog_detected:
                origin_row = s_target[s_target["period"] == orig]
                if origin_row.empty:
                    missing_exog_detected = True
                    missing_exog_reason = f"No observation found at forecast origin {orig}."
                else:
                    x_origin = origin_row[EXOG_COLUMNS].values[0].astype(float)
                    if not np.isfinite(x_origin).all():
                        missing_exog_detected = True
                        missing_exog_reason = f"Origin-known vector at {orig} contains non-finite/NaN values."

            if missing_exog_detected:
                logger.warning(
                    "Missing-exog encountered for %s-%s at origin %s: %s",
                    direction,
                    hs2,
                    orig,
                    missing_exog_reason,
                )
                fit_log_records.append({
                    "run_id": run_id,
                    "direction": direction,
                    "hs2": hs2,
                    "is_total": is_total,
                    "forecast_origin": orig,
                    "fit_status": "EXOG_UNAVAILABLE",
                    "optimizer_used": "NONE",
                    "retry_count": 0,
                    "retry_reason": "",
                    "converged": False,
                    "failure_type": "MISSING_EXOG",
                    "failure_message": missing_exog_reason,
                    "warning_type": "NONE",
                    "created_at": now_iso,
                })
                # Populate NaNs for all active folds
                for fold in active_folds:
                    h = fold["horizon"]
                    fc_period = fold["forecast_period"]
                    actual_val = target_indexed.get((direction, hs2, fc_period), np.nan)
                    predictions_records.append({
                        "run_id": run_id,
                        "fold_id": fold["fold_id"],
                        "direction": direction,
                        "hs2": hs2,
                        "is_total": is_total,
                        "horizon": h,
                        "forecast_origin": orig,
                        "forecast_period": fc_period,
                        "train_start": fold["train_start"],
                        "train_end": fold["train_end"],
                        "model_type": "sarimax",
                        "exog_strategy": FUTURE_EXOG_STRATEGY,
                        "prediction": np.nan,
                        "actual_value": float(actual_val) if pd.notnull(actual_val) else np.nan,
                        "absolute_error": np.nan,
                        "squared_error": np.nan,
                        "smape_component": np.nan,
                        "lower_80": np.nan,
                        "upper_80": np.nan,
                        "lower_95": np.nan,
                        "upper_95": np.nan,
                        "fit_status": "EXOG_UNAVAILABLE",
                        "optimizer_used": "NONE",
                        "retry_count": 0,
                        "converged": False,
                        "warning_type": "NONE",
                        "failure_type": "MISSING_EXOG",
                        "failure_message": missing_exog_reason,
                        "created_at": now_iso,
                    })
                continue

            # Build future persistence exogenous matrix
            max_h = max(active_horizons)
            X_future = np.tile(x_origin, (max_h, 1))

            # Fit SARIMAX and forecast
            fit_res = predictor.fit_and_forecast(y_train, X_train, X_future, active_horizons)
            fit_status = fit_res["fit_status"]
            opt_used = fit_res["optimizer_used"]
            retry_cnt = fit_res["retry_count"]
            converged = fit_res["converged"]
            fail_type = fit_res["failure_type"]
            fail_msg = fit_res["failure_message"]
            warn_type = fit_res["warning_type"]

            fit_log_records.append({
                "run_id": run_id,
                "direction": direction,
                "hs2": hs2,
                "is_total": is_total,
                "forecast_origin": orig,
                "fit_status": fit_status,
                "optimizer_used": opt_used,
                "retry_count": retry_cnt,
                "retry_reason": str(fit_res.get("retry_reason", "")),
                "converged": converged,
                "failure_type": fail_type,
                "failure_message": fail_msg,
                "warning_type": warn_type,
                "created_at": now_iso,
            })

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

                predictions_records.append({
                    "run_id": run_id,
                    "fold_id": f_id,
                    "direction": direction,
                    "hs2": hs2,
                    "is_total": is_total,
                    "horizon": h,
                    "forecast_origin": orig,
                    "forecast_period": fc_period,
                    "train_start": tr_start,
                    "train_end": tr_end,
                    "model_type": "sarimax",
                    "exog_strategy": FUTURE_EXOG_STRATEGY,
                    "prediction": pred_val,
                    "actual_value": actual_val,
                    "absolute_error": abs_err,
                    "squared_error": sq_err,
                    "smape_component": smape_comp,
                    "lower_80": l80,
                    "upper_80": u80,
                    "lower_95": l95,
                    "upper_95": u95,
                    "fit_status": fit_status,
                    "optimizer_used": opt_used,
                    "retry_count": retry_cnt,
                    "converged": converged,
                    "warning_type": warn_type,
                    "failure_type": fail_type,
                    "failure_message": fail_msg,
                    "created_at": now_iso,
                })

    predictions_df = pd.DataFrame(predictions_records)
    predictions_df = predictions_df.sort_values(["direction", "hs2", "horizon", "forecast_origin"]).reset_index(drop=True)

    fit_log_df = pd.DataFrame(fit_log_records)
    fit_log_df = fit_log_df.sort_values(["direction", "hs2", "forecast_origin"]).reset_index(drop=True)

    # 8. Full-Development Residual Diagnostics (2012-01 to 2024-12)
    logger.info("Computing full-development SARIMAX residual diagnostics across 14 series")
    diag_records: list[dict[str, Any]] = []
    for series_info in series_list:
        direction = series_info["direction"]
        hs2 = series_info["hs2"]
        series_order_spec = orders_by_key[(direction, hs2)]

        s_target = target_df[
            (target_df["direction"] == direction) & (target_df["hs2"] == hs2)
        ].sort_values("period")

        y_full = s_target["target"].values.astype(float)
        X_full = s_target[EXOG_COLUMNS].values.astype(float)

        rec = fit_full_development_sarimax_diagnostic(
            y_full=y_full,
            X_full=X_full,
            direction=direction,
            hs2=hs2,
            order=series_order_spec["order"],
            seasonal_order=series_order_spec["seasonal_order"],
            trend=series_order_spec["trend"],
            exog_names=EXOG_COLUMNS,
            ljung_box_lag=12,
        )
        rec["run_id"] = run_id
        rec["created_at"] = now_iso
        diag_records.append(rec)

    diagnostics_df = pd.DataFrame(diag_records)
    diagnostics_df = diagnostics_df.sort_values(["direction", "hs2"]).reset_index(drop=True)

    # 9. Aggregate 42 series x horizon metrics
    metrics_df = aggregate_sarimax_group_metrics(predictions_df, run_id)

    # 10. Run metadata
    run_meta = {
        "run_id": run_id,
        "spec_version": SPEC_VERSION,
        "model_version": MODEL_VERSION,
        "model_type": "sarimax",
        "exogenous_features": EXOG_COLUMNS,
        "future_exog_strategy": FUTURE_EXOG_STRATEGY,
        "total_folds_evaluated": len(predictions_df),
        "total_unique_fits": len(fit_log_df),
        "successful_fits": int((fit_log_df["fit_status"] == "SUCCESS").sum()),
        "failed_fits": int((fit_log_df["fit_status"] != "SUCCESS").sum()),
        "converged_fits": int(fit_log_df["converged"].sum()),
        "valid_predictions": int(predictions_df["prediction"].notnull().sum()),
        "failed_predictions": int(predictions_df["prediction"].isnull().sum()),
        "metric_groups_count": len(metrics_df),
        "selected_sector_manifest_hash": compute_file_sha256(sectors_yaml_path),
        "feature_manifest_hash": compute_file_sha256(feature_spec_path),
        "macro_sources_manifest_hash": compute_file_sha256(macro_sources_path),
        "validation_fold_manifest_hash": compute_file_sha256(folds_yaml_path),
        "sarima_order_manifest_hash": compute_file_sha256(sarima_orders_yaml_path),
        "data_snapshot_id": f"dev_features_{compute_file_sha256(dev_features_path)[:12]}",
        "created_at": now_iso,
    }

    # 11. Serialize artifacts
    if save_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving SARIMAX predictions to %s", predictions_path)
        predictions_df.to_parquet(predictions_path, index=False)

        logger.info("Saving SARIMAX metrics to %s", metrics_path)
        metrics_df.to_parquet(metrics_path, index=False)

        logger.info("Saving SARIMAX diagnostics to %s", diagnostics_path)
        diagnostics_df.to_parquet(diagnostics_path, index=False)

        logger.info("Saving SARIMAX fit log to %s", fit_log_path)
        fit_log_df.to_parquet(fit_log_path, index=False)

        logger.info("Saving exogenous audit to %s", exog_audit_path)
        exog_audit_df.to_parquet(exog_audit_path, index=False)

        logger.info("Saving run metadata to %s", run_json_path)
        with open(run_json_path, "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)

        # Generate M8 methodological configuration manifest
        logger.info("Saving M8 configuration manifest to %s", sarimax_spec_yaml_path)
        spec_content = {
            "spec_version": SPEC_VERSION,
            "sarimax_version": MODEL_VERSION,
            "created_at": now_iso,
            "model_family": "sarimax",
            "structural_order_policy": {
                "source": "configs/sarima_orders_v1_1.yaml",
                "rule": "reuse_m7_sarima_orders_without_reselection",
                "trend_policy": "trend='c' if d==0 and D==0 else 'n'",
                "search_grid_permitted": False,
            },
            "exogenous_feature_policy": {
                "feature_count": 5,
                "variables": EXOG_COLUMNS,
                "source_dataset": "data/processed/model_features_development.parquet",
                "publication_safe_accessor_verified": True,
                "transformations_permitted": "none",
                "feature_selection_permitted": False,
            },
            "future_exogenous_strategy": {
                "strategy_name": FUTURE_EXOG_STRATEGY,
                "description": "5-dimensional vector of publication-safe macro features knowable at origin snapshot held constant across all forecast horizons (h=1,3,6).",
                "future_macro_actuals_permitted": False,
                "macro_forecasting_models_permitted": False,
            },
            "missing_exog_policy": {
                "rule": "fail_closed_loudly_never_fabricate",
                "interpolation_permitted": False,
                "zero_substitution_permitted": False,
                "regressor_dropping_permitted": False,
                "fallback_to_univariate_permitted": False,
            },
            "validation_protocol": {
                "fold_registry_source": "configs/validation_folds_v1_1.yaml",
                "validation_strategy": "expanding_walk_forward_rolling_origin",
                "total_folds": 281,
                "horizons": [1, 3, 6],
                "minimum_initial_history_months": 60,
                "development_start": "2012-01",
                "development_end": "2024-12",
                "holdout_period": "2025-01 to 2025-12",
                "holdout_quarantine": "strict_untouched_exclusion",
            },
            "prediction_intervals": {
                "levels": [0.80, 0.95],
                "method": "native_sarimax_forecast_covariance",
                "clipping_permitted": False,
                "conditioning_assumption": "conditioned_on_deterministic_persistence_path_without_macro_trajectory_uncertainty",
            },
            "metrics": {
                "primary": "smape",
                "secondary": "mae",
                "diagnostic": "rmse",
                "champion_selection_deferred_to": "M10",
            },
        }
        with open(sarimax_spec_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(spec_content, f, sort_keys=False, indent=2)

    return predictions_df, metrics_df, diagnostics_df, fit_log_df, exog_audit_df, run_meta
