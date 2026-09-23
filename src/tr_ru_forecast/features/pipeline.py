"""Feature Engineering Pipeline for Milestone M5.

Orchestrates the creation, validation, and serialization of the canonical development
feature dataset (model_features_development.parquet) and audit artifact (feature_engineering_audit.json).
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.constants import (
    DEVELOPMENT_DATA_END,
    DEVELOPMENT_DATA_START,
    HOLDOUT_DATA_END,
    HOLDOUT_DATA_START,
)
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.data.macro.registry import (
    REQUIRED_INDICATOR_IDS,
)
from tr_ru_forecast.features.builder import (
    REQUIRED_FEATURE_CONTRACT_COLUMNS,
    build_complete_calendar_panel,
    compute_autoregressive_lags,
    compute_backward_rolling_features,
    compute_calendar_features,
    compute_growth_features,
    compute_target_labels,
    enforce_development_target_guard,
)
from tr_ru_forecast.features.macro_aligner import (
    MACRO_FEATURE_COLUMN_MAP,
    align_publication_safe_macro,
)

logger = get_logger(__name__)

FEATURE_VERSION = "1.1.0"
SPEC_VERSION = "1.1"

FINAL_OUTPUT_COLUMNS = [
    "period",
    "direction",
    "hs2",
    "hs_description",
    "is_total",
    "target",
    "target_h1",
    "target_h3",
    "target_h6",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12",
    "rolling_std_3",
    "rolling_std_6",
    "rolling_std_12",
    "mom_growth",
    "yoy_growth",
    "month",
    "quarter",
    "year",
    "usd_try_lagged",
    "rub_usd_lagged",
    "brent_lagged",
    "tr_ip_lagged",
    "ru_ip_lagged",
    "feature_snapshot_date",
]


def load_input_datasets(
    target_path: Path | None = None,
    macro_path: Path | None = None,
    sectors_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load verified target series, macroeconomic data, and sector manifests."""
    settings = get_settings()

    target_file = target_path or (settings.processed_data_dir / "forecast_target_monthly.parquet")
    macro_file = macro_path or (settings.processed_data_dir / "macro_monthly.parquet")
    sectors_file = sectors_path or (settings.project_root / "configs" / "selected_sectors_v1_1.yaml")

    if not target_file.exists():
        raise FileNotFoundError(f"Target dataset not found: {target_file}")
    if not macro_file.exists():
        raise FileNotFoundError(f"Macro dataset not found: {macro_file}")
    if not sectors_file.exists():
        raise FileNotFoundError(f"Selected sectors manifest not found: {sectors_file}")

    logger.info("Loading forecast target dataset from %s", target_file)
    target_df = pd.read_parquet(target_file)

    logger.info("Loading macro dataset from %s", macro_file)
    macro_df = pd.read_parquet(macro_file)

    logger.info("Loading selected sectors manifest from %s", sectors_file)
    with open(sectors_file, encoding="utf-8") as f:
        sectors_manifest = yaml.safe_load(f)

    return target_df, macro_df, sectors_manifest


def build_development_features(
    target_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    sectors_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Execute the complete feature engineering sequence for development period."""
    # 1. Enforce strict development quarantine BEFORE any feature operations
    logger.info("Enforcing development quarantine (period <= %s)", DEVELOPMENT_DATA_END)
    dev_target_df = enforce_development_target_guard(target_df, max_period=DEVELOPMENT_DATA_END)

    # 2. Build complete contiguous monthly calendar panel (14 series * 156 months)
    logger.info("Building contiguous calendar panel for 14 modeling series")
    panel_df = build_complete_calendar_panel(
        dev_target_df,
        sectors_manifest,
        start_period=DEVELOPMENT_DATA_START,
        end_period=DEVELOPMENT_DATA_END,
    )

    # 3. Compute forward target labels with development boundary enforcement
    logger.info("Computing forward target labels (h=1, 3, 6)")
    df = compute_target_labels(panel_df, max_dev_period=DEVELOPMENT_DATA_END)

    # 4. Compute autoregressive lags
    logger.info("Computing autoregressive lags (1, 2, 3, 6, 12)")
    df = compute_autoregressive_lags(df)

    # 5. Compute backward-only rolling statistics
    logger.info("Computing backward-only rolling features (mean, std with ddof=1)")
    df = compute_backward_rolling_features(df, ddof=1)

    # 6. Compute growth features
    logger.info("Computing MoM and YoY growth features")
    df = compute_growth_features(df)

    # 7. Compute calendar features and snapshot dates
    logger.info("Computing calendar features and month-end feature snapshot dates")
    df = compute_calendar_features(df)

    # 8. Align publication-safe macroeconomic features
    logger.info("Aligning publication-safe macro indicators via as-of snapshot lookup")
    df, audit_trace = align_publication_safe_macro(df, macro_df)

    # 9. Verify contract columns and arrange output
    missing_cols = set(REQUIRED_FEATURE_CONTRACT_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Feature table missing required contract columns: {missing_cols}")

    ordered_df = df[FINAL_OUTPUT_COLUMNS].copy()

    # Safety checks
    # Assert no infinite values
    numeric_cols = ordered_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(ordered_df[col]).any():
            raise ValueError(f"Column '{col}' contains infinite values!")

    return ordered_df, audit_trace


def build_audit_metadata(
    feature_df: pd.DataFrame,
    audit_trace: list[dict[str, Any]],
    sectors_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Generate comprehensive machine-readable audit report for M5."""
    settings = get_settings()

    # Per-feature missingness analysis
    feature_stats: dict[str, dict[str, Any]] = {}
    total_rows = len(feature_df)

    for col in FINAL_OUTPUT_COLUMNS:
        null_count = int(feature_df[col].isnull().sum())
        is_numeric = pd.api.types.is_numeric_dtype(feature_df[col])
        is_finite = bool(not np.isinf(feature_df[col].dropna()).any()) if is_numeric else True
        feature_stats[col] = {
            "dtype": str(feature_df[col].dtype),
            "null_count": null_count,
            "null_ratio": round(null_count / total_rows, 6),
            "is_finite_valid": is_finite,
        }

    # Per-horizon label availability
    horizon_stats: dict[str, dict[str, Any]] = {}
    for h in [1, 3, 6]:
        col = f"target_h{h}"
        valid_sub = feature_df[feature_df[col].notnull()]
        horizon_stats[f"h{h}"] = {
            "label_column": col,
            "valid_count": int(len(valid_sub)),
            "missing_count": int(feature_df[col].isnull().sum()),
            "first_labeled_origin": str(valid_sub["period"].min()) if not valid_sub.empty else None,
            "last_labeled_origin": str(valid_sub["period"].max()) if not valid_sub.empty else None,
        }

    # Per-series breakdown
    series_stats: list[dict[str, Any]] = []
    for (dir_name, hs2_code), group in feature_df.groupby(["direction", "hs2"]):
        series_stats.append({
            "direction": dir_name,
            "hs2": hs2_code,
            "description": group["hs_description"].iloc[0],
            "is_total": bool(group["is_total"].iloc[0]),
            "n_rows": len(group),
            "period_min": str(group["period"].min()),
            "period_max": str(group["period"].max()),
            "target_nulls": int(group["target"].isnull().sum()),
            "target_mean_usd": float(group["target"].mean()),
        })

    # Macro coverage breakdown
    macro_trace_df = pd.DataFrame(audit_trace)
    macro_stats: dict[str, dict[str, Any]] = {}

    for ind in REQUIRED_INDICATOR_IDS:
        ind_trace = macro_trace_df[macro_trace_df["indicator_id"] == ind]
        feature_col = MACRO_FEATURE_COLUMN_MAP[ind]
        avail_trace = ind_trace[ind_trace["is_available"]]

        macro_stats[ind] = {
            "feature_column": feature_col,
            "available_feature_count": int(len(avail_trace)),
            "missing_feature_count": int(len(ind_trace) - len(avail_trace)),
            "earliest_macro_reference_used": str(avail_trace["selected_macro_reference_period"].min()) if not avail_trace.empty else None,
            "latest_macro_reference_used": str(avail_trace["selected_macro_reference_period"].max()) if not avail_trace.empty else None,
            "earliest_availability_date_used": str(avail_trace["selected_macro_availability_date"].min()) if not avail_trace.empty else None,
            "latest_availability_date_used": str(avail_trace["selected_macro_availability_date"].max()) if not avail_trace.empty else None,
        }

    # First valid period for lags and rolling features
    feature_first_valid: dict[str, str] = {}
    for col in [
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
        "rolling_std_3", "rolling_std_6", "rolling_std_12",
        "mom_growth", "yoy_growth",
    ]:
        valid_periods = feature_df[feature_df[col].notnull()]["period"]
        feature_first_valid[col] = str(valid_periods.min()) if not valid_periods.empty else "NEVER"

    audit_payload = {
        "metadata": {
            "spec_version": SPEC_VERSION,
            "feature_version": FEATURE_VERSION,
            "milestone": "M5",
            "created_at": datetime.now(UTC).isoformat(),
            "target_policy": "Specification v1.1 official Türkiye-reported bilateral series",
            "development_period": {
                "start": DEVELOPMENT_DATA_START,
                "end": DEVELOPMENT_DATA_END,
                "n_months": 156,
            },
            "holdout_period": {
                "start": HOLDOUT_DATA_START,
                "end": HOLDOUT_DATA_END,
                "n_months": 12,
                "status": "UNTOUCHED_ISOLATED",
            },
            "total_modeling_series": 14,
            "total_rows": total_rows,
            "unique_keys_count": len(feature_df[["direction", "hs2", "period"]].drop_duplicates()),
            "duplicate_keys_count": int(feature_df.duplicated(subset=["direction", "hs2", "period"]).sum()),
            "output_parquet_path": str(settings.processed_data_dir / "model_features_development.parquet"),
        },
        "selected_sectors": {
            "TR_RU": [s["hs2"] for s in sectors_manifest.get("TR_RU", [])],
            "RU_TR": [s["hs2"] for s in sectors_manifest.get("RU_TR", [])],
        },
        "feature_statistics": feature_stats,
        "feature_first_valid_period": feature_first_valid,
        "horizon_label_availability": horizon_stats,
        "series_summary": series_stats,
        "macro_alignment_summary": macro_stats,
        "macro_alignment_trace_sample": audit_trace[:15],  # First 15 entries for quick inspection
    }

    return audit_payload


def run_feature_pipeline() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute end-to-end M5 Feature Engineering pipeline and persist artifacts."""
    settings = get_settings()

    target_df, macro_df, sectors_manifest = load_input_datasets()

    logger.info("Executing M5 feature engineering builder...")
    feature_df, audit_trace = build_development_features(target_df, macro_df, sectors_manifest)

    logger.info("Compiling M5 audit report...")
    audit_data = build_audit_metadata(feature_df, audit_trace, sectors_manifest)

    # Persist parquet
    output_parquet = settings.processed_data_dir / "model_features_development.parquet"
    logger.info("Saving development features to %s (%d rows)", output_parquet, len(feature_df))
    feature_df.to_parquet(output_parquet, index=False)

    # Persist audit JSON
    output_audit = settings.processed_data_dir / "feature_engineering_audit.json"
    logger.info("Saving feature audit artifact to %s", output_audit)
    with open(output_audit, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    logger.info("Milestone M5 Feature Engineering pipeline completed successfully.")
    return feature_df, audit_data
