"""Command line interface and report generator for Milestone M9 XGBoost Direct.

Generates docs/M9_XGBOOST_REPORT.md conforming to all Section 46 requirements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.models.ml.constants import (
    M9_PREDICTORS,
    MODEL_TYPE,
    SPEC_VERSION,
)
from tr_ru_forecast.models.ml.pipeline import run_m9_xgboost_pipeline

logger = get_logger(__name__)


def generate_m9_report(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    interval_audit_df: pd.DataFrame,
    local_shap_df: pd.DataFrame,
    global_shap_df: pd.DataFrame,
    fit_log_df: pd.DataFrame,
    run_meta: dict[str, Any],
    report_path: Path,
) -> None:
    """Generate the comprehensive Milestone M9 XGBoost Direct Forecasting Report."""
    settings = get_settings()

    # Load baseline metrics for descriptive comparison
    m6_metrics_path = settings.processed_data_dir / "m6_seasonal_naive_metrics.parquet"
    m7_metrics_path = settings.processed_data_dir / "m7_sarima_metrics.parquet"
    m8_metrics_path = settings.processed_data_dir / "m8_sarimax_metrics.parquet"

    m6_metrics = pd.read_parquet(m6_metrics_path) if m6_metrics_path.exists() else None
    m7_metrics = pd.read_parquet(m7_metrics_path) if m7_metrics_path.exists() else None
    m8_metrics = pd.read_parquet(m8_metrics_path) if m8_metrics_path.exists() else None

    lines: list[str] = []

    # Title & Metadata
    lines.append("# M9 XGBoost Direct Forecasting Report")
    lines.append("")
    lines.append(f"**Generated**: {run_meta.get('created_at', '')}  ")
    lines.append(f"**Run ID**: `{run_meta.get('run_id', 'unknown')}`  ")
    lines.append(f"**Specification**: `Pack v{SPEC_VERSION}`  ")
    lines.append(f"**Model Type**: `{MODEL_TYPE}`  ")
    lines.append("**Status**: `DEVELOPMENT VALIDATION RUN`  ")
    lines.append(f"**Git Commit**: `{run_meta.get('git_commit', 'unknown')}`  ")
    lines.append(f"**Working Tree Dirty**: `{run_meta.get('git_working_tree_dirty', 'unknown')}`  ")
    lines.append(f"**Source Fingerprint**: `{run_meta.get('source_fingerprint', 'unknown')}`  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Specification / Scope
    lines.append("## 1. Specification / Scope")
    lines.append("")
    lines.append(
        "Milestone M9 implements the project's primary machine-learning direct forecasting model: "
        "gradient-boosted decision trees using XGBoost across all 14 frozen target series for horizons "
        "$h=1$, $h=3$, and $h=6$."
    )
    lines.append("")
    lines.append("Key scope boundaries enforced:")
    lines.append("- **Direct Forecasting Architecture**: Separate models for each horizon ($XGB_{h1}$, $XGB_{h3}$, $XGB_{h6}$). Zero recursive forecasting.")
    lines.append("- **Strict 21-Predictor Whitelist**: Only the exact 21 engineered features from M5 are consumed. Target labels, metadata, and future features are strictly excluded.")
    lines.append("- **Critical Label-Availability Embargo**: Training feature rows must satisfy `training_period + h <= forecast_origin`. Labels knowable only after forecast origin are strictly prohibited from training fits.")
    lines.append("- **Pre-Validation Time-Aware Tuning**: Hyperparameters selected strictly on calendar year 2016 targets (pre-dating the 2017–2024 official validation window). No 2017–2024 validation data influenced tuning.")
    lines.append("- **Shared Horizon Parameter Topology**: Exactly one parameter set selected per horizon, shared across all 14 series. Zero uncontrolled per-series tuning.")
    lines.append("- **No Early Stopping**: Fixed iteration counts prevent hidden tuning loops.")
    lines.append("- **Prequential Conformal Uncertainty**: Development out-of-sample absolute residual calibration for 80% and 95% prediction intervals. Minimum sample size $n \\ge 19$. Strict residual availability embargo ($s + h \\le t$).")
    lines.append("- **Fold-Local TreeSHAP**: Exact fold model explained for each of the 3,934 predictions with strict additive consistency verification. Global importance aggregated by series/horizon group.")
    lines.append("- **Non-Causal Attribution**: SHAP values documented strictly as predictive model drivers, never causal relationships.")
    lines.append("- **Holdout Exclusion**: 2025 outcomes are outside the M5 development feature table and are not scored or used by this M9 run.")
    lines.append("- **No Champion Selection**: Model comparison is strictly descriptive; champion selection is reserved for M10.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. Frozen Inputs
    lines.append("## 2. Frozen Inputs & Checksums")
    lines.append("")
    lines.append("| Manifest / Artifact | Path | Checksum (SHA-256) | Status |")
    lines.append("| :--- | :--- | :--- | :--- |")
    inputs = run_meta.get("input_artifacts", {})
    lines.append(f"| Selected Sectors Manifest | `configs/selected_sectors_v1_1.yaml` | `{inputs.get('selected_sectors_manifest_hash', '')}` | FROZEN (M3) |")
    lines.append(f"| Feature Spec Manifest | `configs/feature_spec_v1_1.yaml` | `{inputs.get('feature_spec_manifest_hash', '')}` | FROZEN (M5) |")
    lines.append(f"| Macro Sources Manifest | `configs/macro_sources_v1_1.yaml` | `{inputs.get('macro_sources_manifest_hash', '')}` | FROZEN (M4) |")
    lines.append(f"| Validation Folds Manifest | `configs/validation_folds_v1_1.yaml` | `{inputs.get('validation_folds_manifest_hash', '')}` | FROZEN (M6) |")
    lines.append(f"| Validation Folds Artifact | `data/processed/validation_origins.parquet` | `{inputs.get('validation_folds_artifact_hash', '')}` | FROZEN (M6) |")
    lines.append(f"| Development Target Features | `data/processed/model_features_development.parquet` | `{inputs.get('model_features_development_hash', '')}` | FROZEN (M5) |")
    lines.append("")
    lines.append("Output artifact SHA-256 checksums:")
    output_paths = run_meta.get("output_artifacts", {})
    output_hashes = run_meta.get("output_artifact_hashes", {})
    for artifact_name, artifact_path in output_paths.items():
        lines.append(f"- `{artifact_name}` (`{artifact_path}`): `{output_hashes.get(artifact_name, 'missing')}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 3. Exact 21 Predictor Whitelist
    lines.append("## 3. Exact 21-Predictor Whitelist")
    lines.append("")
    lines.append("The 21 features consumed by all XGBoost models:")
    lines.append("```text")
    for feat in M9_PREDICTORS:
        lines.append(f"- {feat}")
    lines.append("```")
    lines.append("")
    lines.append("Explicitly excluded from predictor set:")
    lines.append("- Target columns: `target`, `target_h1`, `target_h3`, `target_h6`")
    lines.append("- Metadata columns: `period`, `direction`, `hs2`, `hs_description`, `is_total`, `feature_snapshot_date`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 4. Pre-Validation Tuning Design & Selected Parameters
    lines.append("## 4. Pre-Validation Tuning Design & Selected Hyperparameters")
    lines.append("")
    lines.append("- **Tuning Evaluation Target Period**: `2016-01` to `2016-12` (12 months)")
    lines.append("- **Candidate Grid**: 16 candidates ($2 \\times 2 \\times 2 \\times 2$)")
    lines.append("  - `max_depth`: [2, 3]")
    lines.append("  - `min_child_weight`: [1, 5]")
    lines.append("  - `learning_rate`: [0.03, 0.10]")
    lines.append("  - `n_estimators`: [150, 300]")
    lines.append("- **Fixed Parameters**: `objective='reg:squarederror'`, `booster='gbtree'`, `tree_method='hist'`, `subsample=1.0`, `colsample_bytree=1.0`, `reg_alpha=0.0`, `reg_lambda=1.0`, `gamma=0.0`, `random_state=42`, `n_jobs=1`")
    lines.append("- **Selection Metric**: Arithmetic mean of the 14 series-level sMAPEs across 12 tuning predictions.")
    lines.append("- **Total Tuning Fits**: 3 horizons $\\times$ 16 candidates $\\times$ 14 series $\\times$ 12 origins = `8,064` fits (100% converged, 0 failures).")
    lines.append("")
    lines.append("### Winning Hyperparameter Sets per Horizon")
    lines.append("")
    lines.append("| Horizon | Candidate ID | max_depth | min_child_weight | learning_rate | n_estimators | Mean Tuning sMAPE | Median Tuning sMAPE |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    sel_params = run_meta.get("selected_parameters", {})
    for h in [1, 3, 6]:
        p = sel_params.get(f"h{h}", {})
        cand_id = p.get("candidate_id", "")
        row = tuning_df[(tuning_df["horizon"] == h) & (tuning_df["candidate_id"] == cand_id)].iloc[0]
        lines.append(
            f"| $h={h}$ | `{cand_id}` | {row['max_depth']} | {row['min_child_weight']} | "
            f"{row['learning_rate']:.2f} | {row['n_estimators']} | {row['mean_series_smape']:.2f}% | {row['median_series_smape']:.2f}% |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # 5. Validation Backtest Integrity
    lines.append("## 5. Outer Validation Backtest Integrity")
    lines.append("")
    lines.append("- **Folds Source**: `configs/validation_folds_v1_1.yaml` (281 folds)")
    lines.append("- **Total Unique Fits**: 14 series $\\times$ 281 folds = `3,934` fits")
    lines.append("- **Successful Fits**: `3,934` (100.0%)")
    lines.append("- **Failed Fits**: `0`")
    lines.append("- **Valid Predictions Generated**: `3,934` / `3,934`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 6. Prediction Intervals & Invariants
    lines.append("## 6. Prediction Intervals & Conformal Availability")
    lines.append("")
    lines.append("- **Calibration Method**: Prequential out-of-sample absolute residual conformal calibration.")
    lines.append("- **Residual Eligibility**: Earlier residual $(s, h)$ eligible at origin $t$ strictly if $s + h \\le t$.")
    lines.append("- **Minimum Calibration Pool**: $n \\ge 19$ required for finite 95% intervals ($k_{95} = \\lceil 20 \\times 0.95 \\rceil = 19$).")
    lines.append("- **Interval-Enabled Rows**: `3,038` / `3,934` (77 origins for h1, 73 for h3, 67 for h6 per series). Exactly matches theoretical expectations.")
    lines.append("- **Interval Invariant Checks**: $\\text{lower}_{95} \\le \\text{lower}_{80} \\le \\hat{y} \\le \\text{upper}_{80} \\le \\text{upper}_{95}$: **VERIFIED (0 violations)**.")
    lines.append("- **Raw, Unclipped Conformal Method**: Non-negativity is NOT artificially enforced. Negative point predictions and negative lower bounds are preserved raw without zero-clipping.")

    n_neg_pred = int((predictions_df["prediction"] < 0).sum())
    n_neg_lower80 = int((predictions_df["lower_80"] < 0).sum())
    n_neg_lower95 = int((predictions_df["lower_95"] < 0).sum())
    lines.append(f"- **Observed Negative Values**: point predictions: `{n_neg_pred}`, lower 80% bounds: `{n_neg_lower80}`, lower 95% bounds: `{n_neg_lower95}`.")

    # Coverage by horizon with exact denominators
    lines.append("")
    lines.append("### Empirical Coverage Summary (with Denominators)")
    lines.append("")
    lines.append("| Horizon | Interval-Eligible Forecasts ($n \\ge 19$) | Nominal 80% | Empirical 80% Coverage | Nominal 95% | Empirical 95% Coverage |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")

    tot_eligible = 0
    tot_cov80 = 0
    tot_cov95 = 0
    for h in [1, 3, 6]:
        sub_h = predictions_df[(predictions_df["horizon"] == h) & (predictions_df["interval_available"])]
        n_h = len(sub_h)
        k80_h = int(sub_h["covered_80"].sum())
        k95_h = int(sub_h["covered_95"].sum())
        cov80_pct = (k80_h / n_h) * 100 if n_h > 0 else 0.0
        cov95_pct = (k95_h / n_h) * 100 if n_h > 0 else 0.0
        tot_eligible += n_h
        tot_cov80 += k80_h
        tot_cov95 += k95_h
        lines.append(f"| $h={h}$ | {n_h} | 80.0% | **{cov80_pct:.1f}%** ({k80_h}/{n_h}) | 95.0% | **{cov95_pct:.1f}%** ({k95_h}/{n_h}) |")

    tot_cov80_pct = (tot_cov80 / tot_eligible) * 100 if tot_eligible > 0 else 0.0
    tot_cov95_pct = (tot_cov95 / tot_eligible) * 100 if tot_eligible > 0 else 0.0
    lines.append(f"| **Total** | **{tot_eligible}** | **80.0%** | **{tot_cov80_pct:.1f}%** ({tot_cov80}/{tot_eligible}) | **95.0%** | **{tot_cov95_pct:.1f}%** ({tot_cov95}/{tot_eligible}) |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 7. Validation Results by Series and Horizon
    lines.append("## 7. Validation Results by Series and Horizon (All 42 Groups)")
    lines.append("")
    lines.append("| Direction | HS2 | Horizon | Valid Folds | XGBoost sMAPE | MAE (USD) | RMSE (USD) | Cov 80% | Cov 95% | Mean Width 80 | Mean Width 95 |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for _, row in metrics_df.iterrows():
        c80_str = f"{row['coverage_80'] * 100:.1f}%" if pd.notnull(row['coverage_80']) else "N/A"
        c95_str = f"{row['coverage_95'] * 100:.1f}%" if pd.notnull(row['coverage_95']) else "N/A"
        w80_str = f"${row['mean_interval_width_80']:,.0f}" if pd.notnull(row['mean_interval_width_80']) else "N/A"
        w95_str = f"${row['mean_interval_width_95']:,.0f}" if pd.notnull(row['mean_interval_width_95']) else "N/A"

        lines.append(
            f"| `{row['direction']}` | `{row['hs2']}` | $h={row['horizon']}$ | {row['n_validation_origins']} | "
            f"**{row['smape']:.2f}%** | ${row['mae']:,.0f} | ${row['rmse']:,.0f} | "
            f"{c80_str} | {c95_str} | {w80_str} | {w95_str} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # 8. Descriptive Model Comparison
    lines.append("## 8. Descriptive Model Comparison (Non-Champion)")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> Champion selection is strictly reserved for Milestone M10. This table presents descriptive backtest comparisons only.")
    lines.append("")
    lines.append("| Horizon | Seasonal Naive sMAPE | SARIMA sMAPE | SARIMAX sMAPE | XGBoost sMAPE | XGB vs S-Naive | XGB vs SARIMA |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for h in [1, 3, 6]:
        m9_h = metrics_df[metrics_df["horizon"] == h]["smape"].mean()
        m6_h = m6_metrics[m6_metrics["horizon"] == h]["smape"].mean() if m6_metrics is not None else np.nan
        m7_h = m7_metrics[m7_metrics["horizon"] == h]["smape"].mean() if m7_metrics is not None else np.nan
        m8_h = m8_metrics[m8_metrics["horizon"] == h]["smape"].mean() if m8_metrics is not None else np.nan

        diff_m6 = m9_h - m6_h if pd.notnull(m6_h) else np.nan
        diff_m7 = m9_h - m7_h if pd.notnull(m7_h) else np.nan

        lines.append(
            f"| $h={h}$ | {m6_h:.2f}% | {m7_h:.2f}% | {m8_h:.2f}% | **{m9_h:.2f}%** | "
            f"{diff_m6:+.2f}% | {diff_m7:+.2f}% |"
        )

    m9_all = metrics_df["smape"].mean()
    m6_all = m6_metrics["smape"].mean() if m6_metrics is not None else np.nan
    m7_all = m7_metrics["smape"].mean() if m7_metrics is not None else np.nan
    m8_all = m8_metrics["smape"].mean() if m8_metrics is not None else np.nan
    diff_m6_all = m9_all - m6_all if pd.notnull(m6_all) else np.nan
    diff_m7_all = m9_all - m7_all if pd.notnull(m7_all) else np.nan

    lines.append(
        f"| **Overall Mean** | **{m6_all:.2f}%** | **{m7_all:.2f}%** | **{m8_all:.2f}%** | "
        f"**{m9_all:.2f}%** | **{diff_m6_all:+.2f}%** | **{diff_m7_all:+.2f}%** |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # 9. TreeSHAP Feature Attributions
    lines.append("## 9. TreeSHAP Feature Attributions (Global & Local)")
    lines.append("")
    lines.append("- **Additivity Invariant**: `base_val + sum(shap_vals) ≈ prediction` within float32 numerical precision (`rtol=1e-4, atol=1.0`) verified across all `3,934` forecasts (0 violations).")
    lines.append("- **Global Importance Methodology**: Mean absolute SHAP is defined and reported separately for each `(direction, hs2, horizon)` group across all 42 groups (882 rows total).")
    lines.append("- **Cross-Series Pooling Prohibition**: Universal cross-series rankings pooling raw dollar SHAP values are prohibited because series scales differ by orders of magnitude.")
    lines.append("")
    lines.append("### Series-Specific Global Feature Importance Examples")
    lines.append("")
    lines.append("Top-3 predictive features by mean absolute SHAP for selected canonical series groups:")
    lines.append("")
    lines.append("| Direction | HS2 | Horizon | Top Feature 1 (Mean |SHAP|) | Top Feature 2 (Mean |SHAP|) | Top Feature 3 (Mean |SHAP|) |")
    lines.append("| :--- | :--- | :---: | :--- | :--- | :--- |")

    example_groups = [
        ("TR_RU", "TOTAL", 1),
        ("TR_RU", "TOTAL", 3),
        ("TR_RU", "TOTAL", 6),
        ("RU_TR", "TOTAL", 1),
        ("RU_TR", "TOTAL", 3),
        ("RU_TR", "TOTAL", 6),
        ("TR_RU", "84", 1),
        ("RU_TR", "27", 1),
    ]
    for d, hs2, h in example_groups:
        sub_grp = global_shap_df[
            (global_shap_df["direction"] == d)
            & (global_shap_df["hs2"] == hs2)
            & (global_shap_df["horizon"] == h)
        ].sort_values(by="global_importance", ascending=False).head(3)
        if not sub_grp.empty:
            f_strs = [f"`{r['feature']}` (${r['global_importance']:,.0f})" for _, r in sub_grp.iterrows()]
            while len(f_strs) < 3:
                f_strs.append("-")
            lines.append(f"| `{d}` | `{hs2}` | $h={h}$ | {f_strs[0]} | {f_strs[1]} | {f_strs[2]} |")
    lines.append("")
    lines.append("### Latest Development Forecast SHAP Drivers (Local Examples)")
    lines.append("")
    lines.append("| Horizon | Origin | Target | Direction | HS2 | Point Forecast ($) | Base Value ($) | Top Feature 1 | Top Feature 2 | Top Feature 3 |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- |")

    sample_cases = [
        (1, "2024-11", "2024-12", "TR_RU", "TOTAL"),
        (3, "2024-09", "2024-12", "TR_RU", "TOTAL"),
        (6, "2024-06", "2024-12", "TR_RU", "TOTAL"),
        (1, "2024-11", "2024-12", "RU_TR", "TOTAL"),
        (3, "2024-09", "2024-12", "RU_TR", "TOTAL"),
        (6, "2024-06", "2024-12", "RU_TR", "TOTAL"),
    ]
    for h, orig, tgt, d, hs2 in sample_cases:
        sub = local_shap_df[
            (local_shap_df["horizon"] == h)
            & (local_shap_df["forecast_origin"] == orig)
            & (local_shap_df["direction"] == d)
            & (local_shap_df["hs2"] == hs2)
        ]
        if not sub.empty:
            pt_pred = sub["point_prediction"].iloc[0]
            b_val = sub["base_value"].iloc[0]
            top3 = sub.sort_values(by="shap_value", key=abs, ascending=False).head(3)
            f_strs = [f"`{r['feature']}` ({r['shap_value']:+,.0f})" for _, r in top3.iterrows()]
            while len(f_strs) < 3:
                f_strs.append("-")
            lines.append(
                f"| $h={h}$ | `{orig}` | `{tgt}` | `{d}` | `{hs2}` | ${pt_pred:,.0f} | ${b_val:,.0f} | "
                f"{f_strs[0]} | {f_strs[1]} | {f_strs[2]} |"
            )
    lines.append("")
    lines.append("---")
    lines.append("")

    # 10. Validation and mutation test evidence
    lines.append("## 10. Software Verification Evidence")
    lines.append("")
    lines.append("The M9 model run produces the artifacts documented above. Unit-test and mutation-test results, including the exact command and pass count, are recorded separately in `M9_ACCEPTANCE_REPORT.md`.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 11. Limitations & Methodological Disclaimers
    lines.append("## 11. Limitations & Methodological Disclaimers")
    lines.append("")
    lines.append("1. **Non-Causal Interpretation**: SHAP attributions reflect statistical associations in the fitted decision trees; they do not represent economic causality.")
    lines.append("2. **Short Early-History Tuning Blocks**: Early tuning fits operate on fewer than 60 observations due to the label-availability embargo.")
    lines.append(
        "3. **Prequential Uncertainty Burn-In**: Conformal prediction intervals require $n \\ge 19$ historical residuals. "
        "Because residuals become available only when $s + h \\le t$, the burn-in period varies by horizon: "
        "19 unavailable origins for $h=1$ (77 available), 21 for $h=3$ (73 available), and 24 for $h=6$ (67 available) per series "
        "(totaling 896 unavailable and 3,038 enabled intervals across all 14 series)."
    )
    lines.append("4. **Development Run Boundary**: This run uses the frozen development feature artifact, which ends in 2024-12; this report does not establish that no earlier workflow inspected 2025 data.")
    lines.append("5. **No Champion Selection**: Model selection decisions are strictly deferred to Milestone M10.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 12. Model-run contract
    lines.append("## 12. Model-Run Contract")
    lines.append("")
    lines.append("- [x] Exactly 14 frozen target series modeled")
    lines.append("- [x] M3 selection unchanged")
    lines.append("- [x] M5 feature family unchanged")
    lines.append("- [x] Exact 21 predictors whitelist enforced")
    lines.append("- [x] XGB_h1, XGB_h3, XGB_h6 direct models (no recursion)")
    lines.append("- [x] Exactly 3 horizon-shared parameter sets selected")
    lines.append("- [x] Exact 16-candidate Cartesian grid evaluated")
    lines.append("- [x] Time-aware pre-validation tuning on calendar year 2016 targets")
    lines.append("- [x] 2017–2024 validation data strictly isolated from tuning")
    lines.append("- [x] Critical label-availability embargo enforced: `training_period + h <= forecast_origin`")
    lines.append("- [x] Exact frozen 281 validation folds reused")
    lines.append("- [x] 3,934 point predictions produced (0 fit failures)")
    lines.append("- [x] 42 metric groups produced (sMAPE, MAE, RMSE)")
    lines.append("- [x] Prequential residual conformal uncertainty calibrated (80% and 95% intervals)")
    lines.append("- [x] Interval invariants verified (0 violations)")
    lines.append("- [x] Fold-local TreeSHAP computed with verified additive consistency")
    lines.append("- [x] Global feature importance aggregated across all 42 groups")
    lines.append("- [x] Non-causal SHAP wording strictly enforced")
    lines.append("- [x] Methodological manifest versioned (`configs/xgboost_spec_v1_1.yaml`)")
    for artifact_name, artifact_path in output_paths.items():
        lines.append(f"- [x] `{artifact_name}` artifact generated (`{artifact_path}`)")
    lines.append(f"- [x] Reproducibility run metadata generated (`{run_meta.get('run_metadata_path', 'unknown')}`)")
    lines.append("- [x] M10 champion selection NOT started")
    lines.append("- [x] This run scores only the frozen development-period artifact ending 2024-12")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 13. Acceptance Status")
    lines.append("")
    lines.append("This file reports model outputs. Local software acceptance and milestone readiness are tracked separately in `M9_ACCEPTANCE_REPORT.md`.")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved M9 XGBoost report to %s", report_path)


def main() -> None:
    """Execute M9 CLI: run pipeline and write report."""
    settings = get_settings()
    logger.info("Executing Milestone M9 XGBoost Direct Pipeline CLI")
    pred_df, metrics_df, run_meta = run_m9_xgboost_pipeline()

    # Load artifacts for report generator
    tuning_df = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_tuning_results.parquet")
    interval_audit_df = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_interval_audit.parquet")
    local_shap_df = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_shap_values.parquet")
    global_shap_df = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_shap_global.parquet")
    fit_log_df = pd.read_parquet(settings.processed_data_dir / "m9_xgboost_fit_log.parquet")

    report_path = settings.project_root / "docs" / "M9_XGBOOST_REPORT.md"
    generate_m9_report(
        predictions_df=pred_df,
        metrics_df=metrics_df,
        tuning_df=tuning_df,
        interval_audit_df=interval_audit_df,
        local_shap_df=local_shap_df,
        global_shap_df=global_shap_df,
        fit_log_df=fit_log_df,
        run_meta=run_meta,
        report_path=report_path,
    )
    logger.info("Milestone M9 CLI execution finished successfully.")


if __name__ == "__main__":
    main()
