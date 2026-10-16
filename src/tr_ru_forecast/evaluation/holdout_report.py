"""Holdout tables and scientific reporting, including failure and availability denominators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tr_ru_forecast.evaluation.holdout_runner import compute_holdout_metrics
from tr_ru_forecast.models.ml.records import JsonObject


def render_table(frame: pd.DataFrame) -> str:
    """Render a dependency-free Markdown table. Args: frame. Returns: Markdown. Raises: none."""

    def _format_cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}" if np.isfinite(value) else "unavailable"
        return str(value)

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines += [
        "| " + " | ".join(_format_cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def build_global_holdout_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return role/horizon diagnostics with explicit denominators. Args: predictions. Raises: schema errors."""
    rows = []
    for horizon in (1, 3, 6, 0):
        selected = predictions if horizon == 0 else predictions.loc[predictions.horizon.eq(horizon)]
        for role, group in selected.groupby("role"):
            rows.append(
                {
                    "horizon": "overall" if horizon == 0 else str(horizon),
                    "role": role,
                    **compute_holdout_metrics(group),
                }
            )
    return pd.DataFrame(rows)


def build_shap_summary(local: pd.DataFrame) -> pd.DataFrame:
    """Return per-task feature importance and signed attribution. Args: local SHAP. Raises: schema errors."""
    frame = local.assign(absolute_shap=local.shap_value.abs())
    keys = ["sector_id", "horizon", "model_family", "feature"]
    return (
        frame.groupby(keys)
        .agg(
            mean_abs_shap=("absolute_shap", "mean"),
            mean_signed_shap=("shap_value", "mean"),
            n_origins=("shap_value", "size"),
        )
        .reset_index()
        .sort_values(keys)
        .reset_index(drop=True)
    )


def build_holdout_report(
    leaderboard: pd.DataFrame, predictions: pd.DataFrame, metadata: JsonObject
) -> str:
    """Generate M11 report. Args: leaderboard, predictions, metadata. Returns: Markdown. Raises: schema errors."""
    sections = [
        "# M11 Final Holdout Evaluation Report",
        "",
        f"**Generated:** {metadata['created_at']}",
        f"**Run ID:** `{metadata['run_id']}`",
        "**Status:** Final retrospective holdout evaluation; assignments unchanged.",
        "",
        "## 1. Executive Summary & Acceptance Scope",
        "",
        _build_scope_text(),
        "",
        *_build_comparison_sections(leaderboard, predictions),
        "## 5. Generalization, Drift and Economic Interpretation",
        "",
        _build_interpretation(leaderboard),
        "",
        "## 6. Reproducibility and Provenance",
        "",
        _build_provenance_text(metadata),
    ]
    return "\n".join(sections) + "\n"


def _build_scope_text() -> str:
    return (
        "**Retrospective latest-vintage holdout evaluation under deterministic lag rules.** "
        "All 42 tasks cover twelve 2025 targets with origins target-month minus h1/h3/h6, giving 504 task/origin cells and 1,512 role rows. "
        "Frozen champion/fallback identities, orders, parameters, feature families and interval levels are reused without tuning. "
        "Trade values are assumed available at reference-month end (zero added lag), consistent with the authorized retrospective convention. "
        "Latest-vintage macro values are joined as-of each historical feature-row month-end using the frozen approximate release dates. "
        "Historical publication timestamps, original vintages and real-time actionable performance are not established.\n\n"
        "Early h3/h6 origins precede the 2024-12 model-selection freeze and are retrospective reconstructions using subsequently frozen assignments; "
        "they are not ex-ante deployments of those assignments. Rolling holdout refits and chronological residual updates were amended explicitly before scoring.\n\n"
        "XGBoost calibration admits only same-series/horizon/model out-of-sample residuals with target <= origin; "
        "later CV and holdout residuals are admitted as their dates become eligible. Minimum count 19 is checked. "
        "Native classical intervals and prequential conformal intervals are assessed separately; Seasonal Naive intervals are unavailable.\n\n"
        "Invalid/negative/non-converged forecasts are retained raw, with failure logs and valid-origin denominators. "
        "Operational fallback uses only the pre-registered fallback; failed fallbacks leave the served prediction unavailable. "
        "Primary metrics exclude invalid forecasts but raw values/errors remain in the prediction artifact. "
        "Coverage is conditional on valid available intervals, with interval-origin counts. No retrospective champion changes or interval tuning occurs."
    )


def _build_interpretation(leaderboard: pd.DataFrame) -> str:
    drift = leaderboard.cv_vs_holdout_smape_drift
    improved = int(leaderboard.mean_smape.lt(leaderboard.baseline_mean_smape).sum())
    return (
        f"Champions have lower mean valid-origin sMAPE than the full Seasonal Naive benchmark in {improved}/42 tasks. "
        f"Mean CV-to-holdout drift is {drift.mean():.4f} percentage points; median is {drift.median():.4f}. "
        "The matrix retains deteriorations and high-error cases. Different valid-origin counts must be considered before comparing failed models. "
        "Stored skill uses paired valid primary origins; raw complete-origin errors and operational fallback forecasts remain separately accessible.\n\n"
        "Differences across fruit, energy, metals, machinery and TOTAL reflect predictive heterogeneity; they do not identify causal macro effects. "
        "XGBoost uses trade history/calendar alongside macro features. SHAP is a forecast-driver decomposition, not economic causality. "
        "Native and conformal empirical coverage can differ from nominal 80%/95%; no calibration parameters were adjusted after these outcomes."
    )


def _build_provenance_text(metadata: JsonObject) -> str:
    return (
        f"Git commit: `{metadata['git_commit']}`. Dirty source tree: `{metadata['is_working_tree_dirty']}`. "
        f"Source fingerprint: `{metadata['source_fingerprint']}`. Input and output SHA-256 values are in `m11_holdout_run.json`. "
        "Commands: `make m11-holdout`; independent reproduction: `make m11-holdout M11_OUTPUT_DIR=/tmp/m11-repro`. "
        "Outputs never overwrite existing files. Data hashes and forecasts must reproduce; timestamps/Git state may differ. "
        "Exact software-test results are recorded separately in `M11_ACCEPTANCE_REPORT.md`, not asserted by this generator."
    )


def _build_comparison_sections(leaderboard: pd.DataFrame, predictions: pd.DataFrame) -> list[str]:
    family = (
        predictions.groupby(["role", "model_family"])
        .apply(lambda group: pd.Series(compute_holdout_metrics(group)), include_groups=False)
        .reset_index()
    )
    return [
        "## 2. Global Champion / Fallback / Baseline Performance",
        "",
        render_table(build_global_holdout_table(predictions)),
        "",
        "## 3. Empirical Coverage by Role and Family",
        "",
        render_table(family),
        "",
        "## 4. Complete 42-Task CV-to-Holdout Matrix",
        "",
        render_table(leaderboard[MATRIX_COLUMNS]),
        "",
    ]


MATRIX_COLUMNS = [
    "sector_id",
    "horizon",
    "champion_model",
    "fallback_model",
    "mean_smape",
    "cv_mean_smape",
    "cv_vs_holdout_smape_drift",
    "baseline_mean_smape",
    "skill_vs_naive_pct",
    "n_valid_origins",
    "n_operational_fallbacks",
    "empirical_coverage_80",
    "empirical_coverage_95",
    "n_interval_origins",
]
