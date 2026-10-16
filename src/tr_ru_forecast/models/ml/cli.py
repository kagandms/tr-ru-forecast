"""M9 report generation from immutable result frames; no evaluation or selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.models.ml.pipeline import run_m9_xgboost_pipeline
from tr_ru_forecast.models.ml.records import JsonObject

logger = get_logger(__name__)


@dataclass(frozen=True)
class ReportFrames:
    """Separate M9 artifacts with explicit report roles."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    tuning: pd.DataFrame
    interval_audit: pd.DataFrame
    local_shap: pd.DataFrame
    global_shap: pd.DataFrame
    fit_log: pd.DataFrame


def _render_table(frame: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    return [header, separator] + [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]


def _build_report_metadata(metadata: JsonObject) -> list[str]:
    lines = [
        "# M9 XGBoost Direct Forecasting Report",
        "",
        f"**Run ID:** `{metadata['run_id']}`",
        f"**Generated:** {metadata['created_at']}",
        f"**Git commit:** `{metadata['git_commit']}`",
        f"**Source fingerprint:** `{metadata['source_fingerprint']}`",
        "**Status:** Development validation run",
        "",
        "## 1. Specification / Scope",
        "",
        "Direct h1/h3/h6 models use frozen pre-validation tuning, 281 monthly CV folds, 21 predictors, origin-safe residual calibration and fold-local TreeSHAP. "
        "This report neither selects champions nor evaluates the final holdout.",
        "",
        "## 2. Frozen Inputs & Outputs",
        "",
    ]
    for name, digest in metadata["output_artifact_hashes"].items():
        lines.append(f"- `{name}`: `{digest}`")
    return lines


def _build_report_results(frames: ReportFrames) -> list[str]:
    return (
        ["", "## 3. Frozen Hyperparameters", ""]
        + _render_table(frames.tuning.loc[frames.tuning.selected])
        + ["", "## 4. Complete Validation Matrix", ""]
        + _render_table(frames.metrics)
        + ["", "## 5. Empirical Interval Coverage", ""]
        + _build_coverage_table(frames.metrics)
        + [
            "",
            "Coverage is conditional on interval availability; nominal levels are 80% and 95%.",
            "",
            "## 6. Non-Causal SHAP Forecast Drivers",
            "",
        ]
        + _render_table(frames.global_shap)
        + [
            "",
            "Local SHAP is retained in the separate long-form artifact; raw-dollar contributions must not be pooled across differently scaled series.",
            "",
            "## 7. Limitations & Acceptance",
            "",
            "Retrospective latest-vintage data and approximate macro publication lags do not establish historical-vintage or real-time decision fidelity. "
            "Negative forecasts and lower interval bounds remain unclipped. Test and source-review evidence is recorded in M9_ACCEPTANCE_REPORT.md.",
        ]
    )


def generate_m9_report(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    interval_audit_df: pd.DataFrame,
    local_shap_df: pd.DataFrame,
    global_shap_df: pd.DataFrame,
    fit_log_df: pd.DataFrame,
    run_meta: JsonObject,
    report_path: Path,
) -> None:
    """Write audited M9 report. Args: artifact frames, metadata, path. Returns: None. Raises: I/O errors.

    Legacy argument shape is preserved; internal helpers consume one immutable artifact bundle.
    """
    frames = ReportFrames(
        predictions_df,
        metrics_df,
        tuning_df,
        interval_audit_df,
        local_shap_df,
        global_shap_df,
        fit_log_df,
    )
    lines = _build_report_metadata(run_meta) + _build_report_results(frames)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved M9 report to %s", report_path)


def main() -> None:
    """Run M9 CLI. Args: none. Returns: None. Raises: pipeline or file errors."""
    settings = get_settings()
    predictions, metrics, metadata = run_m9_xgboost_pipeline()
    names = ("tuning_results", "interval_audit", "shap_values", "shap_global", "fit_log")
    frames = [
        pd.read_parquet(settings.processed_data_dir / f"m9_xgboost_{name}.parquet")
        for name in names
    ]
    generate_m9_report(
        predictions, metrics, *frames, metadata, settings.project_root / "docs/M9_XGBOOST_REPORT.md"
    )


def _build_coverage_table(metrics: pd.DataFrame) -> list[str]:
    coverage = (
        metrics.groupby("horizon")[["coverage_80", "coverage_95", "interval_available_count"]]
        .mean()
        .reset_index()
    )
    return _render_table(coverage)


if __name__ == "__main__":
    main()
