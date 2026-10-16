"""Immutable M11 execution entry point and artifact packaging."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tr_ru_forecast.core.logging import get_logger, setup_logging
from tr_ru_forecast.evaluation.holdout_io import (
    build_holdout_provenance,
    load_holdout_inputs,
    publish_holdout_outputs,
)
from tr_ru_forecast.evaluation.holdout_report import build_holdout_report, build_shap_summary
from tr_ru_forecast.evaluation.holdout_runner import (
    build_holdout_leaderboard,
    run_holdout_predictions,
)

logger = get_logger(__name__)


def run_m11_holdout(root: Path, output_directory: Path | None = None) -> dict[str, object]:
    """Run frozen holdout. Args: root, fresh output path. Returns: metadata. Raises: logged validation/I/O errors."""
    directory = output_directory or root / "data/processed"
    if (directory / "m11_holdout_run.json").exists():
        raise FileExistsError("M11 output exists; use a fresh output directory")
    inputs = load_holdout_inputs(root)
    metadata = {
        **build_holdout_provenance(inputs),
        "run_id": f"m11_{inputs.input_hashes['configs/holdout_spec_v1_1.yaml'][:12]}",
        "created_at": datetime.now(UTC).isoformat(),
        "holdout_start": "2025-01",
        "holdout_end": "2025-12",
        "horizons": [1, 3, 6],
    }
    predictions, local_shap = run_holdout_predictions(inputs)
    leaderboard = build_holdout_leaderboard(predictions, inputs.registry.champions)
    tables = _build_tables(predictions, leaderboard, local_shap)
    report = build_holdout_report(leaderboard, predictions, metadata)
    return _publish_holdout_package((root, output_directory, directory), (tables, report), metadata)


def _build_tables(
    predictions: pd.DataFrame, leaderboard: pd.DataFrame, shap_rows: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    log_columns = [
        "sector_id",
        "horizon",
        "forecast_origin",
        "forecast_period",
        "role",
        "model_family",
        "fit_status",
        "converged",
        "failure_message",
        "optimizer_used",
        "retry_count",
        "warning_type",
        "train_end",
        "calibration_count",
        "max_calibration_target_period",
        "has_operational_fallback",
    ]
    return {
        "m11_holdout_predictions.parquet": predictions,
        "m11_holdout_leaderboard.parquet": leaderboard,
        "m11_holdout_leaderboard.csv": leaderboard,
        "m11_holdout_fit_log.parquet": predictions[log_columns],
        "m11_shap_values.parquet": shap_rows,
        "m11_shap_summary.parquet": build_shap_summary(shap_rows),
    }


def _publish_default_report(root: Path, report: str) -> None:
    path = root / "docs/M11_FINAL_HOLDOUT_REPORT.md"
    with path.open("x", encoding="utf-8") as target:
        target.write(report)


def main() -> None:
    """Run command. Args: argv. Returns: None. Raises: logged errors."""
    parser = argparse.ArgumentParser(description="Frozen retrospective M11 holdout evaluation")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()
    setup_logging()
    try:
        run_m11_holdout(arguments.project_root, arguments.output_dir)
    except Exception:
        logger.exception("M11 holdout failed")
        raise


def _publish_holdout_package(
    paths: tuple[Path, Path | None, Path],
    content: tuple[dict[str, pd.DataFrame], str],
    metadata: dict[str, object],
) -> dict[str, object]:
    root, output_directory, directory = paths
    tables, report = content
    predictions = tables["m11_holdout_predictions.parquet"]
    leaderboard = tables["m11_holdout_leaderboard.parquet"]
    hashes = publish_holdout_outputs(directory, tables, {"M11_FINAL_HOLDOUT_REPORT.md": report})
    metadata.update(
        {
            "output_hashes": hashes,
            "n_task_origins": 504,
            "n_role_rows": len(predictions),
            "n_tasks": len(leaderboard),
        }
    )
    with (directory / "m11_holdout_run.json").open("x", encoding="utf-8") as output:
        output.write(json.dumps(metadata, indent=2))
    if output_directory is None:
        _publish_default_report(root, report)
    logger.info("M11 completed: %d task/origin cells, %d role predictions", 504, len(predictions))
    return metadata


if __name__ == "__main__":
    main()
