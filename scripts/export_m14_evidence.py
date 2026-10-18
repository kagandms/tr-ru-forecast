"""Export source-derived M11 task metrics for the M14 research package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data/processed/m11_holdout_leaderboard.parquet"
OUTPUT_PATH = PROJECT_ROOT / "docs/research/results/holdout-task-results.csv"
RESULT_COLUMNS = [
    "sector_id",
    "direction",
    "hs2",
    "horizon",
    "champion_model",
    "fallback_model",
    "n_expected_origins",
    "n_valid_origins",
    "n_failed_origins",
    "operational_mean_smape",
    "operational_mean_mae",
    "baseline_mean_smape",
    "skill_vs_naive_pct",
    "raw_mean_smape",
    "n_operational_fallbacks",
    "n_interval_origins",
    "empirical_coverage_80",
    "empirical_coverage_95",
    "cv_mean_smape",
    "cv_vs_holdout_smape_drift",
]


def build_task_results(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Select the public, derived M11 metrics from the frozen task leaderboard.

    Args:
        leaderboard: M11 leaderboard containing one row per direction, HS2 and horizon.

    Returns:
        A sorted frame with task-level model, forecast, baseline, interval and CV metrics.

    Raises:
        ValueError: If required columns are missing or task keys are duplicated.
    """
    missing_columns = sorted(set(RESULT_COLUMNS) - set(leaderboard.columns))
    if missing_columns:
        raise ValueError(f"M11 leaderboard is missing required columns: {missing_columns}")
    task_keys = ["direction", "hs2", "horizon"]
    if leaderboard.duplicated(task_keys).any():
        raise ValueError("M11 leaderboard contains duplicate direction/HS2/horizon tasks")
    return leaderboard.loc[:, RESULT_COLUMNS].sort_values(task_keys).reset_index(drop=True)


def export_task_results(source_path: Path, output_path: Path) -> None:
    """Write the derived M11 task table without copying raw trade observations.

    Args:
        source_path: Path to the local, protected M11 leaderboard Parquet file.
        output_path: Destination CSV path for the public research evidence table.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the protected local M11 source artifact is absent.
        ValueError: If the source does not satisfy the frozen task-table contract.
    """
    if not source_path.is_file():
        raise FileNotFoundError(f"Required local M11 artifact is missing: {source_path}")
    leaderboard = pd.read_parquet(source_path)
    task_results = build_task_results(leaderboard)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_results.to_csv(output_path, index=False, float_format="%.8f", lineterminator="\n")


def main() -> None:
    """Export the default M14 task table from the local M11 evaluation artifact.

    Args:
        None.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the local M11 leaderboard has not been mounted.
        ValueError: If the leaderboard violates the frozen task-table contract.
    """
    export_task_results(SOURCE_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
