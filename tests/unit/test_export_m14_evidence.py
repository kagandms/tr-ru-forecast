"""Contract tests for the generated M14 holdout evidence table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import export_m14_evidence
from scripts.export_m14_evidence import RESULT_COLUMNS, build_task_results, export_task_results


def test_build_task_results_preserves_operational_and_raw_scores() -> None:
    """The public table keeps raw champion scores distinct from governed forecasts."""
    row = {column: 0 for column in RESULT_COLUMNS}
    row.update(
        {
            "sector_id": "TR_RU:TOTAL",
            "direction": "TR_RU",
            "hs2": "TOTAL",
            "horizon": 1,
            "champion_model": "sarima",
            "fallback_model": "xgboost",
            "operational_mean_smape": 12.5,
            "raw_mean_smape": 13.0,
        }
    )

    result = build_task_results(pd.DataFrame([row]))

    assert result.loc[0, "operational_mean_smape"] == 12.5
    assert result.loc[0, "raw_mean_smape"] == 13.0
    assert result.loc[0, "sector_id"] == "TR_RU:TOTAL"


def test_build_task_results_rejects_duplicate_task_keys() -> None:
    """Duplicate direction, sector and horizon combinations cannot enter the table."""
    row = {column: 0 for column in RESULT_COLUMNS}
    duplicate_rows = pd.DataFrame([row, row])

    with pytest.raises(ValueError, match="duplicate"):
        build_task_results(duplicate_rows)


def test_build_task_results_rejects_incomplete_source_contract() -> None:
    """A partial leaderboard fails instead of emitting an incomplete result table."""
    incomplete = pd.DataFrame({"direction": ["TR_RU"]})

    with pytest.raises(ValueError, match="missing required columns"):
        build_task_results(incomplete)


def test_export_task_results_writes_only_the_derived_result_contract(tmp_path: Path) -> None:
    """The exporter writes a task table while omitting unrelated source columns."""
    row = {column: 0 for column in RESULT_COLUMNS}
    row["direction"] = "TR_RU"
    row["hs2"] = "08"
    row["horizon"] = 1
    source_path = tmp_path / "leaderboard.parquet"
    output_path = tmp_path / "holdout.csv"
    pd.DataFrame([{**row, "unpublished_source_column": "excluded"}]).to_parquet(source_path)

    export_task_results(source_path, output_path)

    result = pd.read_csv(output_path, dtype={"hs2": str})
    assert len(result) == 1
    assert result.loc[0, "hs2"] == "08"
    assert "unpublished_source_column" not in result.columns


def test_main_uses_configured_default_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command entry point exports from its configured source to its configured target."""
    row = {column: 0 for column in RESULT_COLUMNS}
    source_path = tmp_path / "leaderboard.parquet"
    output_path = tmp_path / "nested" / "holdout.csv"
    pd.DataFrame([row]).to_parquet(source_path)
    monkeypatch.setattr(export_m14_evidence, "SOURCE_PATH", source_path)
    monkeypatch.setattr(export_m14_evidence, "OUTPUT_PATH", output_path)

    export_m14_evidence.main()

    assert output_path.is_file()
    assert len(pd.read_csv(output_path)) == 1
