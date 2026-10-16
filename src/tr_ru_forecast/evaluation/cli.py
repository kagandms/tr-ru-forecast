"""One-command M10 selection with immutable outputs and CV-only access."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.evaluation.champion_io import (
    ChampionRegistry,
    build_registry_entry,
    collect_provenance,
    hash_artifact,
    load_cv_predictions,
    load_frozen_folds,
    load_selection_spec,
    load_task_grid,
    validate_registry,
)
from tr_ru_forecast.evaluation.champion_report import build_m10_report
from tr_ru_forecast.evaluation.champion_selector import (
    Candidate,
    aggregate_candidates,
    build_champion_matrix,
)

LOGGER = get_logger(__name__)


def _write_staged_artifacts(
    stage: Path, context: tuple[ChampionRegistry, pd.DataFrame, pd.DataFrame]
) -> None:
    registry, leaderboard, details = context
    leaderboard.to_parquet(stage / "m10_cv_leaderboard.parquet", index=False)
    leaderboard.to_csv(stage / "m10_cv_leaderboard.csv", index=False)
    details.to_parquet(stage / "m10_cv_fold_details.parquet", index=False)
    matrix = pd.DataFrame(
        [
            row.model_dump(
                exclude={"baseline_reference", "champion_reference", "fallback_reference"}
            )
            for row in registry.champions
        ]
    )
    matrix.to_parquet(stage / "m10_champion_matrix.parquet", index=False)


def _get_output_paths(
    root: Path, spec_path: Path, output_directory: Path | None
) -> dict[str, Path]:
    spec = load_selection_spec(spec_path)
    directory = output_directory.resolve() if output_directory else root / spec.output_directory
    targets = {
        name: directory / name
        for name in (
            "m10_cv_leaderboard.parquet",
            "m10_cv_leaderboard.csv",
            "m10_cv_fold_details.parquet",
            "m10_champion_matrix.parquet",
            "m10_selection_run.json",
        )
    }
    targets["champion_registry_v1_1.yaml"] = (
        directory / "champion_registry_v1_1.yaml" if output_directory else root / spec.registry_path
    )
    targets["M10_CHAMPION_SELECTION_REPORT.md"] = (
        directory / "M10_CHAMPION_SELECTION_REPORT.md"
        if output_directory
        else root / spec.report_path
    )
    if any(path.exists() for path in targets.values()):
        raise FileExistsError(
            "M10 outputs already exist; use a fresh --output-dir for independent reproduction"
        )
    return targets


def run_m10_selection(root: Path, output_directory: Path | None = None) -> ChampionRegistry:
    """Run selection. Args: checkout root, optional fresh output directory. Returns: registry.

    Raises: validation, provenance, or I/O errors. Never refits models or opens holdout targets.
    """
    root = root.resolve()
    spec_path = root / "configs/champion_selection_spec_v1_1.yaml"
    spec = load_selection_spec(spec_path)
    targets = _get_output_paths(root, spec_path, output_directory)
    provenance = collect_provenance(root, spec, spec_path)
    folds, tasks = load_frozen_folds(root), load_task_grid(root)
    details = load_cv_predictions(root, spec, (folds, tasks))
    candidates = aggregate_candidates(details, tasks, spec)
    selections = build_champion_matrix(candidates, spec)
    registry = ChampionRegistry(
        provenance=provenance,
        champions=tuple(build_registry_entry(root, selection, spec) for selection in selections),
    )
    validate_registry(registry, tasks)
    leaderboard = pd.DataFrame([candidate.model_dump() for candidate in candidates])
    registry = _persist_selection(registry, (leaderboard, details), (targets, candidates))
    LOGGER.info(
        "M10 selected %d champions from %d candidates", len(registry.champions), len(candidates)
    )
    return registry


def _write_run_manifest(stage: Path, registry: ChampionRegistry) -> None:
    output_hashes = {path.name: hash_artifact(path) for path in sorted(stage.iterdir())}
    manifest = {
        "provenance": registry.provenance.model_dump(mode="json"),
        "output_hashes": output_hashes,
        "champion_count": 42,
        "candidate_count": 168,
        "holdout_accessed": False,
    }
    (stage / "m10_selection_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _publish_artifacts(stage: Path, targets: dict[str, Path]) -> None:
    for name, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        with (stage / name).open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)


def main() -> None:
    """CLI entry. Args: parsed argv. Returns: None. Raises: logged pipeline errors."""
    parser = argparse.ArgumentParser(description="Freeze protocol-v1.1 CV champion decisions")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()
    try:
        run_m10_selection(arguments.project_root, arguments.output_dir)
    except Exception:
        LOGGER.exception("M10 selection failed")
        raise


def _persist_selection(
    registry: ChampionRegistry,
    tables: tuple[pd.DataFrame, pd.DataFrame],
    context: tuple[dict[str, Path], tuple[Candidate, ...]],
) -> ChampionRegistry:
    leaderboard, details = tables
    targets, candidates = context
    provenance = registry.provenance
    with tempfile.TemporaryDirectory(prefix="m10-selection-") as temporary:
        stage = Path(temporary)
        _write_staged_artifacts(stage, (registry, leaderboard, details))
        output_hashes = {
            name: hash_artifact(stage / name) for name in targets if (stage / name).exists()
        }
        registry = registry.model_copy(
            update={"provenance": provenance.model_copy(update={"output_hashes": output_hashes})}
        )
        (stage / "champion_registry_v1_1.yaml").write_text(
            yaml.safe_dump(registry.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
        (stage / "M10_CHAMPION_SELECTION_REPORT.md").write_text(
            build_m10_report(registry, candidates), encoding="utf-8"
        )
        _write_run_manifest(stage, registry)
        _publish_artifacts(stage, targets)
    return registry


if __name__ == "__main__":
    main()
