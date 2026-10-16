"""Build one checksum-verified, immutable application snapshot from disk."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import pandas as pd

from tr_ru_forecast.serving.artifacts import (
    ArtifactError,
    ArtifactTables,
    hash_file,
    load_tables,
    read_document,
    verify_artifacts,
)
from tr_ru_forecast.serving.catalog_schemas import Provenance, TaskComparison
from tr_ru_forecast.serving.comparison import build_comparisons, build_global_kpis
from tr_ru_forecast.serving.config import ServingSettings
from tr_ru_forecast.serving.explanations import build_explanation
from tr_ru_forecast.serving.forecast_schemas import ForecastResponse
from tr_ru_forecast.serving.forecasts import build_forecast_task
from tr_ru_forecast.serving.history import build_history, build_sector
from tr_ru_forecast.serving.indicator_schemas import IndicatorPoint
from tr_ru_forecast.serving.integrity import validate_table_contracts
from tr_ru_forecast.serving.mirror import build_mirror_points
from tr_ru_forecast.serving.schemas import EVALUATION_SCOPE
from tr_ru_forecast.serving.snapshot import FrozenSnapshot


class SnapshotRepository(Protocol):
    """Data-access abstraction; services consume immutable snapshots only."""

    def load(self) -> FrozenSnapshot:
        """Load a verified snapshot. Args: self. Returns: snapshot. Raises: ArtifactError."""
        ...


class DiskArtifactRepository:
    """Private load-time frames are discarded after immutable projection."""

    def __init__(self, settings: ServingSettings) -> None:
        """Bind locations. Args: settings. Returns: none. Raises: none."""
        self._settings = settings

    def load(self) -> FrozenSnapshot:
        """Verify and project frozen bytes. Args: self. Returns: snapshot. Raises: ArtifactError/I/O errors."""
        manifest, digest = verify_artifacts(self._settings)
        tables = load_tables(self._settings)
        validate_table_contracts(tables)
        tasks = build_comparisons(tables)
        forecasts = _build_forecasts(tables, tasks)
        if len(tables.predictions) != 1512 or len(tables.shap) != 9072:
            raise ArtifactError("Incomplete holdout/SHAP snapshot")
        return FrozenSnapshot(
            sectors=tuple(build_sector(_select_history(tables, item.sector_id), tuple(
                task for task in tasks if task.sector_id == item.sector_id
            )) for item in forecasts), comparisons=tasks, forecasts=forecasts,
            explanations=tuple(build_explanation(task, _select_task(tables.shap, task),
                                                 _select_task(tables.shap_summary, task)) for task in tasks),
            mirrors=build_mirror_points(tables.mirror), indicators=_build_indicators(tables.macro),
            kpis=build_global_kpis(tasks), m10=_build_provenance(self._settings, "m10_selection_run.json"),
            m11=_build_provenance(self._settings, "m11_holdout_run.json"),
            manifest_sha256=digest, verified_file_count=len(manifest.files),
            prediction_count=len(tables.predictions), shap_row_count=len(tables.shap),
        )


def _select_task(frame: pd.DataFrame, task: TaskComparison) -> pd.DataFrame:
    return frame.loc[frame.sector_id.eq(task.sector_id) & frame.horizon.eq(task.horizon)]


def _select_history(tables: ArtifactTables, sector_id: str) -> pd.DataFrame:
    direction, hs2 = sector_id.split(":")
    return tables.history.loc[tables.history.direction.eq(direction) & tables.history.hs2.eq(hs2)]


def _build_forecasts(
    tables: ArtifactTables, tasks: tuple[TaskComparison, ...]
) -> tuple[ForecastResponse, ...]:
    forecasts = []
    for sector_id in sorted({task.sector_id for task in tasks}):
        history = build_history(_select_history(tables, sector_id))
        selected = tuple(task for task in tasks if task.sector_id == sector_id)
        forecasts.append(ForecastResponse(
            sector_id=sector_id, direction=selected[0].direction, hs2=selected[0].hs2,
            evaluation_scope=EVALUATION_SCOPE, observation_cutoff=history[-1].period, history=history,
            tasks=tuple(build_forecast_task(_select_task(tables.predictions, task), task) for task in selected),
        ))
    return tuple(forecasts)


def _build_indicators(frame: pd.DataFrame) -> tuple[IndicatorPoint, ...]:
    records = frame.sort_values(["indicator_id", "period"])[list(IndicatorPoint.model_fields)].to_dict(orient="records")
    return tuple(IndicatorPoint.model_validate(record) for record in records)


def _build_provenance(settings: ServingSettings, filename: str) -> Provenance:
    path = settings.artifact_root / filename
    document = read_document(path)
    nested = document.get("provenance")
    metadata: Mapping[str, object] = nested if isinstance(nested, dict) else document
    return Provenance.model_validate({
        "git_commit": metadata["git_commit"], "source_fingerprint": metadata["source_fingerprint"],
        "created_at": metadata["created_at"], "run_id": metadata.get("run_id"),
        "manifest_sha256": hash_file(path),
    })
