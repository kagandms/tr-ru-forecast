"""Catalog, provenance and model comparison contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from tr_ru_forecast.serving.schemas import (
    Direction,
    FrozenModel,
    Horizon,
    IntervalDiagnostics,
    MetricSummary,
    ModelFamily,
    Month,
    Number,
)


class Provenance(FrozenModel):
    """Immutable provenance contract."""

    git_commit: str
    source_fingerprint: str
    manifest_sha256: str
    created_at: datetime
    run_id: str | None = None


class ArtifactReadiness(FrozenModel):
    """Immutable artifact readiness contract."""

    m10: bool
    m11: bool
    history: bool
    mirror: bool
    macro: bool


class HealthResponse(FrozenModel):
    """Immutable health response contract."""

    status: Literal["healthy", "degraded"]
    timestamp: datetime
    version: str
    environment: str
    specification_version: Literal["1.1"] = "1.1"
    evaluation_scope: str
    checksum_status: Literal["verified", "failed"]
    verification_scope: str
    verified_file_count: int
    serving_manifest_sha256: str | None
    total_tasks: int
    total_holdout_predictions: int
    total_shap_rows: int
    readiness: ArtifactReadiness
    m10: Provenance | None
    m11: Provenance | None
    error: str | None = None


class TaskAssignment(FrozenModel):
    """Immutable task assignment contract."""

    horizon: Horizon
    champion_model: ModelFamily
    fallback_model: ModelFamily


class Sector(FrozenModel):
    """Immutable sector contract."""

    sector_id: str
    direction: Direction
    hs2: str
    title: str
    is_total: bool
    history_start: Month
    history_end: Month
    latest_trade_value_usd: str | None
    source: str
    reporter_code: int
    flow: str
    retrieved_at: datetime
    tasks: tuple[TaskAssignment, ...]


class SectorCatalog(FrozenModel):
    """Immutable sector catalog contract."""

    evaluation_scope: str
    sectors: tuple[Sector, ...]


class CvCandidate(FrozenModel):
    """One frozen M10 candidate; means use only its recorded valid CV origins."""

    model_family: ModelFamily
    mean_smape: Number
    mean_mae: Number
    mean_rmse: Number
    eligibility_status: str
    n_expected_folds: int
    n_valid_folds: int
    n_negative_predictions: int
    convergence_rate: Number


class TaskComparison(FrozenModel):
    """Immutable task comparison contract."""

    sector_id: str
    direction: Direction
    hs2: str
    horizon: Horizon
    champion_model: ModelFamily
    fallback_model: ModelFamily
    selection_reason: str
    has_stability_flag: bool
    cv: MetricSummary
    champion: MetricSummary
    operational: MetricSummary
    fallback: MetricSummary
    baseline: MetricSummary
    cv_candidates: tuple[CvCandidate, ...]
    raw_champion_smape: Number
    cv_vs_holdout_drift: Number
    fallback_replacements: int
    intervals: IntervalDiagnostics


class GlobalKpis(FrozenModel):
    """Immutable global kpis contract."""

    operational_mean_smape: Number
    baseline_mean_smape: Number
    cv_mean_skill_pct: Number
    sarima_champions: int
    xgboost_champions: int
    fallback_replacements: int
    total_tasks: int
    weighting: Literal["equal_task"] = "equal_task"


class LeaderboardResponse(FrozenModel):
    """Immutable leaderboard response contract."""

    evaluation_scope: str
    summary: GlobalKpis
    tasks: tuple[TaskComparison, ...]
