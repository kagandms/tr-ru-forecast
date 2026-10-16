"""Framework-independent read services over immutable verified snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from tr_ru_forecast.serving.catalog_schemas import (
    LeaderboardResponse,
    SectorCatalog,
    TaskComparison,
)
from tr_ru_forecast.serving.explain_schemas import AvailableExplanation, ExplanationResponse
from tr_ru_forecast.serving.forecast_schemas import ForecastResponse, TradeResponse
from tr_ru_forecast.serving.indicator_schemas import IndicatorsResponse, MethodologyResponse
from tr_ru_forecast.serving.mirror import build_mirror_response
from tr_ru_forecast.serving.mirror_schemas import MirrorResponse
from tr_ru_forecast.serving.schemas import EVALUATION_SCOPE, ChampionFamily, Direction, Horizon
from tr_ru_forecast.serving.snapshot import FrozenSnapshot


class CatalogNotFound(Exception):
    """Unknown or contradictory public catalog selection."""


@dataclass(frozen=True)
class TaskFilter:
    """Immutable task filter contract."""

    direction: Direction | None = None
    hs2: str | None = None
    horizon: Horizon | None = None
    champion_model: ChampionFamily | None = None


@dataclass(frozen=True)
class PageFilter:
    """Immutable page filter contract."""

    offset: int = 0
    limit: int = 1000
    start_month: str | None = None
    end_month: str | None = None


@dataclass(frozen=True)
class MirrorFilter:
    """Immutable mirror filter contract."""

    sector_id: str | None = None
    hs2: str | None = None
    direction: Direction | None = None


@dataclass(frozen=True)
class ArtifactService:
    """Domain projections; errors contain no filesystem paths or secrets."""

    snapshot: FrozenSnapshot

    def sectors(self) -> SectorCatalog:
        """Return all series/tasks. Args: self. Returns: catalog. Raises: none."""
        return SectorCatalog(evaluation_scope=EVALUATION_SCOPE, sectors=self.snapshot.sectors)

    def leaderboard(self, filters: TaskFilter) -> LeaderboardResponse:
        """Filter task rows; KPIs remain global. Args: filters. Returns: leaderboard. Raises: CatalogNotFound."""
        if filters.hs2 and not any(task.hs2 == filters.hs2 for task in self.snapshot.comparisons):
            raise CatalogNotFound("Unknown HS2 code")
        tasks = tuple(task for task in self.snapshot.comparisons if _matches_task(task, filters))
        return LeaderboardResponse(evaluation_scope=EVALUATION_SCOPE, summary=self.snapshot.kpis, tasks=tasks)

    def forecasts(self, sector_id: str, horizon: Horizon | None) -> ForecastResponse:
        """Select history/tasks. Args: sector_id, horizon. Returns: forecasts. Raises: CatalogNotFound."""
        canonical = self._resolve_sector(sector_id)
        record = next(item for item in self.snapshot.forecasts if item.sector_id == canonical)
        tasks = tuple(task for task in record.tasks if horizon is None or task.horizon == horizon)
        return record.model_copy(update={"tasks": tasks})

    def explain(self, sector_id: str, selection: tuple[Horizon, str | None]) -> ExplanationResponse:
        """Filter local month without changing global importance. Args: sector_id, selection. Returns: explanation. Raises: CatalogNotFound."""
        canonical = self._resolve_sector(sector_id)
        horizon, month = selection
        record = next(item for item in self.snapshot.explanations if item.sector_id == canonical and item.horizon == horizon)
        if isinstance(record, AvailableExplanation) and month:
            return record.model_copy(update={"local": tuple(item for item in record.local if item.target_month == month)})
        return record

    def mirror(self, filters: MirrorFilter, page: PageFilter) -> MirrorResponse:
        """Select mirror records with full-selection summaries. Args: filters, page. Returns: mirror. Raises: CatalogNotFound."""
        canonical = self._resolve_mirror(filters.sector_id) if filters.sector_id else None
        if filters.hs2 and not any(item.hs2 == filters.hs2 for item in self.snapshot.mirrors):
            raise CatalogNotFound("Unknown mirror HS2 code")
        selected = tuple(item for item in self.snapshot.mirrors if
                         (canonical is None or item.sector_id == canonical) and
                         (filters.direction is None or item.direction == filters.direction) and
                         (filters.hs2 is None or item.hs2 == filters.hs2) and
                         _within_dates(item.period, page))
        return build_mirror_response(selected, page.offset, page.limit)

    def trade(self, sector_id: str, page: PageFilter) -> TradeResponse:
        """Paginate exact-decimal official history. Args: sector_id, page. Returns: trade. Raises: CatalogNotFound."""
        record = self.forecasts(sector_id, None)
        selected = tuple(point for point in record.history if _within_dates(point.period, page))
        return TradeResponse(sector_id=record.sector_id, observations=selected[page.offset:page.offset + page.limit],
                             total_observations=len(selected), offset=page.offset, limit=page.limit)

    def indicators(self, indicator_id: str | None, page: PageFilter) -> IndicatorsResponse:
        """Filter persisted macro observations. Args: indicator_id, page. Returns: indicators. Raises: CatalogNotFound."""
        if indicator_id and not any(item.indicator_id == indicator_id for item in self.snapshot.indicators):
            raise CatalogNotFound("Unknown indicator")
        selected = tuple(item for item in self.snapshot.indicators if
                         (indicator_id is None or item.indicator_id == indicator_id) and _within_dates(item.period, page))
        return IndicatorsResponse(evaluation_scope=EVALUATION_SCOPE,
                                  observations=selected[page.offset:page.offset + page.limit],
                                  total_observations=len(selected), offset=page.offset, limit=page.limit)

    def methodology(self) -> MethodologyResponse:
        """Explain frozen scope and serialization. Args: self. Returns: methodology. Raises: none."""
        return MethodologyResponse(
            specification_version="1.1", evaluation_scope=EVALUATION_SCOPE,
            forecast_kind="historical_holdout", unit="USD", interval_coverage_unit="fraction",
            monetary_serialization="Official observations are decimal USD strings from persisted source precision; model forecasts/SHAP/metrics are finite numbers or null.",
            target_policy=("TR_RU: Türkiye reporter 792 export to Russia 643",
                           "RU_TR: Türkiye reporter 792 import from Russia 643"),
            limitations=("No historical publication-time/vintage fidelity proof.",
                         "Early origins reconstruct subsequently frozen model assignments.",
                         "Nominal 80% XGBoost champion coverage was 67.59%; no causal SHAP claims.",
                         "Trade assumed available at reference-month end; macro approximate as-of rules.",
                         "No forward inference, refit, live-source refresh or original-vintage guarantee."),
            m10_run_digest=self.snapshot.m10.manifest_sha256,
            m11_run_digest=self.snapshot.m11.manifest_sha256,
        )

    def _resolve_sector(self, sector_id: str) -> str:
        canonical = sector_id.replace("__", ":")
        if not any(item.sector_id == canonical for item in self.snapshot.sectors):
            raise CatalogNotFound("Unknown sector_id")
        return canonical

    def _resolve_mirror(self, sector_id: str) -> str:
        canonical = sector_id.replace("__", ":")
        if not any(item.sector_id == canonical for item in self.snapshot.mirrors):
            raise CatalogNotFound("Unknown mirror sector_id")
        return canonical


def _matches_task(task: TaskComparison, filters: TaskFilter) -> bool:
    return all(value is None or getattr(task, name) == value for name, value in (
        ("direction", filters.direction), ("hs2", filters.hs2),
        ("horizon", filters.horizon), ("champion_model", filters.champion_model),
    ))


def _within_dates(month: str, page: PageFilter) -> bool:
    return (page.start_month is None or month >= page.start_month) and (page.end_month is None or month <= page.end_month)
