"""Versioned read-only routes with strongly typed request/response contracts."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response

from api.dependencies import RuntimeDependency, ServiceDependency
from api.queries import (
    ExplainQuery,
    ForecastQuery,
    ForecastSectorQuery,
    IndicatorQuery,
    MirrorQuery,
    TaskQuery,
    TradeQuery,
)
from tr_ru_forecast import __version__
from tr_ru_forecast.config import get_settings
from tr_ru_forecast.serving.catalog_schemas import (
    HealthResponse,
    LeaderboardResponse,
    SectorCatalog,
)
from tr_ru_forecast.serving.explain_schemas import ExplanationResponse
from tr_ru_forecast.serving.forecast_schemas import ForecastResponse, TradeResponse
from tr_ru_forecast.serving.indicator_schemas import (
    IndicatorsResponse,
    MethodologyResponse,
    OverviewResponse,
)
from tr_ru_forecast.serving.mirror_schemas import MirrorResponse
from tr_ru_forecast.serving.overview import build_overview
from tr_ru_forecast.serving.schemas import ErrorResponse
from tr_ru_forecast.serving.service import MirrorFilter, PageFilter, TaskFilter

router = APIRouter(responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
                              503: {"model": ErrorResponse}})


@router.get("/health", tags=["System"], responses={503: {"model": HealthResponse}})
def health(runtime: RuntimeDependency, response: Response) -> HealthResponse:
    """Return liveness and verified snapshot readiness; degraded snapshots return 503."""
    result = runtime.health((__version__, get_settings().app_env))
    response.status_code = 200 if result.status == "healthy" else 503
    response.headers["Cache-Control"] = "no-store"
    return result


@router.get("/sectors", tags=["Catalog"])
def sectors(service: ServiceDependency) -> SectorCatalog:
    """Return all selected series with source metadata and horizon assignments."""
    return service.sectors()


@router.get("/leaderboard", tags=["Models"])
@router.get("/metrics", tags=["Models"])
def leaderboard(service: ServiceDependency, filters: Annotated[TaskQuery, Query()]) -> LeaderboardResponse:
    """Return filtered CV/holdout comparisons and global unfiltered KPIs."""
    return service.leaderboard(TaskFilter(**filters.model_dump()))


@router.get("/forecasts/{sector_id}", tags=["Forecasts"])
def forecasts(sector_id: str, service: ServiceDependency, selection: Annotated[ForecastQuery, Query()]) -> ForecastResponse:
    """Return historical context and twelve retrospective targets per selected horizon."""
    return service.forecasts(sector_id, selection.horizon)


@router.get("/forecasts", tags=["Forecasts"])
def forecast_query(service: ServiceDependency, selection: Annotated[ForecastSectorQuery, Query()]) -> ForecastResponse:
    """Return forecasts through the equivalent query-form sector/horizon selection."""
    return service.forecasts(selection.sector_id, selection.horizon)


@router.get("/explainability/{sector_id}", tags=["Explainability"])
def explainability(sector_id: str, service: ServiceDependency, selection: Annotated[ExplainQuery, Query()]) -> ExplanationResponse:
    """Return global importance and local SHAP only for a registered XGBoost role."""
    return service.explain(sector_id, (selection.horizon, selection.target_month))


@router.get("/mirror-analytics", tags=["Mirror"])
@router.get("/mirror", tags=["Mirror"])
def mirror(service: ServiceDependency, selection: Annotated[MirrorQuery, Query()]) -> MirrorResponse:
    """Return exporter/importer gaps, provenance and honest post-2022 availability."""
    filters = MirrorFilter(selection.sector_id, selection.hs2, selection.direction)
    page = PageFilter(selection.offset, selection.limit, selection.start_month, selection.end_month)
    return service.mirror(filters, page)


@router.get("/trade", tags=["Trade"])
def trade(service: ServiceDependency, selection: Annotated[TradeQuery, Query()]) -> TradeResponse:
    """Return bounded official forecast-target history with source precision."""
    page = PageFilter(selection.offset, selection.limit, selection.start_month, selection.end_month)
    return service.trade(selection.sector_id, page)


@router.get("/indicators", tags=["Indicators"])
def indicators(service: ServiceDependency, selection: Annotated[IndicatorQuery, Query()]) -> IndicatorsResponse:
    """Return bounded macro observations with their release-lag metadata."""
    page = PageFilter(selection.offset, selection.limit, selection.start_month, selection.end_month)
    return service.indicators(selection.indicator_id, page)


@router.get("/methodology", tags=["Methodology"])
def methodology(service: ServiceDependency) -> MethodologyResponse:
    """Return target policy, scope limitations and provenance digests."""
    return service.methodology()


@router.get("/overview", tags=["Overview"])
def overview(service: ServiceDependency) -> OverviewResponse:
    """Return aligned TOTAL-only aggregates and global KPIs."""
    return build_overview(service.snapshot)
