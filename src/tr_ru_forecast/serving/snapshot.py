"""Immutable in-memory projection shared by read-only application services."""

from __future__ import annotations

from dataclasses import dataclass

from tr_ru_forecast.serving.catalog_schemas import GlobalKpis, Provenance, Sector, TaskComparison
from tr_ru_forecast.serving.explain_schemas import ExplanationResponse
from tr_ru_forecast.serving.forecast_schemas import ForecastResponse
from tr_ru_forecast.serving.indicator_schemas import IndicatorPoint
from tr_ru_forecast.serving.mirror_schemas import MirrorPoint


@dataclass(frozen=True)
class FrozenSnapshot:
    """Validated snapshot; all nested records are frozen and sequences are tuples."""

    sectors: tuple[Sector, ...]
    comparisons: tuple[TaskComparison, ...]
    forecasts: tuple[ForecastResponse, ...]
    explanations: tuple[ExplanationResponse, ...]
    mirrors: tuple[MirrorPoint, ...]
    indicators: tuple[IndicatorPoint, ...]
    kpis: GlobalKpis
    m10: Provenance
    m11: Provenance
    manifest_sha256: str
    verified_file_count: int
    prediction_count: int
    shap_row_count: int
