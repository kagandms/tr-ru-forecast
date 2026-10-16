"""Macro, methodology and overview contracts required by the milestone semantics."""

from __future__ import annotations

from datetime import datetime

from tr_ru_forecast.serving.catalog_schemas import GlobalKpis
from tr_ru_forecast.serving.schemas import FrozenModel, Month, Number


class IndicatorPoint(FrozenModel):
    """Immutable indicator point contract."""

    indicator_id: str
    indicator_name: str
    period: Month
    value: Number
    unit: str
    source_provider: str
    source_series_id: str
    availability_date_or_approximation: str
    availability_is_exact: bool
    release_lag_rule: str
    retrieved_at: datetime


class IndicatorsResponse(FrozenModel):
    """Immutable indicators response contract."""

    evaluation_scope: str
    observations: tuple[IndicatorPoint, ...]
    total_observations: int
    offset: int
    limit: int


class MethodologyResponse(FrozenModel):
    """Immutable methodology response contract."""

    specification_version: str
    evaluation_scope: str
    forecast_kind: str
    target_policy: tuple[str, ...]
    limitations: tuple[str, ...]
    unit: str
    monetary_serialization: str
    interval_coverage_unit: str
    m10_run_digest: str
    m11_run_digest: str


class OverviewResponse(FrozenModel):
    """Immutable overview response contract."""

    evaluation_scope: str
    observation_cutoff: Month
    summary: GlobalKpis
    tr_ru_total_usd: str | None
    ru_tr_total_usd: str | None
    bilateral_total_usd: str | None
    turkey_trade_balance_usd: str | None
