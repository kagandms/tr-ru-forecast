"""Historical observations and retrospective forecast explorer contracts."""

from __future__ import annotations

from datetime import date, datetime
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


class HistoricalPoint(FrozenModel):
    """Immutable historical point contract."""

    period: Month
    trade_value_usd: str | None
    data_status: str
    source: str
    reporter_code: int
    partner_code: int
    flow: str
    retrieved_at: datetime


class ForecastPoint(FrozenModel):
    """Immutable forecast point contract."""

    origin_date: date
    target_date: Month
    actual: Number
    operational_forecast: Number
    champion_forecast: Number
    fallback_forecast: Number
    seasonal_naive_forecast: Number
    used_fallback: bool
    operational_model: ModelFamily | None
    status: Literal["available", "unavailable"]
    champion_is_valid: bool
    fallback_is_valid: bool
    lower_80: Number
    upper_80: Number
    lower_95: Number
    upper_95: Number
    interval_policy: Literal["native", "prequential_conformal", "unavailable"]
    calibration_pool_size: int
    train_end: Month
    pre_freeze_origin: bool


class ForecastTask(FrozenModel):
    """Immutable forecast task contract."""

    horizon: Horizon
    champion_model: ModelFamily
    fallback_model: ModelFamily
    metrics: MetricSummary
    primary_metrics: MetricSummary
    intervals: IntervalDiagnostics
    points: tuple[ForecastPoint, ...]


class ForecastResponse(FrozenModel):
    """Immutable forecast response contract."""

    sector_id: str
    direction: Direction
    hs2: str
    evaluation_scope: str
    forecast_kind: Literal["historical_holdout"] = "historical_holdout"
    unit: Literal["USD"] = "USD"
    observation_cutoff: Month
    history: tuple[HistoricalPoint, ...]
    tasks: tuple[ForecastTask, ...]


class TradeResponse(FrozenModel):
    """Immutable trade response contract."""

    sector_id: str
    observations: tuple[HistoricalPoint, ...]
    total_observations: int
    offset: int
    limit: int
    unit: Literal["USD"] = "USD"
