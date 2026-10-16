"""Serialize official history without new floating-point monetary arithmetic."""

from __future__ import annotations

import math
from decimal import Decimal

import pandas as pd

from tr_ru_forecast.serving.catalog_schemas import Sector, TaskAssignment, TaskComparison
from tr_ru_forecast.serving.forecast_schemas import HistoricalPoint


def serialize_usd(value: object) -> str | None:
    """Preserve persisted decimal value as JSON string. Args: value. Returns: USD/null. Raises: ValueError."""
    if value is None or not math.isfinite(float(str(value))):
        return None
    return format(Decimal(str(value)), "f")


def build_history(frame: pd.DataFrame) -> tuple[HistoricalPoint, ...]:
    """Project monthly source records. Args: frame. Returns: observations. Raises: validation errors."""
    columns = HistoricalPoint.model_fields
    records = frame.sort_values("period").loc[:, list(columns)].to_dict(orient="records")
    return tuple(HistoricalPoint.model_validate({
        **record, "trade_value_usd": serialize_usd(record["trade_value_usd"]),
    }) for record in records)


def build_sector(
    history: pd.DataFrame, tasks: tuple[TaskComparison, ...]
) -> Sector:
    """Build one selected sector catalog entry. Args: history, tasks. Returns: sector. Raises: schema errors."""
    first, latest = history.sort_values("period").iloc[[0, -1]].iterrows()
    start = first[1]
    end = latest[1]
    assignment = tasks[0]
    return Sector(
        sector_id=assignment.sector_id, direction=assignment.direction, hs2=assignment.hs2,
        title="Total bilateral trade" if assignment.hs2 == "TOTAL" else str(end.hs_description),
        is_total=assignment.hs2 == "TOTAL", history_start=start.period, history_end=end.period,
        latest_trade_value_usd=serialize_usd(end.trade_value_usd), source=end.source,
        reporter_code=end.reporter_code, flow=end.flow, retrieved_at=end.retrieved_at,
        tasks=tuple(TaskAssignment(horizon=task.horizon, champion_model=task.champion_model,
                                   fallback_model=task.fallback_model) for task in tasks),
    )
