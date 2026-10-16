"""Expose governed forecasts and intervals without fitting or recalibration."""

from __future__ import annotations

import pandas as pd

from tr_ru_forecast.serving.catalog_schemas import TaskComparison
from tr_ru_forecast.serving.forecast_schemas import ForecastPoint, ForecastTask


def build_forecast_task(frame: pd.DataFrame, task: TaskComparison) -> ForecastTask:
    """Project 12 target points. Args: frame, task. Returns: task. Raises: schema errors."""
    points = tuple(_build_point(group) for _, group in frame.groupby("forecast_period", sort=True))
    calibration = min(point.calibration_pool_size for point in points)
    diagnostics = task.intervals.model_copy(update={"calibration_pool_size": calibration})
    return ForecastTask(
        horizon=task.horizon, champion_model=task.champion_model, fallback_model=task.fallback_model,
        metrics=task.operational, primary_metrics=task.champion, intervals=diagnostics, points=points,
    )


def _build_point(group: pd.DataFrame) -> ForecastPoint:
    roles = group.set_index("role")
    champion, fallback = roles.loc["champion"], roles.loc["fallback"]
    served = fallback if champion.has_operational_fallback else champion
    return ForecastPoint(
        origin_date=champion.origin_date, target_date=champion.forecast_period,
        actual=champion.actual, operational_forecast=champion.operational_prediction,
        champion_forecast=champion.prediction, fallback_forecast=fallback.prediction,
        seasonal_naive_forecast=roles.loc["baseline", "prediction"],
        used_fallback=bool(champion.has_operational_fallback),
        operational_model=champion.operational_model_family,
        status="available" if pd.notna(champion.operational_prediction) else "unavailable",
        champion_is_valid=bool(champion.is_valid_prediction),
        fallback_is_valid=bool(fallback.is_valid_prediction),
        lower_80=champion.operational_lower_80, upper_80=champion.operational_upper_80,
        lower_95=champion.operational_lower_95, upper_95=champion.operational_upper_95,
        interval_policy=_get_interval_policy(served), calibration_pool_size=int(served.calibration_count),
        train_end=champion.train_end, pre_freeze_origin=bool(champion.pre_freeze_origin),
    )


def _get_interval_policy(record: pd.Series) -> str:
    if not record.is_valid_prediction or not record.interval_available:
        return "unavailable"
    return "prequential_conformal" if record.model_family == "xgboost" else "native"
