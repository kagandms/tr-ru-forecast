"""Origin-safe prequential residual calibration, retaining the M9 numeric contract."""

from __future__ import annotations

import math

import numpy as np

from tr_ru_forecast.models.ml.constants import MIN_CALIBRATION_SIZE
from tr_ru_forecast.models.ml.records import JsonObject, ResidualRecord


def compute_conformal_quantile(sorted_residuals: np.ndarray, alpha: float) -> float:
    """Return ceil((n+1)*(1-alpha)) order statistic. Args: residuals, alpha. Raises: ValueError."""
    count = len(sorted_residuals)
    index = math.ceil((count + 1) * (1.0 - alpha))
    if index < 1 or index > count:
        raise ValueError(
            f"Order statistic k={index} out of range for calibration pool size n={count}"
        )
    return float(sorted_residuals[index - 1])


def _build_calibration_audit(eligible: list[ResidualRecord]) -> JsonObject:
    return {
        "interval_available": len(eligible) >= MIN_CALIBRATION_SIZE,
        "calibration_count": len(eligible),
        "max_calibration_origin": max(row["forecast_origin"] for row in eligible)
        if eligible
        else None,
        "max_calibration_target_period": max(row["forecast_period"] for row in eligible)
        if eligible
        else None,
    }


def _build_interval_fields(
    prediction: float, quantiles: tuple[float, float], actual: float | None
) -> JsonObject:
    q80, q95 = quantiles
    lower_80, upper_80 = float(prediction - q80), float(prediction + q80)
    lower_95, upper_95 = float(prediction - q95), float(prediction + q95)
    if not lower_95 <= lower_80 <= prediction <= upper_80 <= upper_95:
        raise ValueError("CRITICAL INTERVAL INVARIANT VIOLATED")
    has_actual = actual is not None and not np.isnan(actual)
    return {
        "q80": q80,
        "q95": q95,
        "lower_80": lower_80,
        "upper_80": upper_80,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "interval_width_80": upper_80 - lower_80,
        "interval_width_95": upper_95 - lower_95,
        "covered_80": bool(lower_80 <= actual <= upper_80) if has_actual else None,
        "covered_95": bool(lower_95 <= actual <= upper_95) if has_actual else None,
    }


class PrequentialResidualCalibrator:
    """Store OOS evidence; calibrate only target months realized by the origin."""

    def __init__(self, direction: str, hs2: str, horizon: int) -> None:
        self.direction, self.hs2, self.horizon = direction, hs2, horizon
        self.history: list[ResidualRecord] = []

    def record_historical_prediction(
        self, forecast_origin: str, forecast_period: str, actual: float, prediction: float
    ) -> None:
        """Append evidence. Args: origin, target, actual, prediction. Returns: None. Raises: none."""
        self.history.append(
            {
                "forecast_origin": str(forecast_origin),
                "forecast_period": str(forecast_period),
                "actual": float(actual),
                "prediction": float(prediction),
                "absolute_residual": float(abs(actual - prediction)),
            }
        )

    def calibrate(
        self, current_origin: str, point_prediction: float, actual_val: float | None = None
    ) -> JsonObject:
        """Return calibrated bounds and audit. Args: origin, prediction, actual. Raises: ValueError."""
        eligible = [row for row in self.history if row["forecast_period"] <= current_origin]
        audit = _build_calibration_audit(eligible)
        if len(eligible) < MIN_CALIBRATION_SIZE:
            unavailable = dict.fromkeys(
                (
                    "q80",
                    "q95",
                    "lower_80",
                    "upper_80",
                    "lower_95",
                    "upper_95",
                    "interval_width_80",
                    "interval_width_95",
                    "covered_80",
                    "covered_95",
                )
            )
            return {**audit, **unavailable}
        residuals = np.sort(np.array([row["absolute_residual"] for row in eligible], dtype=float))
        quantiles = (
            compute_conformal_quantile(residuals, 0.20),
            compute_conformal_quantile(residuals, 0.05),
        )
        return {**audit, **_build_interval_fields(point_prediction, quantiles, actual_val)}
