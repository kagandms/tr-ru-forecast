"""Prequential residual / conformal prediction interval calibrator for M9 XGBoost.

Implements Section 16, 17, 18, 19:
- Out-of-sample absolute residual calibration.
- Series-specific, horizon-specific, expanding prequential calibration.
- Strict residual eligibility rule: residual_target_period <= forecast_origin.
- Order statistic: k = ceil((n + 1) * (1 - alpha)).
- Minimum calibration size: n >= 19.
- No clipping of predictions or negative lower bounds.
- Full interval invariant verification.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tr_ru_forecast.models.ml.constants import MIN_CALIBRATION_SIZE


def compute_conformal_quantile(sorted_residuals: np.ndarray, alpha: float) -> float:
    """Compute order statistic quantile for nominal coverage 1 - alpha.

    Formula:
        k = ceil((n + 1) * (1 - alpha))
        quantile = sorted_residuals[k - 1] (0-indexed)
    """
    n = len(sorted_residuals)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k < 1 or k > n:
        raise ValueError(f"Order statistic k={k} out of range for calibration pool size n={n}")
    return float(sorted_residuals[k - 1])


class PrequentialResidualCalibrator:
    """Maintains historical out-of-sample prediction residuals and calibrates prequential prediction intervals."""

    def __init__(self, direction: str, hs2: str, horizon: int) -> None:
        self.direction = direction
        self.hs2 = hs2
        self.horizon = horizon
        # Store records: list of dict(origin, target_period, actual, prediction, abs_residual)
        self.history: list[dict[str, Any]] = []

    def record_historical_prediction(
        self,
        forecast_origin: str,
        forecast_period: str,
        actual: float,
        prediction: float,
    ) -> None:
        """Record an out-of-sample forecast origin result."""
        abs_res = abs(actual - prediction)
        self.history.append({
            "forecast_origin": str(forecast_origin),
            "forecast_period": str(forecast_period),
            "actual": float(actual),
            "prediction": float(prediction),
            "absolute_residual": float(abs_res),
        })

    def calibrate(
        self,
        current_origin: str,
        point_prediction: float,
        actual_val: float | None = None,
    ) -> dict[str, Any]:
        """Compute prequential calibrated prediction intervals for current_origin.

        Residual eligibility rule (Section 16):
            residual_target_period <= current_origin
        """
        # Find all historical residuals whose target period is <= current_origin
        eligible = [
            r for r in self.history
            if r["forecast_period"] <= current_origin
        ]

        n = len(eligible)

        if n < MIN_CALIBRATION_SIZE:
            # Under Section 17 & 18: intervals unavailable before n >= 19
            return {
                "interval_available": False,
                "calibration_count": n,
                "max_calibration_origin": eligible[-1]["forecast_origin"] if eligible else None,
                "max_calibration_target_period": eligible[-1]["forecast_period"] if eligible else None,
                "q80": None,
                "q95": None,
                "lower_80": None,
                "upper_80": None,
                "lower_95": None,
                "upper_95": None,
                "interval_width_80": None,
                "interval_width_95": None,
                "covered_80": None,
                "covered_95": None,
            }

        sorted_residuals = np.sort(np.array([r["absolute_residual"] for r in eligible], dtype=float))

        q80 = compute_conformal_quantile(sorted_residuals, alpha=0.20)
        q95 = compute_conformal_quantile(sorted_residuals, alpha=0.05)

        lower_80 = float(point_prediction - q80)
        upper_80 = float(point_prediction + q80)
        lower_95 = float(point_prediction - q95)
        upper_95 = float(point_prediction + q95)

        # Invariant check: lower_95 <= lower_80 <= prediction <= upper_80 <= upper_95
        if not (lower_95 <= lower_80 <= point_prediction <= upper_80 <= upper_95):
            raise ValueError(
                f"CRITICAL INTERVAL INVARIANT VIOLATED for ({self.direction}, {self.hs2}, h={self.horizon}) "
                f"at {current_origin}: lower95={lower_95}, lower80={lower_80}, pred={point_prediction}, "
                f"upper80={upper_80}, upper95={upper_95}"
            )

        max_orig = max(r["forecast_origin"] for r in eligible)
        max_target = max(r["forecast_period"] for r in eligible)

        # Audit invariant: max_calibration_target_period <= current_origin
        if max_target > current_origin:
            raise ValueError(
                f"CRITICAL CALIBRATION LEAKAGE: max target period '{max_target}' > current origin '{current_origin}'"
            )

        covered_80 = None
        covered_95 = None
        if actual_val is not None and not np.isnan(actual_val):
            covered_80 = bool(lower_80 <= actual_val <= upper_80)
            covered_95 = bool(lower_95 <= actual_val <= upper_95)

        return {
            "interval_available": True,
            "calibration_count": n,
            "max_calibration_origin": max_orig,
            "max_calibration_target_period": max_target,
            "q80": q80,
            "q95": q95,
            "lower_80": lower_80,
            "upper_80": upper_80,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "interval_width_80": upper_80 - lower_80,
            "interval_width_95": upper_95 - lower_95,
            "covered_80": covered_80,
            "covered_95": covered_95,
        }
