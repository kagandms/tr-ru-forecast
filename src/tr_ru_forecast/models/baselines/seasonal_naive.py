"""Seasonal Naive baseline forecasting model.

Implements the authoritative frozen benchmark rule:
    Forecast(t+h) = Actual(t+h-12)

for h in {1, 3, 6}, where:
    t = forecast origin month
    t+h = forecast target month
    t+h-12 = seasonal source month

Invariants:
- Origin safety: (t+h-12) <= t strictly holds.
- Series isolation: Predictions only query historical observations of the exact same (direction, hs2) series.
- Pure historical benchmark: No exogenous macro variables, no lag features, no rolling statistics are used.
"""

from typing import Any

import numpy as np
import pandas as pd


class SeasonalNaivePredictor:
    """Deterministic Seasonal Naive predictor using exact 12-month calendar lag."""

    def __init__(self, target_panel_df: pd.DataFrame) -> None:
        """Initialize predictor with historical development target panel.

        Args:
            target_panel_df: DataFrame containing at minimum ['direction', 'hs2', 'period', 'target'].
        """
        # Build lookup table: (direction, hs2, period) -> target_value
        self._lookup: dict[tuple[str, str, str], float] = {}

        for _, row in target_panel_df.iterrows():
            dir_val = str(row["direction"]).strip()
            hs2_val = str(row["hs2"]).strip()
            period_val = str(row["period"]).strip()
            target_val = float(row["target"]) if pd.notnull(row["target"]) else np.nan

            self._lookup[(dir_val, hs2_val, period_val)] = target_val

    def predict_one(
        self,
        direction: str,
        hs2: str,
        forecast_origin: str,
        horizon: int,
    ) -> dict[str, Any]:
        """Generate a single Seasonal Naive prediction.

        Args:
            direction: Bilateral direction ('TR_RU' or 'RU_TR').
            hs2: HS2 chapter code or 'TOTAL'.
            forecast_origin: Anchor month string 'YYYY-MM'.
            horizon: Step horizon (1, 3, or 6).

        Returns:
            Dict containing forecast_period, seasonal_source_period, seasonal_source_value, prediction.
        """
        origin_p = pd.Period(forecast_origin.strip(), freq="M")
        forecast_p = origin_p + horizon
        source_p = forecast_p - 12

        forecast_period_str = str(forecast_p)
        source_period_str = str(source_p)

        # Invariant check
        if source_period_str > forecast_origin:
            raise ValueError(
                f"LEAKAGE VIOLATION: seasonal_source_period {source_period_str} > origin {forecast_origin}"
            )

        key = (direction.strip(), hs2.strip(), source_period_str)
        source_value = self._lookup.get(key, np.nan)

        return {
            "direction": direction,
            "hs2": hs2,
            "forecast_origin": forecast_origin,
            "horizon": horizon,
            "forecast_period": forecast_period_str,
            "seasonal_source_period": source_period_str,
            "seasonal_source_value": source_value,
            "prediction": source_value,
        }

    def predict_batch(
        self,
        folds_df: pd.DataFrame,
        series_list: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Generate predictions for all folds and all series.

        Args:
            folds_df: Validation fold DataFrame from splits.py.
            series_list: List of dicts specifying target series ({direction, hs2, is_total}).

        Returns:
            DataFrame of row-level predictions.
        """
        records = []

        for s in series_list:
            dir_name = s["direction"]
            hs2_code = s["hs2"]
            is_tot = s.get("is_total", hs2_code == "TOTAL")

            for _, fold in folds_df.iterrows():
                fold_id = fold["fold_id"]
                h = int(fold["horizon"])
                origin = fold["forecast_origin"]
                train_start = fold["train_start"]
                train_end = fold["train_end"]
                forecast_p = fold["forecast_period"]
                source_p = fold["seasonal_source_period"]

                # Seasonal Naive prediction
                key_source = (dir_name, hs2_code, source_p)
                source_val = self._lookup.get(key_source, np.nan)

                # Actual value at forecast period (for validation scoring only)
                key_actual = (dir_name, hs2_code, forecast_p)
                actual_val = self._lookup.get(key_actual, np.nan)

                records.append({
                    "fold_id": fold_id,
                    "direction": dir_name,
                    "hs2": hs2_code,
                    "is_total": is_tot,
                    "horizon": h,
                    "forecast_origin": origin,
                    "train_start": train_start,
                    "train_end": train_end,
                    "forecast_period": forecast_p,
                    "seasonal_source_period": source_p,
                    "seasonal_source_value": source_val,
                    "actual_value": actual_val,
                    "prediction": source_val,
                    "model_type": "seasonal_naive",
                })

        return pd.DataFrame(records)
