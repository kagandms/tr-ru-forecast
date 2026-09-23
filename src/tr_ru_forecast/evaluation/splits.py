"""Expanding walk-forward validation fold generator and registry.

Generates deterministic monthly rolling-origin backtest folds for the TR-RU forecasting platform:
- Minimum initial training history: 60 months (2012-01 to 2016-12 -> first origin 2016-12).
- Expanding training window: train_start is frozen at 2012-01, train_end = forecast_origin.
- Horizon-specific origin bounds:
    h=1: last origin = 2024-11 (forecast_period = 2024-12, 96 origins)
    h=3: last origin = 2024-09 (forecast_period = 2024-12, 94 origins)
    h=6: last origin = 2024-06 (forecast_period = 2024-12, 91 origins)
- Total shared validation folds: 281 folds.
- Reusable across M6 Seasonal Naive, M7 SARIMA, M8 SARIMAX, and M9 XGBoost.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
import yaml

from tr_ru_forecast.core.constants import (
    DEVELOPMENT_DATA_END,
    DEVELOPMENT_DATA_START,
    FORECAST_HORIZONS,
)
from tr_ru_forecast.core.logging import get_logger

logger = get_logger(__name__)

FOLD_VERSION: Final[str] = "1.1.0"
SPEC_VERSION: Final[str] = "1.1"
MIN_INITIAL_HISTORY_MONTHS: Final[int] = 60

HORIZON_TERMINAL_ORIGINS: Final[dict[int, str]] = {
    1: "2024-11",
    3: "2024-09",
    6: "2024-06",
}


def derive_first_validation_origin(
    dev_start: str = DEVELOPMENT_DATA_START,
    min_history_months: int = MIN_INITIAL_HISTORY_MONTHS,
) -> str:
    """Derive the earliest validation origin that provides at least min_history_months.

    Given complete monthly records starting at dev_start, the earliest origin month t
    such that observations from dev_start through t inclusive count >= min_history_months
    is: dev_start + (min_history_months - 1) months.

    For dev_start='2012-01' and min_history=60:
    2012-01 + 59 months = 2016-12 (exactly 60 months: 2012-01 through 2016-12 inclusive).
    """
    start_p = pd.Period(dev_start.strip(), freq="M")
    first_origin_p = start_p + (min_history_months - 1)
    return str(first_origin_p)


def generate_validation_folds(
    start_period: str = DEVELOPMENT_DATA_START,
    end_period: str = DEVELOPMENT_DATA_END,
    min_history_months: int = MIN_INITIAL_HISTORY_MONTHS,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Generate the canonical expanding walk-forward validation fold table.

    Returns:
        DataFrame with columns:
            [fold_id, horizon, forecast_origin, train_start, train_end,
             forecast_period, seasonal_source_period]
    """
    if horizons is None:
        horizons = FORECAST_HORIZONS

    first_origin = derive_first_validation_origin(start_period, min_history_months)
    fold_rows = []

    for h in horizons:
        last_origin = HORIZON_TERMINAL_ORIGINS.get(h)
        if not last_origin:
            # Fallback calendar calculation if horizon not in predefined map
            end_p = pd.Period(end_period, freq="M")
            last_origin = str(end_p - h)

        origins = pd.date_range(
            f"{first_origin}-01",
            f"{last_origin}-01",
            freq="MS",
        ).strftime("%Y-%m").tolist()

        for idx, origin in enumerate(origins, start=1):
            origin_p = pd.Period(origin, freq="M")
            target_p = origin_p + h
            source_p = target_p - 12

            # Assert origin safety invariant: seasonal_source_period <= forecast_origin
            if str(source_p) > origin:
                raise ValueError(
                    f"CRITICAL LEAKAGE INVARIANT VIOLATED: seasonal_source_period '{source_p}' "
                    f"> forecast_origin '{origin}' for horizon {h}."
                )

            # Assert holdout safety: forecast_period <= end_period
            if str(target_p) > end_period:
                raise ValueError(
                    f"CRITICAL HOLDOUT INVARIANT VIOLATED: forecast_period '{target_p}' "
                    f"> development end '{end_period}'."
                )

            fold_id = f"fold_h{h}_{origin}"

            fold_rows.append({
                "fold_id": fold_id,
                "horizon": int(h),
                "forecast_origin": str(origin),
                "train_start": str(start_period),
                "train_end": str(origin),
                "forecast_period": str(target_p),
                "seasonal_source_period": str(source_p),
            })

    folds_df = pd.DataFrame(fold_rows)
    return folds_df


def build_validation_manifest(
    folds_df: pd.DataFrame,
    start_period: str = DEVELOPMENT_DATA_START,
    end_period: str = DEVELOPMENT_DATA_END,
    min_history_months: int = MIN_INITIAL_HISTORY_MONTHS,
) -> dict[str, Any]:
    """Build machine-readable validation fold registry manifest."""
    first_origin = derive_first_validation_origin(start_period, min_history_months)

    horizon_summaries = {}
    for h, group in folds_df.groupby("horizon"):
        h_int = int(h)
        horizon_summaries[f"h{h_int}"] = {
            "horizon_months": h_int,
            "first_origin": str(group["forecast_origin"].min()),
            "last_origin": str(group["forecast_origin"].max()),
            "number_of_origins": len(group),
            "first_forecast_period": str(group["forecast_period"].min()),
            "last_forecast_period": str(group["forecast_period"].max()),
        }

    manifest = {
        "spec_version": SPEC_VERSION,
        "fold_version": FOLD_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "development_start": str(start_period),
        "development_end": str(end_period),
        "minimum_initial_history_months": int(min_history_months),
        "first_validation_origin": str(first_origin),
        "origin_frequency": "monthly",
        "total_folds": len(folds_df),
        "horizons": horizon_summaries,
        "validation_strategy": "expanding_walk_forward_rolling_origin",
        "invariants": {
            "source_before_origin": "seasonal_source_period <= forecast_origin strictly holds",
            "holdout_isolated": "forecast_period <= development_end strictly holds",
            "train_start_fixed": "train_start is fixed at development_start (2012-01)",
            "train_end_expanding": "train_end equals forecast_origin and expands monthly",
        },
    }
    return manifest


def save_validation_registry(
    folds_df: pd.DataFrame,
    yaml_path: Path,
    parquet_path: Path,
) -> dict[str, Any]:
    """Persist both YAML manifest and row-level parquet fold table."""
    manifest = build_validation_manifest(folds_df)

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    logger.info("Saved validation fold manifest to %s", yaml_path)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    folds_df.to_parquet(parquet_path, index=False)
    logger.info("Saved validation origins table to %s (%d folds)", parquet_path, len(folds_df))

    return manifest
