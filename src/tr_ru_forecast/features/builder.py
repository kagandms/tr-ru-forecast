"""Feature Engineering Builder module for TR-RU Forecasting Platform.

Implements deterministic time-series feature engineering for the frozen 14 target series:
- Pre-feature development cutoff and holdout quarantine guard
- Forward target labels (h=1, 3, 6) with boundary cutoff enforcement
- Autoregressive target lags (1, 2, 3, 6, 12)
- Backward-only rolling statistics (mean, std with ddof=1, strictly excluding current target)
- Zero-safe growth features (MoM, YoY)
- Deterministic calendar features (month, quarter, year)
- Calendar month-end feature snapshot date computation
"""

from typing import Final

import numpy as np
import pandas as pd

from tr_ru_forecast.core.constants import (
    DEVELOPMENT_DATA_END,
    DEVELOPMENT_DATA_START,
    FORECAST_HORIZONS,
    TradeDirection,
)

# Required feature columns defined in M5 Feature Contract
REQUIRED_FEATURE_CONTRACT_COLUMNS: Final[list[str]] = [
    "period",
    "direction",
    "hs2",
    "target",
    "target_h1",
    "target_h3",
    "target_h6",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12",
    "rolling_std_3",
    "rolling_std_6",
    "rolling_std_12",
    "mom_growth",
    "yoy_growth",
    "month",
    "quarter",
    "year",
    "usd_try_lagged",
    "rub_usd_lagged",
    "brent_lagged",
    "tr_ip_lagged",
    "ru_ip_lagged",
    "feature_snapshot_date",
]

# Rolling window specifications
ROLLING_WINDOWS: Final[list[int]] = [3, 6, 12]
LAG_OFFSETS: Final[list[int]] = [1, 2, 3, 6, 12]
ROLLING_DDOF: Final[int] = 1  # Unbiased sample standard deviation (N - 1)


def compute_feature_snapshot_date(period_str: str) -> str:
    """Compute the calendar month-end date for a given YYYY-MM period.

    Args:
        period_str: Monthly period formatted as 'YYYY-MM'.

    Returns:
        ISO formatted date string 'YYYY-MM-DD' corresponding to the last day of the month.
    """
    p = pd.Period(period_str.strip(), freq="M")
    return p.end_time.strftime("%Y-%m-%d")


def enforce_development_target_guard(
    df: pd.DataFrame,
    max_period: str = DEVELOPMENT_DATA_END,
) -> pd.DataFrame:
    """Strictly quarantine and filter target observations to development period.

    Ensures that any observation beyond max_period (e.g. 2025-01 onwards)
    is completely isolated and excluded before any feature calculations or
    target labeling occur.

    Args:
        df: Input trade DataFrame.
        max_period: Upper boundary period (inclusive), default '2024-12'.

    Returns:
        Filtered DataFrame containing only rows where period <= max_period.
    """
    if "period" not in df.columns:
        raise ValueError("DataFrame must contain 'period' column.")

    # Isolate development observations strictly
    clean_df = df[df["period"] <= max_period].copy()
    return clean_df.reset_index(drop=True)


def build_complete_calendar_panel(
    df: pd.DataFrame,
    selected_manifest: dict,
    start_period: str = DEVELOPMENT_DATA_START,
    end_period: str = DEVELOPMENT_DATA_END,
) -> pd.DataFrame:
    """Construct a full, contiguous monthly panel for all 14 modeling series.

    Guarantees that each modeling series contains exactly the required monthly
    calendar observations without missing period gaps.

    Args:
        df: Filtered development target DataFrame.
        selected_manifest: Manifest dict containing selected sectors per direction.
        start_period: Start period 'YYYY-MM' (default '2012-01').
        end_period: End period 'YYYY-MM' (default '2024-12').

    Returns:
        Panel DataFrame with (direction, hs2, period) regular grid populated with target data.
    """
    periods = pd.date_range(
        start=f"{start_period}-01",
        end=f"{end_period}-01",
        freq="MS",
    ).strftime("%Y-%m").tolist()

    series_keys = []
    # Collect TR_RU sectors + TOTAL
    tr_sectors = [str(s["hs2"]).zfill(2) for s in selected_manifest.get("TR_RU", [])]
    if "TOTAL" not in tr_sectors:
        tr_sectors.append("TOTAL")
    for s in tr_sectors:
        series_keys.append((TradeDirection.TR_RU.value, s))

    # Collect RU_TR sectors + TOTAL
    ru_sectors = [str(s["hs2"]).zfill(2) if s["hs2"] != "TOTAL" else "TOTAL" for s in selected_manifest.get("RU_TR", [])]
    if "TOTAL" not in ru_sectors:
        ru_sectors.append("TOTAL")
    for s in ru_sectors:
        series_keys.append((TradeDirection.RU_TR.value, s))

    # Create full grid
    grid_rows = []
    for direction, hs2 in series_keys:
        for p in periods:
            grid_rows.append({"direction": direction, "hs2": hs2, "period": p})

    grid_df = pd.DataFrame(grid_rows)

    # Standardize hs2 in input df to match
    input_df = df.copy()
    input_df["hs2_clean"] = input_df["hs2"].apply(
        lambda x: "TOTAL" if str(x).upper() == "TOTAL" else str(x).zfill(2)
    )

    # Filter input_df to matching series
    match_mask = pd.Series(False, index=input_df.index)
    for direction, hs2 in series_keys:
        match_mask |= (input_df["direction"] == direction) & (input_df["hs2_clean"] == hs2)

    matched_input = input_df[match_mask].copy()

    # Determine description and is_total lookup
    meta_lookup = {}
    for _, row in matched_input.iterrows():
        key = (row["direction"], row["hs2_clean"])
        if key not in meta_lookup:
            meta_lookup[key] = {
                "hs_description": row.get("hs_description", "Total Bilateral Trade" if row["hs2_clean"] == "TOTAL" else ""),
                "is_total": row.get("is_total", row["hs2_clean"] == "TOTAL"),
            }

    # Merge target data onto regular grid
    merged = pd.merge(
        grid_df,
        matched_input[["direction", "hs2_clean", "period", "trade_value_usd"]],
        left_on=["direction", "hs2", "period"],
        right_on=["direction", "hs2_clean", "period"],
        how="left",
    )

    # Populate metadata
    merged["target"] = merged["trade_value_usd"]
    merged["hs_description"] = merged.apply(
        lambda r: meta_lookup.get((r["direction"], r["hs2"]), {}).get("hs_description", "Total Bilateral Trade" if r["hs2"] == "TOTAL" else ""),
        axis=1,
    )
    merged["is_total"] = merged["hs2"] == "TOTAL"

    merged = merged.drop(columns=["hs2_clean", "trade_value_usd"])
    merged = merged.sort_values(["direction", "hs2", "period"]).reset_index(drop=True)
    return merged


def compute_target_labels(
    panel_df: pd.DataFrame,
    max_dev_period: str = DEVELOPMENT_DATA_END,
) -> pd.DataFrame:
    """Compute forward target labels (target_h1, target_h3, target_h6).

    Invariant:
    - Labels must represent future values at exactly t+1, t+3, t+6 calendar months.
    - Because M5 is restricted strictly to development data (period <= max_dev_period),
      any future label requiring an observation beyond max_dev_period is assigned NaN.
    - Last valid origin for h1: 2024-11 (label = 2024-12)
    - Last valid origin for h3: 2024-09 (label = 2024-12)
    - Last valid origin for h6: 2024-06 (label = 2024-12)

    Args:
        panel_df: Regular panel DataFrame sorted by [direction, hs2, period].
        max_dev_period: Maximum development period string (default '2024-12').

    Returns:
        DataFrame with added target_h1, target_h3, target_h6 columns.
    """
    df = panel_df.copy()

    for h in FORECAST_HORIZONS:
        label_col = f"target_h{h}"
        # Shift -h within group
        df[label_col] = df.groupby(["direction", "hs2"])["target"].shift(-h)

        # Calendar check to ensure shift(-h) matches exactly t+h months
        # and enforce that if t+h exceeds max_dev_period, label is NaN
        for idx, row in df.iterrows():
            current_period = row["period"]
            # Target future period
            p_dt = pd.Period(current_period, freq="M")
            future_p_dt = p_dt + h
            future_period_str = str(future_p_dt)

            if future_period_str > max_dev_period:
                df.at[idx, label_col] = np.nan

    return df


def compute_autoregressive_lags(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute autoregressive target lags (lag_1, lag_2, lag_3, lag_6, lag_12).

    All lags are computed strictly per (direction, hs2) series independently.
    No cross-series leakage is permitted.

    Args:
        panel_df: Regular panel DataFrame sorted by [direction, hs2, period].

    Returns:
        DataFrame with added lag columns.
    """
    df = panel_df.copy()
    grouped = df.groupby(["direction", "hs2"])["target"]

    for lag in LAG_OFFSETS:
        col_name = f"lag_{lag}"
        df[col_name] = grouped.shift(lag)

    return df


def compute_backward_rolling_features(
    panel_df: pd.DataFrame,
    ddof: int = ROLLING_DDOF,
) -> pd.DataFrame:
    """Compute backward-only rolling statistics (mean and std for windows 3, 6, 12).

    CRITICAL INVARIANTS:
    - The rolling window must strictly use observations BEFORE t (i.e. y_{t-1}, y_{t-2}, ...).
    - Current target y_t is strictly EXCLUDED.
    - Conceptually: target.shift(1).rolling(window, min_periods=window).
    - Windows require complete observations (min_periods = window).
    - Sample standard deviation uses ddof=1 (unbiased).

    Args:
        panel_df: Regular panel DataFrame sorted by [direction, hs2, period].
        ddof: Degrees of freedom for sample std (default 1).

    Returns:
        DataFrame with added rolling_mean_* and rolling_std_* columns.
    """
    df = panel_df.copy()

    # Apply shift(1) within group to strictly exclude current period y_t
    for w in ROLLING_WINDOWS:
        mean_col = f"rolling_mean_{w}"
        std_col = f"rolling_std_{w}"

        # Groupby shifted target to compute rolling stats per series
        df[mean_col] = df.groupby(["direction", "hs2"])["target"].transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=w).mean()
        )
        df[std_col] = df.groupby(["direction", "hs2"])["target"].transform(
            lambda s: s.shift(1).rolling(window=w, min_periods=w).std(ddof=ddof)
        )

    return df


def compute_growth_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Month-over-Month (MoM) and Year-over-Year (YoY) target growth rates.

    Formulas:
        mom_growth(t) = y_t / y_{t-1} - 1
        yoy_growth(t) = y_t / y_{t-12} - 1

    Safety rules:
        - If numerator (y_t) is missing -> NaN
        - If denominator (y_{t-1} or y_{t-12}) is missing or 0 -> NaN
        - Never emit inf or -inf
        - No clipping or winsorizing

    Args:
        panel_df: DataFrame with target, lag_1, and lag_12 columns.

    Returns:
        DataFrame with added mom_growth and yoy_growth columns.
    """
    df = panel_df.copy()

    # MoM Growth
    denom_mom = df["lag_1"].replace(0.0, np.nan)
    mom = (df["target"] / denom_mom) - 1.0
    mom = mom.replace([np.inf, -np.inf], np.nan)
    df["mom_growth"] = mom

    # YoY Growth
    denom_yoy = df["lag_12"].replace(0.0, np.nan)
    yoy = (df["target"] / denom_yoy) - 1.0
    yoy = yoy.replace([np.inf, -np.inf], np.nan)
    df["yoy_growth"] = yoy

    return df


def compute_calendar_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute calendar features (month, quarter, year) deterministically from period.

    Args:
        panel_df: DataFrame containing 'period' formatted as 'YYYY-MM'.

    Returns:
        DataFrame with integer columns month (1-12), quarter (1-4), year.
    """
    df = panel_df.copy()
    dt_series = pd.to_datetime(df["period"] + "-01")

    df["month"] = dt_series.dt.month.astype(int)
    df["quarter"] = dt_series.dt.quarter.astype(int)
    df["year"] = dt_series.dt.year.astype(int)
    df["feature_snapshot_date"] = df["period"].apply(compute_feature_snapshot_date)

    return df
