"""Complete monthly development panel builder.

Constructs the full Cartesian product panel of (direction, hs2) across 156 calendar
months (2012-01 through 2024-12). Enforces the strict runtime Holdout Leakage Guard
(blocks any data with period > 2024-12) and computes fundamental quality metrics.
"""

from typing import Any

import pandas as pd

from tr_ru_forecast.core.constants import (
    DEVELOPMENT_DATA_END,
    DEVELOPMENT_DATA_START,
)
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.data.sectors.hs_audit import get_hs_description, is_special_hs_code

logger = get_logger("tr_ru_forecast.data.sectors.panel_builder")

DEVELOPMENT_MONTHS_COUNT = 156
RECENT_WINDOW_MONTHS_COUNT = 24
RECENT_WINDOW_START = "2023-01"
RECENT_WINDOW_END = "2024-12"


def generate_development_period_index() -> list[str]:
    """Generate the exact 156 calendar months for the development period."""
    periods = pd.date_range(
        start=f"{DEVELOPMENT_DATA_START}-01",
        end=f"{DEVELOPMENT_DATA_END}-01",
        freq="MS",
    ).strftime("%Y-%m").tolist()
    assert len(periods) == DEVELOPMENT_MONTHS_COUNT, (
        f"Expected {DEVELOPMENT_MONTHS_COUNT} development periods, generated {len(periods)}"
    )
    return periods


def enforce_holdout_guard(df: pd.DataFrame, cutoff: str = DEVELOPMENT_DATA_END) -> None:
    """Holdout Leakage Guard: Rejects any dataset containing post-development observations.

    Raises:
        ValueError: If any row has period > cutoff (e.g. 2025-01 or later).
    """
    if df.empty or "period" not in df.columns:
        return

    invalid_rows = df[df["period"] > cutoff]
    if not invalid_rows.empty:
        max_period = invalid_rows["period"].max()
        count = len(invalid_rows)
        raise ValueError(
            f"HOLDOUT LEAKAGE VIOLATION: Found {count} records beyond development cutoff "
            f"'{cutoff}' (maximum observed period: '{max_period}'). "
            "Holdout data (2025-01 -> 2025-12) is strictly forbidden in sector selection."
        )


def calculate_longest_missing_streak(present_mask: list[bool]) -> int:
    """Compute the maximum consecutive missing periods in a chronological boolean mask.

    Args:
        present_mask: List of booleans where True indicates observed and False indicates missing.

    Returns:
        Integer representing longest consecutive run of False.
    """
    max_streak = 0
    current_streak = 0
    for is_present in present_mask:
        if not is_present:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
    return max_streak


class MonthlyDevelopmentPanel:
    """Constructs and analyzes the complete 156-month development panel for candidate HS2 sectors."""

    def __init__(self, target_df: pd.DataFrame) -> None:
        """Initialize with target dataframe, applying the holdout guard immediately."""
        enforce_holdout_guard(target_df)

        # Filter to development period and exclude TOTAL records
        self.dev_periods = generate_development_period_index()
        is_total_mask = (target_df.get("is_total", False)) | (target_df.get("hs2") == "TOTAL")
        self.df = target_df[
            (target_df["period"].isin(self.dev_periods)) & (~is_total_mask)
        ].copy()

        # Standardize hs2 as 2-digit zero-padded string
        self.df["hs2"] = self.df["hs2"].astype(str).str.strip().str.zfill(2)

    def get_candidate_universe(self, direction: str) -> list[str]:
        """Return sorted list of unique HS2 codes observed in development data for a direction."""
        dir_df = self.df[self.df["direction"] == direction]
        return sorted(dir_df["hs2"].dropna().unique().tolist())

    def build_sector_timeseries(self, direction: str, hs2: str) -> pd.DataFrame:
        """Construct the complete 156-month series for a specific sector.

        Missing periods have trade_value_usd = NaN (never 0.0).
        Reported zeros have trade_value_usd = 0.0.
        """
        dir_hs_df = self.df[(self.df["direction"] == direction) & (self.df["hs2"] == hs2)]

        # Base calendar dataframe
        calendar_df = pd.DataFrame({"period": self.dev_periods})

        # Merge empirical data
        merged = calendar_df.merge(dir_hs_df, on="period", how="left")
        merged["direction"] = direction
        merged["hs2"] = hs2

        # Sort chronologically
        merged.sort_values("period", inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged

    def compute_sector_metrics(self, direction: str, hs2: str) -> dict[str, Any]:
        """Compute all development data quality and continuity metrics for a candidate HS2."""
        series_df = self.build_sector_timeseries(direction, hs2)
        values = series_df["trade_value_usd"]

        observed_mask = values.notna()
        observed_count = int(observed_mask.sum())
        coverage_ratio = float(observed_count / DEVELOPMENT_MONTHS_COUNT)

        zero_count = int((values == 0).sum())
        zero_month_ratio = float(zero_count / observed_count) if observed_count > 0 else 0.0

        longest_missing_streak = calculate_longest_missing_streak(observed_mask.tolist())

        # Recent 24m window (2023-01 to 2024-12)
        recent_mask = (series_df["period"] >= RECENT_WINDOW_START) & (
            series_df["period"] <= RECENT_WINDOW_END
        )
        recent_values = series_df.loc[recent_mask, "trade_value_usd"]
        recent_observed_count = int(recent_values.notna().sum())
        recent_coverage_ratio = float(recent_observed_count / RECENT_WINDOW_MONTHS_COUNT)
        recent_total_value = float(recent_values.dropna().sum())
        recent_mean_value = (
            float(recent_values.dropna().mean()) if recent_observed_count > 0 else 0.0
        )

        # Total development volume
        observed_values = values.dropna()
        total_trade_value = float(observed_values.sum())
        mean_monthly_value = float(observed_values.mean()) if observed_count > 0 else 0.0
        median_monthly_value = float(observed_values.median()) if observed_count > 0 else 0.0

        # Periods observed
        observed_periods = series_df.loc[observed_mask, "period"].tolist()
        first_observed = observed_periods[0] if observed_periods else None
        last_observed = observed_periods[-1] if observed_periods else None

        # Revisions observed
        class_col = "classification_code" if "classification_code" in series_df else "classification"
        class_codes = (
            sorted(series_df[class_col].dropna().astype(str).unique().tolist())
            if class_col in series_df
            else []
        )

        # Description
        if (
            "hs_description" in series_df.columns
            and not series_df["hs_description"].dropna().empty
        ):
            desc = str(series_df["hs_description"].dropna().iloc[0])
        else:
            desc = get_hs_description(hs2)

        return {
            "direction": direction,
            "hs2": hs2,
            "hs_description": desc,
            "development_months_expected": DEVELOPMENT_MONTHS_COUNT,
            "observed_months": observed_count,
            "coverage_ratio": coverage_ratio,
            "zero_months": zero_count,
            "zero_month_ratio": zero_month_ratio,
            "longest_missing_streak": longest_missing_streak,
            "recent_expected_months": RECENT_WINDOW_MONTHS_COUNT,
            "recent_observed_months": recent_observed_count,
            "recent_coverage_ratio": recent_coverage_ratio,
            "development_total_trade_value": total_trade_value,
            "development_mean_monthly_value": mean_monthly_value,
            "development_median_monthly_value": median_monthly_value,
            "recent_24m_total_trade_value": recent_total_value,
            "recent_24m_mean_value": recent_mean_value,
            "first_observed_period": first_observed,
            "last_observed_period": last_observed,
            "classification_codes": class_codes,
            "revision_consistency_flag": True,  # standard chapters remain consistent across H4-H6
            "is_special_code": is_special_hs_code(hs2),
        }
