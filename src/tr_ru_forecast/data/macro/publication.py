"""Publication-awareness and release-lag calculation engine.

Implements conservative availability date computation and the publication-safe
as-of access contract to prevent future-information leakage in forecasting experiments.
"""

import pandas as pd


def compute_conservative_availability_date(
    period_str: str,
    fallback_lag_months: int,
    day_of_month: int = 1,
) -> str:
    """Compute conservative publication/availability date from reference period and release lag.

    Args:
        period_str: Reference monthly period string 'YYYY-MM'.
        fallback_lag_months: Number of months of release lag (must be >= 1).
        day_of_month: Expected day of release within the target month (e.g. 1 or 15).

    Returns:
        ISO formatted date string 'YYYY-MM-DD'.
    """
    if fallback_lag_months < 1:
        raise ValueError(
            f"Release lag must be at least 1 month. Received {fallback_lag_months}."
        )

    # Parse period as first day of month
    ref_dt = pd.to_datetime(f"{period_str}-01")
    # Add lag months
    target_dt = ref_dt + pd.DateOffset(months=fallback_lag_months)
    # Set day of month (clamping to month length if necessary)
    target_year = target_dt.year
    target_month = target_dt.month
    max_days = pd.Period(f"{target_year}-{target_month:02d}").days_in_month
    actual_day = min(day_of_month, max_days)

    return f"{target_year}-{target_month:02d}-{actual_day:02d}"


def normalize_forecast_origin(origin: str) -> str:
    """Normalize a forecast origin string to a standard ISO date 'YYYY-MM-DD'.

    If origin is 'YYYY-MM', it represents the start of that forecast origin month ('YYYY-MM-01').
    """
    origin_clean = str(origin).strip()
    if len(origin_clean) == 7:  # 'YYYY-MM'
        return f"{origin_clean}-01"
    elif len(origin_clean) == 10:  # 'YYYY-MM-DD'
        return origin_clean
    else:
        # Try parsing via pandas
        return pd.to_datetime(origin_clean).strftime("%Y-%m-%d")


class PublicationSafeAccessor:
    """Provides publication-safe as-of filtering over macroeconomic time series."""

    @staticmethod
    def filter_as_of(
        df: pd.DataFrame,
        forecast_origin: str,
        indicator_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return only macroeconomic observations known as of the forecast origin date.

        Enforces the invariant: availability_date_or_approximation <= forecast_origin.

        Args:
            df: Normalized macro DataFrame containing 'availability_date_or_approximation'.
            forecast_origin: Date or month string (e.g. '2024-06-01' or '2024-06').
            indicator_ids: Optional list of indicator IDs to filter by.

        Returns:
            Filtered DataFrame containing only publication-safe observations.
        """
        if df.empty:
            return df.copy()

        origin_iso = normalize_forecast_origin(forecast_origin)

        # Filter by indicator if specified
        sub_df = df.copy()
        if indicator_ids:
            sub_df = sub_df[sub_df["indicator_id"].isin(indicator_ids)]

        # Filter by availability date
        mask = sub_df["availability_date_or_approximation"] <= origin_iso
        safe_df = sub_df[mask].copy()

        # Invariant verification
        if not safe_df.empty:
            max_avail = safe_df["availability_date_or_approximation"].max()
            if max_avail > origin_iso:
                raise ValueError(
                    f"CRITICAL LEAKAGE INVARIANT VIOLATED: Safe view returned observation "
                    f"with availability_date '{max_avail}' > forecast_origin '{origin_iso}'."
                )

        return safe_df.reset_index(drop=True)
