"""Publication-Safe Macroeconomic Feature Alignment Module.

Implements as-of joining of macroeconomic indicators onto monthly trade feature rows
in strict adherence to the M4 publication-safety contract:
- The information cutoff for each trade row period t is feature_snapshot_date (calendar month-end).
- Uses PublicationSafeAccessor.filter_as_of to prevent future-information leakage.
- Selects the latest available macro observation as of feature_snapshot_date.
- Produces macro feature columns: usd_try_lagged, rub_usd_lagged, brent_lagged, tr_ip_lagged, ru_ip_lagged.
- Generates detailed, machine-readable macro alignment audit traces.
"""

from typing import Any, Final

import numpy as np
import pandas as pd

from tr_ru_forecast.data.macro.publication import PublicationSafeAccessor
from tr_ru_forecast.data.macro.registry import REQUIRED_INDICATOR_IDS

# Mapping from indicator_id to feature column name
MACRO_FEATURE_COLUMN_MAP: Final[dict[str, str]] = {
    "usd_try": "usd_try_lagged",
    "rub_usd": "rub_usd_lagged",
    "brent": "brent_lagged",
    "tr_ip": "tr_ip_lagged",
    "ru_ip": "ru_ip_lagged",
}


def align_publication_safe_macro(
    panel_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Align publication-safe macroeconomic indicators with monthly trade panel rows.

    For each unique trade period t:
    1. feature_snapshot_date = month-end date of period t.
    2. Filters macro_df using PublicationSafeAccessor.filter_as_of(macro_df, forecast_origin=feature_snapshot_date).
    3. For each of the 5 required macro indicators, extracts the most recent available observation.
    4. Merges aligned macro values into the panel DataFrame.
    5. Compiles an exhaustive audit trace recording the selected macro reference period and availability date.

    Args:
        panel_df: Feature panel DataFrame containing 'period' and 'feature_snapshot_date'.
        macro_df: Processed macroeconomic DataFrame conforming to M4 standard.

    Returns:
        tuple of:
            - DataFrame with added macro feature columns (usd_try_lagged, etc.).
            - List of audit dicts detailing indicator alignment for each period.
    """
    if "period" not in panel_df.columns or "feature_snapshot_date" not in panel_df.columns:
        raise ValueError("panel_df must contain 'period' and 'feature_snapshot_date' columns.")

    # Unique periods and snapshot dates
    period_snapshots = (
        panel_df[["period", "feature_snapshot_date"]]
        .drop_duplicates()
        .sort_values("period")
        .to_dict(orient="records")
    )

    aligned_records: list[dict[str, Any]] = []
    audit_trace: list[dict[str, Any]] = []

    for entry in period_snapshots:
        period_str = entry["period"]
        snapshot_date = entry["feature_snapshot_date"]

        # Filter macro observations known on or before snapshot_date
        safe_macro = PublicationSafeAccessor.filter_as_of(
            macro_df,
            forecast_origin=snapshot_date,
        )

        row_dict: dict[str, Any] = {"period": period_str}

        for indicator_id in REQUIRED_INDICATOR_IDS:
            feature_col = MACRO_FEATURE_COLUMN_MAP[indicator_id]
            ind_sub = safe_macro[safe_macro["indicator_id"] == indicator_id]

            if not ind_sub.empty:
                # Latest available observation by reference period
                latest_row = ind_sub.sort_values("period").iloc[-1]
                val = float(latest_row["value"]) if pd.notnull(latest_row["value"]) else np.nan
                ref_period = str(latest_row["period"])
                avail_date = str(latest_row["availability_date_or_approximation"])

                row_dict[feature_col] = val
                audit_trace.append({
                    "trade_period": period_str,
                    "feature_snapshot_date": snapshot_date,
                    "indicator_id": indicator_id,
                    "feature_column": feature_col,
                    "selected_macro_reference_period": ref_period,
                    "selected_macro_availability_date": avail_date,
                    "selected_macro_value": val,
                    "is_available": True,
                })
            else:
                row_dict[feature_col] = np.nan
                audit_trace.append({
                    "trade_period": period_str,
                    "feature_snapshot_date": snapshot_date,
                    "indicator_id": indicator_id,
                    "feature_column": feature_col,
                    "selected_macro_reference_period": None,
                    "selected_macro_availability_date": None,
                    "selected_macro_value": None,
                    "is_available": False,
                })

        aligned_records.append(row_dict)

    macro_aligned_df = pd.DataFrame(aligned_records)

    # Merge aligned macro values into panel_df
    merged_df = pd.merge(panel_df, macro_aligned_df, on="period", how="left")

    return merged_df, audit_trace
