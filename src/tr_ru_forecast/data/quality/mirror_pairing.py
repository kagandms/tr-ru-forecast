"""Bilateral mirror pairing and data quality gap engine.

Performs deterministic full outer join between canonical exporter-reported trade flows
and mirror counterparty importer-reported trade flows, computing quality metrics,
gaps, classification mismatches, and status flags in strict accordance with DATA_SPEC.md.

CRITICAL RULE:
Canonical trade data is strictly immutable and is NEVER overwritten or imputed
by mirror data.
"""

from typing import Any

import pandas as pd

from tr_ru_forecast.core.constants import (
    MIRROR_MONTHLY_COLUMNS,
    GapFlag,
    PairStatus,
    TradeDirection,
)
from tr_ru_forecast.core.logging import get_logger

logger = get_logger("tr_ru_forecast.data.quality.mirror_pairing")


def calculate_mirror_gap(
    canonical_val: float | None,
    mirror_val: float | None,
    thresholds: tuple[float, float] | None = None,
) -> tuple[float | None, float | None, float | None, GapFlag]:
    """Calculate absolute, percentage, and absolute-percentage mirror discrepancies safely.

    Definitions:
        mirror_gap_usd = canonical_value_usd - mirror_value_usd
        mirror_gap_pct = ((canonical_value_usd - mirror_value_usd) / canonical_value_usd) * 100
        abs_mirror_gap_pct = abs(mirror_gap_pct)

    Default Gap Flag Policy (V1 M2 Revision Gate):
        - Neutral data presence states: PAIRED (both observed) vs UNPAIRED (one or both missing).
        - Normative flags (NONE, MODERATE, LARGE) are disabled by default and only evaluated
          if explicit numerical thresholds are configured.

    Returns:
        tuple (gap_usd, gap_pct, abs_gap_pct, gap_flag)
    """
    if canonical_val is None or pd.isna(canonical_val) or mirror_val is None or pd.isna(mirror_val):
        return None, None, None, GapFlag.UNPAIRED

    c_val = float(canonical_val)
    m_val = float(mirror_val)

    gap_usd = c_val - m_val

    if c_val == 0.0:
        gap_pct = None
        abs_gap_pct = None
        if thresholds is not None:
            gap_flag = GapFlag.NONE if m_val == 0.0 else GapFlag.LARGE
        else:
            gap_flag = GapFlag.PAIRED
        return gap_usd, gap_pct, abs_gap_pct, gap_flag

    gap_pct = (gap_usd / c_val) * 100.0
    abs_gap_pct = abs(gap_pct)

    if thresholds is not None:
        low_th, high_th = thresholds
        if abs_gap_pct <= low_th:
            gap_flag = GapFlag.NONE
        elif abs_gap_pct <= high_th:
            gap_flag = GapFlag.MODERATE
        else:
            gap_flag = GapFlag.LARGE
    else:
        gap_flag = GapFlag.PAIRED

    return gap_usd, gap_pct, abs_gap_pct, gap_flag


def pair_canonical_and_mirror(
    canonical_df: pd.DataFrame,
    mirror_df: pd.DataFrame,
    direction: TradeDirection | str | None = None,
    gap_thresholds: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Pair canonical and mirror trade datasets on (period, direction, hs2).

    Guarantees:
        1. Canonical values are never overwritten or altered.
        2. Full outer join captures canonical-only, mirror-only, and paired observations.
        3. Strict schema conformance to MIRROR_MONTHLY_COLUMNS.
        4. Safe handling of zero, nulls, and classification code differences.

    Args:
        canonical_df: Standardized canonical DataFrame (flow=Export).
        mirror_df: Standardized mirror DataFrame (flow=Import, direction aligned with physical trade).
        direction: Optional filter to restrict pairing to a single trade direction.
        gap_thresholds: Optional numerical thresholds for normative gap flagging (disabled by default).

    Returns:
        Paired DataFrame conforming to MIRROR_MONTHLY_COLUMNS.
    """
    # Create defensive copies to ensure immutability of inputs
    c_df = canonical_df.copy() if not canonical_df.empty else pd.DataFrame()
    m_df = mirror_df.copy() if not mirror_df.empty else pd.DataFrame()

    dir_str = (
        str(direction.value if isinstance(direction, TradeDirection) else direction)
        if direction
        else None
    )

    if dir_str:
        if not c_df.empty and "direction" in c_df.columns:
            c_df = c_df[c_df["direction"] == dir_str].copy()
        if not m_df.empty and "direction" in m_df.columns:
            m_df = m_df[m_df["direction"] == dir_str].copy()

    # Define join keys
    join_keys = ["period", "direction", "hs2"]

    # Select necessary columns from canonical
    canonical_cols = [
        "period",
        "direction",
        "hs2",
        "trade_value_usd",
        "classification",
        "classification_code",
        "is_original_classification",
        "retrieved_at",
    ]
    mirror_cols = [
        "period",
        "direction",
        "hs2",
        "trade_value_usd",
        "classification",
        "classification_code",
        "is_original_classification",
        "retrieved_at",
    ]

    c_subset = (
        c_df[[col for col in canonical_cols if col in c_df.columns]].copy()
        if not c_df.empty
        else pd.DataFrame(columns=canonical_cols)
    )
    m_subset = (
        m_df[[col for col in mirror_cols if col in m_df.columns]].copy()
        if not m_df.empty
        else pd.DataFrame(columns=mirror_cols)
    )

    merged = pd.merge(
        c_subset,
        m_subset,
        on=join_keys,
        how="outer",
        suffixes=("_canonical", "_mirror"),
    )

    if merged.empty:
        return pd.DataFrame(columns=MIRROR_MONTHLY_COLUMNS)

    paired_records: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        c_val = row.get("trade_value_usd_canonical")
        m_val = row.get("trade_value_usd_mirror")

        c_missing = pd.isna(c_val) or c_val is None
        m_missing = pd.isna(m_val) or m_val is None

        # Resolve pair status
        if not c_missing and not m_missing:
            status = PairStatus.BOTH_PRESENT
        elif not c_missing and m_missing:
            status = PairStatus.CANONICAL_ONLY
        elif c_missing and not m_missing:
            status = PairStatus.MIRROR_ONLY
        else:
            status = PairStatus.BOTH_MISSING

        # Resolve trade values (float or None)
        canonical_val_usd = float(c_val) if not c_missing else None
        mirror_val_usd = float(m_val) if not m_missing else None

        # Calculate gaps and flag
        gap_usd, gap_pct, abs_gap_pct, gap_flag = calculate_mirror_gap(
            canonical_val_usd, mirror_val_usd, thresholds=gap_thresholds
        )

        # Classification metadata
        c_class = row.get("classification_canonical")
        c_code = row.get("classification_code_canonical")
        c_orig = row.get("is_original_classification_canonical")

        m_class = row.get("classification_mirror")
        m_code = row.get("classification_code_mirror")
        m_orig = row.get("is_original_classification_mirror")

        # Classification mismatch detection
        if status == PairStatus.BOTH_PRESENT:
            # If both have classification codes, compare them
            if pd.notna(c_code) and pd.notna(m_code):
                mismatch = bool(str(c_code).strip() != str(m_code).strip())
            elif pd.isna(c_code) and pd.isna(m_code):
                mismatch = False
            else:
                mismatch = True
        else:
            mismatch = False

        c_retrieved = row.get("retrieved_at_canonical")
        m_retrieved = row.get("retrieved_at_mirror")

        record = {
            "period": str(row["period"]),
            "direction": str(row["direction"]),
            "hs2": str(row["hs2"]),
            "canonical_value_usd": canonical_val_usd,
            "mirror_value_usd": mirror_val_usd,
            "mirror_gap_usd": gap_usd,
            "mirror_gap_pct": gap_pct,
            "abs_mirror_gap_pct": abs_gap_pct,
            "canonical_missing": c_missing,
            "mirror_missing": m_missing,
            "pair_status": status.value,
            "canonical_classification": str(c_class) if pd.notna(c_class) else None,
            "canonical_classification_code": str(c_code) if pd.notna(c_code) else None,
            "canonical_is_original_classification": bool(c_orig) if pd.notna(c_orig) else None,
            "mirror_classification": str(m_class) if pd.notna(m_class) else None,
            "mirror_classification_code": str(m_code) if pd.notna(m_code) else None,
            "mirror_is_original_classification": bool(m_orig) if pd.notna(m_orig) else None,
            "classification_mismatch": mismatch,
            "canonical_retrieved_at": str(c_retrieved) if pd.notna(c_retrieved) else None,
            "mirror_retrieved_at": str(m_retrieved) if pd.notna(m_retrieved) else None,
            "gap_flag": gap_flag.value,
        }
        paired_records.append(record)

    out_df = pd.DataFrame(paired_records)
    out_df = out_df.reindex(columns=MIRROR_MONTHLY_COLUMNS)

    # Sort deterministically by period, direction, hs2 (TOTAL first if present)
    def hs2_sort_key(val: str) -> tuple[int, str]:
        if str(val).upper() == "TOTAL":
            return (0, "")
        return (1, str(val))

    out_df["_sort_key"] = out_df["hs2"].apply(hs2_sort_key)
    out_df.sort_values(by=["period", "direction", "_sort_key"], inplace=True)
    out_df.drop(columns=["_sort_key"], inplace=True)
    out_df.reset_index(drop=True, inplace=True)

    logger.info(
        "Paired canonical (%d) and mirror (%d) rows into %d mirror_monthly records",
        len(canonical_df),
        len(mirror_df),
        len(out_df),
    )
    return out_df
