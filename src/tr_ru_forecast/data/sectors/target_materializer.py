"""Forecast target dataset materializer under Specification v1.1.

Materializes the unified forecast target dataset:
- TR_RU: Türkiye exporter-reported trade to Russia (reporter=792, partner=643, flow=Export)
- RU_TR: Türkiye importer-reported trade from Russia (reporter=792, partner=643, flow=Import)

Strictly rejects Russian exporter-reported series as RU_TR forecast target.
"""

from pathlib import Path

import pandas as pd

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.constants import (
    FORECAST_TARGET_FLOWS,
    TradeDirection,
)
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.data.sectors.hs_audit import get_hs_description

logger = get_logger("tr_ru_forecast.data.sectors.target_materializer")

TARGET_COLUMNS = [
    "period",
    "direction",
    "reporter_code",
    "partner_code",
    "flow",
    "hs2",
    "hs_description",
    "trade_value_usd",
    "net_weight_kg",
    "quantity",
    "classification",
    "classification_code",
    "is_original_classification",
    "is_total",
    "source",
    "retrieved_at",
    "data_status",
]


def validate_target_provenance(df: pd.DataFrame, expected_direction: TradeDirection) -> None:
    """Verify that records strictly adhere to the v1.1 forecast target specification.

    Raises:
        ValueError: If records do not match official Türkiye-reported criteria.
    """
    target_spec = FORECAST_TARGET_FLOWS[expected_direction]
    exp_reporter = target_spec["reporter_code"]
    exp_partner = target_spec["partner_code"]
    exp_flow = target_spec["flow"]

    # Check reporter
    reporters = df["reporter_code"].dropna().unique().tolist()
    if any(r != exp_reporter for r in reporters):
        raise ValueError(
            f"Target provenance violation for {expected_direction.value}: "
            f"Expected reporter_code={exp_reporter} (Türkiye), found {reporters}. "
            "Russia-reported series must NOT be used as forecast targets."
        )

    # Check partner
    partners = df["partner_code"].dropna().unique().tolist()
    if any(p != exp_partner for p in partners):
        raise ValueError(
            f"Target provenance violation for {expected_direction.value}: "
            f"Expected partner_code={exp_partner} (Russia), found {partners}."
        )

    # Check flow
    flows = df["flow"].dropna().unique().tolist()
    if any(f != exp_flow for f in flows):
        raise ValueError(
            f"Target provenance violation for {expected_direction.value}: "
            f"Expected flow='{exp_flow}', found {flows}."
        )


def build_forecast_target_dataset(
    tr_ru_path: Path | str | None = None,
    ru_tr_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Build and persist the canonical forecast target monthly dataset under Spec v1.1.

    Args:
        tr_ru_path: Path to Türkiye exporter-reported trade (defaults to processed/trade_monthly_TR_RU.parquet).
        ru_tr_path: Path to Türkiye importer-reported trade (defaults to processed/trade_monthly_mirror_RU_TR.parquet).
        output_path: Destination path for forecast_target_monthly.parquet.

    Returns:
        Unified DataFrame containing standardized forecast target records.
    """
    settings = get_settings()
    tr_ru_path = Path(tr_ru_path or settings.processed_data_dir / "trade_monthly_TR_RU.parquet")
    ru_tr_path = Path(ru_tr_path or settings.processed_data_dir / "trade_monthly_mirror_RU_TR.parquet")
    output_path = Path(output_path or settings.processed_data_dir / "forecast_target_monthly.parquet")

    if not tr_ru_path.exists():
        raise FileNotFoundError(f"TR_RU source dataset not found: {tr_ru_path}")
    if not ru_tr_path.exists():
        raise FileNotFoundError(f"RU_TR source dataset not found: {ru_tr_path}")

    logger.info(f"Loading TR_RU target from {tr_ru_path}")
    df_tr_ru = pd.read_parquet(tr_ru_path)
    validate_target_provenance(df_tr_ru, TradeDirection.TR_RU)

    logger.info(f"Loading RU_TR target from {ru_tr_path}")
    df_ru_tr = pd.read_parquet(ru_tr_path)
    validate_target_provenance(df_ru_tr, TradeDirection.RU_TR)

    # Ensure direction strings are correctly labeled
    df_tr_ru = df_tr_ru.copy()
    df_tr_ru["direction"] = TradeDirection.TR_RU.value

    df_ru_tr = df_ru_tr.copy()
    df_ru_tr["direction"] = TradeDirection.RU_TR.value

    combined = pd.concat([df_tr_ru, df_ru_tr], ignore_index=True)

    # Enrich descriptions where missing
    def _enrich_desc(row: pd.Series) -> str:
        if row.get("is_total", False) or row.get("hs2") == "TOTAL":
            return "Total All Commodities"
        existing = row.get("hs_description")
        if pd.notna(existing) and str(existing).strip():
            return str(existing).strip()
        hs2_val = str(row.get("hs2", ""))
        return get_hs_description(hs2_val)

    combined["hs_description"] = combined.apply(_enrich_desc, axis=1)

    # Standardize column selection
    available_cols = [col for col in TARGET_COLUMNS if col in combined.columns]
    target_df = combined[available_cols].copy()

    # Sort deterministically
    sort_cols = [c for c in ["direction", "hs2", "period"] if c in target_df.columns]
    target_df.sort_values(by=sort_cols, inplace=True)
    target_df.reset_index(drop=True, inplace=True)

    # Verify no null periods or directions
    assert target_df["period"].isna().sum() == 0, "Null periods in target dataset"
    assert target_df["direction"].isna().sum() == 0, "Null direction in target dataset"

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_df.to_parquet(output_path, index=False)
    logger.info(f"Materialized {len(target_df)} forecast target records to {output_path}")

    return target_df
