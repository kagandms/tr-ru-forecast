"""Exporter/importer mirror provenance and bounded bilateral discrepancy views."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from tr_ru_forecast.serving.schemas import Direction, FrozenModel, Month, Number


class MirrorPoint(FrozenModel):
    """Immutable mirror point contract."""

    sector_id: str
    direction: Direction
    hs2: str
    period: Month
    canonical_value_usd: str | None
    mirror_value_usd: str | None
    mirror_gap_usd: str | None
    mirror_gap_pct: Number
    abs_mirror_gap_pct: Number
    pair_status: str
    classification_mismatch: bool
    gap_flag: str
    canonical_retrieved_at: datetime | None
    mirror_retrieved_at: datetime | None
    forecast_target_column: Literal["canonical_value_usd", "mirror_value_usd"]


class MirrorSummary(FrozenModel):
    """Immutable mirror summary contract."""

    period_group: Literal["pre_2022", "post_2022"]
    observation_count: int
    paired_count: int
    mean_abs_gap_pct: Number
    median_abs_gap_pct: Number


class MirrorResponse(FrozenModel):
    """Immutable mirror response contract."""

    source: Literal["UN Comtrade"] = "UN Comtrade"
    reporter_sources: tuple[str, ...] = ("Türkiye reporter", "Russia reporter")
    gap_definition: str
    valuation_note: str
    structural_shift_status: Literal["available", "unavailable_no_paired_post_2022"]
    structural_shift_pp: Number
    summaries: tuple[MirrorSummary, ...]
    observations: tuple[MirrorPoint, ...]
    total_observations: int
    offset: int
    limit: int
