"""Read M2 mirror gaps with exporter/importer semantics and unavailable post-2022 pairs."""

from __future__ import annotations

import statistics

import pandas as pd

from tr_ru_forecast.serving.history import serialize_usd
from tr_ru_forecast.serving.mirror_schemas import MirrorPoint, MirrorResponse, MirrorSummary
from tr_ru_forecast.serving.schemas import normalize_number


def build_mirror_points(frame: pd.DataFrame) -> tuple[MirrorPoint, ...]:
    """Preserve persisted M2 records. Args: frame. Returns: points. Raises: validation errors."""
    return tuple(_build_mirror_point(record) for _, record in frame.sort_values(
        ["direction", "hs2", "period"]).iterrows())


def _build_mirror_point(record: pd.Series) -> MirrorPoint:
    payload = {name: record[name] for name in MirrorPoint.model_fields if name in record.index}
    for name in ("canonical_value_usd", "mirror_value_usd", "mirror_gap_usd"):
        payload[name] = serialize_usd(record[name])
    for name in ("canonical_retrieved_at", "mirror_retrieved_at"):
        payload[name] = record[name] if pd.notna(record[name]) else None
    payload["sector_id"] = f"{record.direction}:{record.hs2}"
    payload["forecast_target_column"] = (
        "canonical_value_usd" if record.direction == "TR_RU" else "mirror_value_usd"
    )
    return MirrorPoint.model_validate(payload)


def build_mirror_response(
    points: tuple[MirrorPoint, ...], offset: int, limit: int
) -> MirrorResponse:
    """Summarize paired row gaps before pagination. Args: points, offset, limit. Returns: response. Raises: none."""
    before = _summarize(points, "pre_2022")
    after = _summarize(points, "post_2022")
    has_both = before.paired_count > 0 and after.paired_count > 0
    shift = _get_shift(before, after) if has_both else None
    return MirrorResponse(
        gap_definition="exporter minus importer; pct = gap / exporter * 100 (M2); not forecast-target minus reference",
        valuation_note="Exports are FOB, imports CIF; gaps include reporting/classification effects. No causal freight adjustment. Independent national datasets are not present.",
        structural_shift_status="available" if has_both else "unavailable_no_paired_post_2022",
        structural_shift_pp=shift, summaries=(before, after), observations=points[offset:offset + limit],
        total_observations=len(points), offset=offset, limit=limit,
    )


def _summarize(points: tuple[MirrorPoint, ...], period_group: str) -> MirrorSummary:
    selected = tuple(point for point in points if (point.period < "2022-01") == (period_group == "pre_2022"))
    paired = tuple(point for point in selected if point.pair_status == "BOTH_PRESENT")
    gaps = tuple(point.abs_mirror_gap_pct for point in paired if point.abs_mirror_gap_pct is not None)
    return MirrorSummary.model_validate({
        "period_group": period_group, "observation_count": len(selected), "paired_count": len(paired),
        "mean_abs_gap_pct": normalize_number(statistics.mean(gaps)) if gaps else None,
        "median_abs_gap_pct": normalize_number(statistics.median(gaps)) if gaps else None,
    })


def _get_shift(before: MirrorSummary, after: MirrorSummary) -> float | None:
    if before.mean_abs_gap_pct is None or after.mean_abs_gap_pct is None:
        return None
    return after.mean_abs_gap_pct - before.mean_abs_gap_pct
