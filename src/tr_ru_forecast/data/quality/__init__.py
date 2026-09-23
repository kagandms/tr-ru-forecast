"""Bilateral mirror statistics and data quality layer (M2)."""

from tr_ru_forecast.data.quality.coverage_analyzer import BilateralCoverageAnalyzer
from tr_ru_forecast.data.quality.mirror_pairing import (
    calculate_mirror_gap,
    pair_canonical_and_mirror,
)
from tr_ru_forecast.data.quality.mirror_pipeline import MirrorQualityPipeline
from tr_ru_forecast.data.quality.mirror_transformer import (
    deduplicate_mirror_records,
    transform_mirror_raw_record,
)

__all__ = [
    "BilateralCoverageAnalyzer",
    "MirrorQualityPipeline",
    "calculate_mirror_gap",
    "deduplicate_mirror_records",
    "pair_canonical_and_mirror",
    "transform_mirror_raw_record",
]
