"""M3 Sector Audit & Selection package.

Implements official data audit, eligibility filtering, multi-factor scoring,
and deterministic sector selection under Specification v1.1.
"""

from tr_ru_forecast.data.sectors.eligibility import (
    calculate_min_coverage_observed,
    evaluate_candidate_eligibility,
)
from tr_ru_forecast.data.sectors.hs_audit import (
    HS2_DESCRIPTIONS,
    SPECIAL_HS_CODES,
    audit_hs_revisions,
    get_hs_description,
    is_special_hs_code,
)
from tr_ru_forecast.data.sectors.panel_builder import (
    DEVELOPMENT_MONTHS_COUNT,
    MonthlyDevelopmentPanel,
    enforce_holdout_guard,
    generate_development_period_index,
)
from tr_ru_forecast.data.sectors.scoring import (
    compute_percentile_rank_scores,
    compute_seasonal_naive_smape,
    score_and_rank_candidates,
)
from tr_ru_forecast.data.sectors.selector import SectorSelector
from tr_ru_forecast.data.sectors.target_materializer import (
    build_forecast_target_dataset,
    validate_target_provenance,
)

__all__ = [
    "DEVELOPMENT_MONTHS_COUNT",
    "HS2_DESCRIPTIONS",
    "SPECIAL_HS_CODES",
    "MonthlyDevelopmentPanel",
    "SectorSelector",
    "audit_hs_revisions",
    "build_forecast_target_dataset",
    "calculate_min_coverage_observed",
    "compute_percentile_rank_scores",
    "compute_seasonal_naive_smape",
    "enforce_holdout_guard",
    "evaluate_candidate_eligibility",
    "generate_development_period_index",
    "get_hs_description",
    "is_special_hs_code",
    "score_and_rank_candidates",
    "validate_target_provenance",
]
