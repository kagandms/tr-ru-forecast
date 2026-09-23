"""Multi-factor weighted scoring and deterministic ranking engine.

Implements the specification's 4-component weighted model:
- Volume Score: 40% (log1p development trade volume percentile rank)
- Continuity Score: 30% (coverage, zero ratio, missing streak composite)
- Recent Relevance Score: 20% (recent volume 80% + recent activity 20%)
- Forecastability Score: 10% (Seasonal Naive h=1 development sMAPE percentile rank)

Applies deterministic tie-breaking:
1. final_sector_score DESC
2. volume_score DESC
3. development_total_trade_value DESC
4. hs2 ASC
"""

from typing import Any

import numpy as np
import pandas as pd

from tr_ru_forecast.core.logging import get_logger

logger = get_logger("tr_ru_forecast.data.sectors.scoring")

# Weight distribution
WEIGHT_VOLUME = 0.40
WEIGHT_CONTINUITY = 0.30
WEIGHT_RECENT_RELEVANCE = 0.20
WEIGHT_FORECASTABILITY = 0.10

# Recent relevance sub-weights
WEIGHT_RECENT_VOLUME = 0.80
WEIGHT_RECENT_ACTIVITY = 0.20

# Continuity sub-weights
WEIGHT_CONT_COVERAGE = 0.60
WEIGHT_CONT_ZERO = 0.25
WEIGHT_CONT_STREAK = 0.15

MIN_VALID_SEASONAL_COMPARISONS = 36


def compute_percentile_rank_scores(
    values: list[float], higher_is_better: bool = True
) -> list[float]:
    """Compute deterministic fractional percentile rank scores in range [0.0, 100.0].

    Args:
        values: Numerical values to rank.
        higher_is_better: If True, higher values get higher scores.
                         If False, lower values get higher scores (e.g. sMAPE).

    Returns:
        List of float scores scaled to 0.0 - 100.0 with identical values receiving equal scores.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1 or all(v == values[0] for v in values):
        return [100.0] * n

    # Sort order: worst to best
    # If higher_is_better: worst is smallest, best is largest -> sort ascending
    # If lower_is_better: worst is largest, best is smallest -> sort descending
    sorted_vals = sorted(values, reverse=not higher_is_better)

    # Compute average zero-based rank for each unique value
    val_to_rank: dict[float, float] = {}
    i = 0
    while i < n:
        j = i
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        val_to_rank[sorted_vals[i]] = avg_rank
        i = j

    return [(val_to_rank[v] / (n - 1)) * 100.0 for v in values]


def compute_seasonal_naive_smape(
    series_df: pd.DataFrame,
) -> tuple[float | None, int, str | None]:
    """Compute minimal Seasonal Naive (h=1, actual(t-12)) sMAPE on development series.

    Only evaluates within development period. Missing observations remain missing.

    Args:
        series_df: Chronologically sorted DataFrame with 'period' and 'trade_value_usd'.

    Returns:
        Tuple of (smape_float_or_None, valid_comparisons_count, error_reason_or_None).
    """
    values = series_df["trade_value_usd"].tolist()
    n_months = len(values)
    if n_months < 13:
        return None, 0, "Insufficient series length (< 13 months)"

    terms: list[float] = []
    for t in range(12, n_months):
        act = values[t]
        fore = values[t - 12]
        if pd.notna(act) and pd.notna(fore):
            denom = abs(act) + abs(fore)
            if denom == 0.0:
                terms.append(0.0)
            else:
                terms.append(2.0 * abs(act - fore) / denom)

    valid_count = len(terms)
    if valid_count < MIN_VALID_SEASONAL_COMPARISONS:
        return (
            None,
            valid_count,
            f"Valid comparisons {valid_count} < required {MIN_VALID_SEASONAL_COMPARISONS}",
        )

    smape = float(np.mean(terms) * 100.0)
    return smape, valid_count, None


def score_and_rank_candidates(
    eligible_candidates: list[dict[str, Any]],
    timeseries_map: dict[str, pd.DataFrame],
    direction: str,
) -> list[dict[str, Any]]:
    """Score and deterministically rank all eligible sectors for a trade direction.

    Args:
        eligible_candidates: List of candidate metrics dicts that passed eligibility.
        timeseries_map: Map of hs2 -> 156-month development DataFrame.
        direction: Trade direction string ('TR_RU' or 'RU_TR').

    Returns:
        List of candidate dictionaries enriched with scores and sorted by deterministic rank.
    """
    if not eligible_candidates:
        logger.warning(f"No eligible candidates to score for {direction}")
        return []

    # 1. Volume Score (40%)
    vol_values = [
        float(np.log1p(c["development_total_trade_value"])) for c in eligible_candidates
    ]
    vol_scores = compute_percentile_rank_scores(vol_values, higher_is_better=True)

    # 2. Continuity Score (30%)
    cont_scores = []
    for c in eligible_candidates:
        cov = c["coverage_ratio"]
        zero_r = c["zero_month_ratio"]
        streak = c["longest_missing_streak"]
        exp_m = c["development_months_expected"]
        raw = (
            WEIGHT_CONT_COVERAGE * cov
            + WEIGHT_CONT_ZERO * (1.0 - zero_r)
            + WEIGHT_CONT_STREAK * (1.0 - (streak / exp_m))
        )
        clamped = max(0.0, min(1.0, raw))
        cont_scores.append(clamped * 100.0)

    # 3. Recent Relevance Score (20%)
    recent_vol_values = [
        float(np.log1p(c["recent_24m_total_trade_value"])) for c in eligible_candidates
    ]
    recent_vol_scores = compute_percentile_rank_scores(recent_vol_values, higher_is_better=True)
    recent_rel_scores = []
    for i, c in enumerate(eligible_candidates):
        act_score = c["recent_coverage_ratio"] * 100.0
        score = WEIGHT_RECENT_VOLUME * recent_vol_scores[i] + WEIGHT_RECENT_ACTIVITY * act_score
        recent_rel_scores.append(score)

    # 4. Forecastability Score (10%)
    smapes: list[float] = []
    for c in eligible_candidates:
        hs2 = c["hs2"]
        ts_df = timeseries_map.get(hs2)
        if ts_df is not None:
            smape, count, err = compute_seasonal_naive_smape(ts_df)
            c["seasonal_naive_smape"] = smape
            c["seasonal_naive_comparisons"] = count
            c["seasonal_naive_error"] = err
            if smape is not None:
                smapes.append(smape)
            else:
                smapes.append(999.0)  # High penalty if insufficient comparisons
        else:
            c["seasonal_naive_smape"] = None
            c["seasonal_naive_comparisons"] = 0
            c["seasonal_naive_error"] = "No timeseries available"
            smapes.append(999.0)

    forecast_scores = compute_percentile_rank_scores(smapes, higher_is_better=False)

    # Compute Final Weighted Score and populate candidates
    scored_candidates = []
    for i, c in enumerate(eligible_candidates):
        v_sc = vol_scores[i]
        c_sc = cont_scores[i]
        r_sc = recent_rel_scores[i]
        f_sc = forecast_scores[i]

        final_score = (
            WEIGHT_VOLUME * v_sc
            + WEIGHT_CONTINUITY * c_sc
            + WEIGHT_RECENT_RELEVANCE * r_sc
            + WEIGHT_FORECASTABILITY * f_sc
        )
        final_score = max(0.0, min(100.0, final_score))

        item = dict(c)
        item["volume_score"] = round(v_sc, 2)
        item["continuity_score"] = round(c_sc, 2)
        item["recent_relevance_score"] = round(r_sc, 2)
        item["forecastability_score"] = round(f_sc, 2)
        item["final_sector_score"] = round(final_score, 2)
        scored_candidates.append(item)

    # Deterministic sorting:
    # 1. final_sector_score DESC
    # 2. volume_score DESC
    # 3. development_total_trade_value DESC
    # 4. hs2 ASC
    scored_candidates.sort(
        key=lambda x: (
            -x["final_sector_score"],
            -x["volume_score"],
            -x["development_total_trade_value"],
            x["hs2"],
        )
    )

    # Assign 1-based ranks
    for rank, item in enumerate(scored_candidates, start=1):
        item["rank"] = rank

    return scored_candidates
