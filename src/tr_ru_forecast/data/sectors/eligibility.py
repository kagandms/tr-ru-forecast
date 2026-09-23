"""Sector eligibility filter evaluation engine.

Implements the five mandatory eligibility criteria (A through E) defined in
DATA_SPEC.md and Specification v1.1:
- A: Minimum potential history >= 96 months
- B: Coverage ratio >= 90% (>= 141 / 156 observed months)
- C: Zero-month ratio <= 10% of observed months
- D: Recent activity >= 21 / 24 months in 2023-01 -> 2024-12
- E: Non-trivial economic size >= 10th percentile (P10) of development trade volume
"""

import math
from typing import Any

import numpy as np

from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.data.sectors.hs_audit import get_special_code_reason, is_special_hs_code

logger = get_logger("tr_ru_forecast.data.sectors.eligibility")

MIN_HISTORY_MONTHS = 96
COVERAGE_THRESHOLD_PCT = 0.90  # 90%
MAX_ZERO_RATIO = 0.10  # 10%
RECENT_WINDOW_MONTHS = 24
MIN_RECENT_OBSERVED_MONTHS = 21  # 21 of 24
SIZE_PERCENTILE = 0.10  # P10


def calculate_min_coverage_observed(total_expected: int = 156) -> int:
    """Calculate the integer threshold for 90% coverage using ceiling."""
    return math.ceil(total_expected * COVERAGE_THRESHOLD_PCT)


def evaluate_candidate_eligibility(
    metrics_list: list[dict[str, Any]],
    direction: str,
) -> tuple[list[dict[str, Any]], float]:
    """Evaluate eligibility filters A through E for all candidates in a trade direction.

    Args:
        metrics_list: List of metric dictionaries produced by MonthlyDevelopmentPanel.
        direction: Trade direction string ('TR_RU' or 'RU_TR').

    Returns:
        Tuple of (updated_metrics_list, p10_size_threshold_usd).
    """
    total_expected = 156
    min_observed_required = calculate_min_coverage_observed(total_expected)

    # First pass: Filters A, B, C, D and special code check
    preliminary_passed: list[dict[str, Any]] = []

    for m in metrics_list:
        failures: list[str] = []
        hs2 = m["hs2"]

        # Special code exclusion (e.g. 77, 98, 99)
        if is_special_hs_code(hs2):
            reason = get_special_code_reason(hs2) or "Special classification / residual code"
            failures.append(f"Excluded: {reason}")

        # Filter A: Minimum potential history (at least 96 months potential or observed)
        # Note: If observed months >= 96 or first-to-last span >= 96 months
        if m["observed_months"] < MIN_HISTORY_MONTHS:
            failures.append(
                f"Filter A (History): Observed {m['observed_months']} months < minimum {MIN_HISTORY_MONTHS}"
            )

        # Filter B: Coverage >= 90% (>= 141 observed months)
        if m["observed_months"] < min_observed_required:
            failures.append(
                f"Filter B (Coverage): Observed {m['observed_months']} months < minimum {min_observed_required} ({COVERAGE_THRESHOLD_PCT*100:.0f}%)"
            )

        # Filter C: Zero ratio <= 10%
        if m["zero_month_ratio"] > MAX_ZERO_RATIO:
            failures.append(
                f"Filter C (Zero ratio): Zero-month ratio {m['zero_month_ratio']*100:.1f}% > maximum {MAX_ZERO_RATIO*100:.0f}%"
            )

        # Filter D: Recent activity >= 21 / 24 observed months
        if m["recent_observed_months"] < MIN_RECENT_OBSERVED_MONTHS:
            failures.append(
                f"Filter D (Recent activity): Recent observed {m['recent_observed_months']}/24 < minimum {MIN_RECENT_OBSERVED_MONTHS}"
            )

        m["preliminary_failures"] = failures
        m["passes_preliminary"] = len(failures) == 0

        if m["passes_preliminary"] and m["development_total_trade_value"] > 0:
            preliminary_passed.append(m)

    # Filter E: Non-trivial economic size (10th percentile among candidates passing A-D)
    if preliminary_passed:
        volumes = np.array([p["development_total_trade_value"] for p in preliminary_passed])
        p10_threshold = float(np.percentile(volumes, SIZE_PERCENTILE * 100))
    else:
        p10_threshold = 0.0

    logger.info(
        f"Direction {direction}: {len(preliminary_passed)} candidates passed A-D. "
        f"P10 size threshold: ${p10_threshold:,.2f}"
    )

    # Second pass: Apply Filter E and finalize eligibility
    for m in metrics_list:
        failures = list(m.get("preliminary_failures", []))

        if m.get("passes_preliminary", False):
            if m["development_total_trade_value"] < p10_threshold:
                failures.append(
                    f"Filter E (Size): Development volume ${m['development_total_trade_value']:,.2f} < P10 threshold ${p10_threshold:,.2f}"
                )

        m["eligibility_pass"] = len(failures) == 0
        m["eligibility_failure_reasons"] = failures
        m["p10_size_threshold_usd"] = p10_threshold

        # Clean up temporary keys
        m.pop("preliminary_failures", None)
        m.pop("passes_preliminary", None)

    return metrics_list, p10_threshold
