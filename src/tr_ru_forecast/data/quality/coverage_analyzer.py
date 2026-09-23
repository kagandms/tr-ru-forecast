"""Analytical coverage and bilateral quality audit engine.

Produces data quality matrices, discrepancy distributions, CIF/FOB margin context,
and empirical analysis of reporting suspensions without hardcoded assumptions,
strictly adhering to DATA_SPEC.md and PROJECT_SPEC.md.

V1 M2 Revision Gate:
- Clear separation between PARTIAL and FULL verification scopes.
- When full_history_verified is False, reporting suspension is flagged as UNVERIFIED,
  never CONFIRMED.
- All direction summaries report complete universe metadata (expected vs observed period counts).
"""

from typing import Any

import pandas as pd

from tr_ru_forecast.core.constants import PairStatus, TradeDirection
from tr_ru_forecast.core.logging import get_logger

logger = get_logger("tr_ru_forecast.data.quality.coverage_analyzer")


class BilateralCoverageAnalyzer:
    """Evaluates coverage, pairing completeness, and gap metrics across bilateral trade flows."""

    def __init__(
        self,
        paired_df: pd.DataFrame,
        full_history_verified: bool = False,
        expected_period_count: int | None = None,
    ) -> None:
        self.df = paired_df.copy() if not paired_df.empty else pd.DataFrame()
        self.full_history_verified = bool(full_history_verified)
        self.verification_scope = "FULL" if self.full_history_verified else "PARTIAL"
        self.expected_period_count = expected_period_count

    def generate_full_report(self) -> dict[str, Any]:
        """Generate comprehensive bilateral trade quality and mirror coverage audit.

        Returns:
            Dictionary containing verification metadata, per-direction metrics,
            yearly coverage breakdown, empirical reporting suspension audit,
            and CIF/FOB accounting notes.
        """
        observed_periods = sorted(self.df["period"].unique().tolist()) if not self.df.empty else []
        obs_count = len(observed_periods)
        exp_count = self.expected_period_count or obs_count

        if self.df.empty:
            return {
                "verification_scope": self.verification_scope,
                "full_history_verified": self.full_history_verified,
                "expected_period_count": exp_count,
                "observed_period_count": 0,
                "period_min": None,
                "period_max": None,
                "total_records": 0,
                "directions": {},
                "reporting_suspension_empirical_audit": {},
                "cif_fob_accounting_note": self.get_cif_fob_accounting_note(),
            }

        directions_summary: dict[str, Any] = {}
        for direction in [TradeDirection.TR_RU.value, TradeDirection.RU_TR.value]:
            dir_df = self.df[self.df["direction"] == direction]
            if not dir_df.empty:
                directions_summary[direction] = self._analyze_direction(dir_df)

        suspension_audit = self._audit_reporting_suspension(obs_count, exp_count)

        return {
            "verification_scope": self.verification_scope,
            "full_history_verified": self.full_history_verified,
            "expected_period_count": exp_count,
            "observed_period_count": obs_count,
            "period_min": observed_periods[0] if observed_periods else None,
            "period_max": observed_periods[-1] if observed_periods else None,
            "total_records": len(self.df),
            "directions": directions_summary,
            "reporting_suspension_empirical_audit": suspension_audit,
            "cif_fob_accounting_note": self.get_cif_fob_accounting_note(),
        }

    def _analyze_direction(self, dir_df: pd.DataFrame) -> dict[str, Any]:
        """Compute detailed quality and coverage metrics for a single trade direction."""
        total = len(dir_df)
        dir_periods = sorted(dir_df["period"].unique().tolist())
        dir_obs_count = len(dir_periods)
        exp_count = self.expected_period_count or dir_obs_count

        status_counts = dir_df["pair_status"].value_counts().to_dict()
        flag_counts = dir_df["gap_flag"].value_counts().to_dict()

        mismatch_count = int(dir_df["classification_mismatch"].sum())

        can_obs_count = int((~dir_df["canonical_missing"]).sum())
        mir_obs_count = int((~dir_df["mirror_missing"]).sum())
        both_count = int((dir_df["pair_status"] == PairStatus.BOTH_PRESENT.value).sum())
        can_only_count = int((dir_df["pair_status"] == PairStatus.CANONICAL_ONLY.value).sum())
        mir_only_count = int((dir_df["pair_status"] == PairStatus.MIRROR_ONLY.value).sum())
        both_miss_count = int((dir_df["pair_status"] == PairStatus.BOTH_MISSING.value).sum())

        both_present_df = dir_df[dir_df["pair_status"] == PairStatus.BOTH_PRESENT.value]

        gap_stats: dict[str, Any] = {
            "both_present_count": both_count,
            "median_abs_gap_usd": None,
            "mean_abs_gap_usd": None,
            "median_abs_gap_pct": None,
            "mean_abs_gap_pct": None,
            "min_gap_usd": None,
            "max_gap_usd": None,
            "top_discrepancies": [],
        }

        if both_count > 0:
            valid_gaps_usd = both_present_df["mirror_gap_usd"].dropna()
            valid_abs_gaps_pct = (
                both_present_df["abs_mirror_gap_pct"].dropna()
                if "abs_mirror_gap_pct" in both_present_df.columns
                else both_present_df["mirror_gap_pct"].dropna().abs()
            )

            if not valid_gaps_usd.empty:
                gap_stats["median_abs_gap_usd"] = float(valid_gaps_usd.abs().median())
                gap_stats["mean_abs_gap_usd"] = float(valid_gaps_usd.abs().mean())
                gap_stats["min_gap_usd"] = float(valid_gaps_usd.min())
                gap_stats["max_gap_usd"] = float(valid_gaps_usd.max())

            if not valid_abs_gaps_pct.empty:
                gap_stats["median_abs_gap_pct"] = float(valid_abs_gaps_pct.median())
                gap_stats["mean_abs_gap_pct"] = float(valid_abs_gaps_pct.mean())

            # Identify top 5 largest absolute percentage discrepancies
            top_discrepant = both_present_df.dropna(subset=["mirror_gap_pct"]).copy()
            if not top_discrepant.empty:
                if "abs_mirror_gap_pct" in top_discrepant.columns:
                    top_discrepant["_rank_abs_pct"] = top_discrepant["abs_mirror_gap_pct"]
                else:
                    top_discrepant["_rank_abs_pct"] = top_discrepant["mirror_gap_pct"].abs()
                top_5 = top_discrepant.sort_values(by="_rank_abs_pct", ascending=False).head(5)
                gap_stats["top_discrepancies"] = [
                    {
                        "period": row["period"],
                        "hs2": row["hs2"],
                        "canonical_usd": row["canonical_value_usd"],
                        "mirror_usd": row["mirror_value_usd"],
                        "gap_usd": row["mirror_gap_usd"],
                        "gap_pct": row["mirror_gap_pct"],
                        "abs_gap_pct": row.get("abs_mirror_gap_pct"),
                        "flag": row["gap_flag"],
                    }
                    for _, row in top_5.iterrows()
                ]

        # Yearly coverage breakdown
        yearly_coverage = self._compute_yearly_coverage(dir_df)

        return {
            "verification_scope": self.verification_scope,
            "full_history_verified": self.full_history_verified,
            "expected_period_count": exp_count,
            "observed_period_count": dir_obs_count,
            "period_min": dir_periods[0] if dir_periods else None,
            "period_max": dir_periods[-1] if dir_periods else None,
            "total_records": total,
            "canonical_observed_count": can_obs_count,
            "mirror_observed_count": mir_obs_count,
            "both_present_count": both_count,
            "canonical_only_count": can_only_count,
            "mirror_only_count": mir_only_count,
            "both_missing_count": both_miss_count,
            "pair_status_distribution": {
                s.value: int(status_counts.get(s.value, 0)) for s in PairStatus
            },
            "classification_mismatch_count": mismatch_count,
            "gap_flag_distribution": flag_counts,
            "paired_gap_statistics": gap_stats,
            "yearly_coverage": yearly_coverage,
        }

    def _compute_yearly_coverage(self, dir_df: pd.DataFrame) -> dict[str, Any]:
        """Calculate coverage and pairing completeness grouped by calendar year."""
        dir_df = dir_df.copy()
        dir_df["year"] = dir_df["period"].str.slice(0, 4)
        years = sorted(dir_df["year"].unique())

        yearly_results: dict[str, Any] = {}
        for yr in years:
            yr_subset = dir_df[dir_df["year"] == yr]
            total_yr = len(yr_subset)
            both_yr = int((yr_subset["pair_status"] == PairStatus.BOTH_PRESENT.value).sum())
            can_yr = int((yr_subset["pair_status"] == PairStatus.CANONICAL_ONLY.value).sum())
            mir_yr = int((yr_subset["pair_status"] == PairStatus.MIRROR_ONLY.value).sum())

            valid_pct = (
                yr_subset["abs_mirror_gap_pct"].dropna()
                if "abs_mirror_gap_pct" in yr_subset.columns
                else yr_subset["mirror_gap_pct"].dropna().abs()
            )
            mean_abs_pct = float(valid_pct.mean()) if not valid_pct.empty else None

            yearly_results[yr] = {
                "total_pairs": total_yr,
                "both_present": both_yr,
                "canonical_only": can_yr,
                "mirror_only": mir_yr,
                "mean_abs_gap_pct": mean_abs_pct,
            }
        return yearly_results

    def _audit_reporting_suspension(self, obs_count: int, exp_count: int) -> dict[str, Any]:
        """Empirically assess reporting discontinuity without claiming unverified full history.

        Rules:
            - If full_history_verified is False: status is UNVERIFIED, empirical_reporting_suspension_detected is False.
            - If full_history_verified is True: evaluates structural drop; status is CONFIRMED or NOT_DETECTED.
        """
        if self.df.empty or "period" not in self.df.columns:
            return {}

        audit_results: dict[str, Any] = {}
        suspension_cutoff = "2022-03"

        for direction in [TradeDirection.TR_RU.value, TradeDirection.RU_TR.value]:
            sub = self.df[self.df["direction"] == direction]
            if sub.empty:
                continue

            pre_cutoff = sub[sub["period"] < suspension_cutoff]
            post_cutoff = sub[sub["period"] >= suspension_cutoff]

            pre_can_count = int(
                (pre_cutoff["pair_status"] == PairStatus.CANONICAL_ONLY.value).sum()
            )
            pre_both_count = int((pre_cutoff["pair_status"] == PairStatus.BOTH_PRESENT.value).sum())
            pre_mir_count = int((pre_cutoff["pair_status"] == PairStatus.MIRROR_ONLY.value).sum())

            post_can_count = int(
                (post_cutoff["pair_status"] == PairStatus.CANONICAL_ONLY.value).sum()
            )
            post_both_count = int(
                (post_cutoff["pair_status"] == PairStatus.BOTH_PRESENT.value).sum()
            )
            post_mir_count = int((post_cutoff["pair_status"] == PairStatus.MIRROR_ONLY.value).sum())

            total_pre = len(pre_cutoff)
            total_post = len(post_cutoff)

            # Ratios computed strictly over observed sample
            pre_can_ratio = (
                ((pre_both_count + pre_can_count) / total_pre * 100.0) if total_pre > 0 else 0.0
            )
            post_can_ratio = (
                ((post_both_count + post_can_count) / total_post * 100.0) if total_post > 0 else 0.0
            )

            pre_mir_ratio = (
                ((pre_both_count + pre_mir_count) / total_pre * 100.0) if total_pre > 0 else 0.0
            )
            post_mir_ratio = (
                ((post_both_count + post_mir_count) / total_post * 100.0) if total_post > 0 else 0.0
            )

            # Full-history claim safety guardrail
            if not self.full_history_verified:
                reporting_suspension_status = "UNVERIFIED"
                empirical_shift_detected = False
                finding = (
                    f"PRELIMINARY / PARTIAL COVERAGE: Direction {direction} audit is based on {obs_count} "
                    f"observed periods (out of {exp_count} expected). Within this observed sample, "
                    f"canonical presence was {pre_can_ratio:.1f}% pre-{suspension_cutoff} and {post_can_ratio:.1f}% "
                    f"post-{suspension_cutoff}, while mirror presence was {post_mir_ratio:.1f}%. "
                    "However, because full-history verification is NO (unauthenticated public preview), "
                    "reporting suspension status remains UNVERIFIED and cannot be classified as a confirmed finding."
                )
            else:
                empirical_shift_detected = bool(
                    pre_can_ratio > 70.0 and post_can_ratio < 20.0 and post_mir_ratio > 50.0
                )
                if empirical_shift_detected:
                    reporting_suspension_status = "CONFIRMED"
                    finding = (
                        f"CONFIRMED FULL HISTORY AUDIT: Direction {direction} demonstrates an empirical "
                        f"structural disruption post-{suspension_cutoff}: Canonical presence fell from {pre_can_ratio:.1f}% "
                        f"to {post_can_ratio:.1f}%, while partner mirror reporting remained at {post_mir_ratio:.1f}%. "
                        "Mirror statistics provide vital visibility into post-disruption flows without violating "
                        "the canonical non-overwrite rule."
                    )
                else:
                    reporting_suspension_status = "NOT_DETECTED"
                    finding = f"CONFIRMED FULL HISTORY AUDIT: Direction {direction} shows stable reporting dynamics across pre- and post-{suspension_cutoff} periods."

            audit_results[direction] = {
                "verification_scope": self.verification_scope,
                "full_history_verified": self.full_history_verified,
                "reporting_suspension_status": reporting_suspension_status,
                "suspension_cutoff": suspension_cutoff,
                "observed_period_count": obs_count,
                "expected_period_count": exp_count,
                "pre_cutoff_records": total_pre,
                "pre_cutoff_canonical_presence_pct": round(pre_can_ratio, 2),
                "pre_cutoff_mirror_presence_pct": round(pre_mir_ratio, 2),
                "post_cutoff_records": total_post,
                "post_cutoff_canonical_presence_pct": round(post_can_ratio, 2),
                "post_cutoff_mirror_presence_pct": round(post_mir_ratio, 2),
                "empirical_reporting_suspension_detected": empirical_shift_detected,
                "analytical_finding": finding,
            }

        return audit_results

    @staticmethod
    def get_cif_fob_accounting_note() -> str:
        """Economic explanation of standard CIF vs FOB accounting discrepancies."""
        return (
            "Accounting Explanation of Mirror Discrepancies (CIF vs FOB):\n"
            "In UN Comtrade and international trade accounting standards, export flows are conventionally "
            "valued on a Free-On-Board (FOB) basis, measuring value at the exporter's border excluding international "
            "freight and maritime/air insurance. Conversely, partner import flows are conventionally recorded on a "
            "Cost, Insurance, and Freight (CIF) basis, measuring value at the importer's port of arrival including "
            "transportation, handling, and insurance margins. Consequently, import mirror values are expected to "
            "exceed canonical export values (resulting in a negative mirror_gap_usd = canonical - mirror) by the "
            "magnitude of the bilateral freight-and-insurance margin. Discrepancies are therefore not necessarily "
            "statistical errors but an economic reflection of international transport valuation margins."
        )
