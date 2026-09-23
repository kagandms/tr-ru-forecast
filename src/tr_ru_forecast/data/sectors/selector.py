"""End-to-end sector auditor, ranker, and selection manifest generator.

Executes the M3 workflow under Specification v1.1:
1. Audits candidate HS2 universe across 156-month development period.
2. Applies strict eligibility filters (A through E).
3. Executes deterministic multi-factor scoring (Volume 40%, Continuity 30%, Recent 20%, Forecastability 10%).
4. Selects top max 6 sectors per direction (or fewer if eligible count < 6).
5. Persists frozen selection manifest (configs/selected_sectors_v1_1.yaml) and parquet/json audit outputs.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.constants import (
    DEVELOPMENT_DATA_END,
    DEVELOPMENT_DATA_START,
    HOLDOUT_DATA_END,
    HOLDOUT_DATA_START,
    MAX_SECTORS_PER_DIRECTION,
    TradeDirection,
)
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.data.sectors.eligibility import evaluate_candidate_eligibility
from tr_ru_forecast.data.sectors.panel_builder import MonthlyDevelopmentPanel
from tr_ru_forecast.data.sectors.scoring import score_and_rank_candidates

logger = get_logger("tr_ru_forecast.data.sectors.selector")


class SectorSelector:
    """Orchestrates sector selection and frozen manifest generation."""

    def __init__(
        self,
        target_df: pd.DataFrame,
        max_sectors_per_direction: int = MAX_SECTORS_PER_DIRECTION,
    ) -> None:
        self.max_sectors = max_sectors_per_direction
        self.panel = MonthlyDevelopmentPanel(target_df)

    def run_selection(self) -> dict[str, Any]:
        """Execute full sector audit, scoring, and selection across both directions."""
        results_by_dir: dict[str, Any] = {}
        all_metrics_records: list[dict[str, Any]] = []

        for direction in [TradeDirection.TR_RU.value, TradeDirection.RU_TR.value]:
            logger.info(f"--- Running sector selection for {direction} ---")
            candidate_codes = self.panel.get_candidate_universe(direction)
            logger.info(f"{direction}: Found {len(candidate_codes)} observed HS2 candidates")

            metrics_list: list[dict[str, Any]] = []
            ts_map: dict[str, pd.DataFrame] = {}
            for hs2 in candidate_codes:
                m = self.panel.compute_sector_metrics(direction, hs2)
                metrics_list.append(m)
                ts_map[hs2] = self.panel.build_sector_timeseries(direction, hs2)

            # Evaluate Eligibility (Filters A-E)
            evaluated_metrics, p10_thresh = evaluate_candidate_eligibility(
                metrics_list, direction
            )

            # Partition into eligible and ineligible
            eligible = [m for m in evaluated_metrics if m["eligibility_pass"]]
            ineligible = [m for m in evaluated_metrics if not m["eligibility_pass"]]

            logger.info(
                f"{direction}: {len(eligible)} eligible candidates passed all criteria "
                f"({len(ineligible)} failed)"
            )

            # Score and Rank Eligible Candidates
            ranked_eligible = score_and_rank_candidates(eligible, ts_map, direction)

            # Select Top <= max_sectors
            selected_sectors = ranked_eligible[: self.max_sectors]
            # Mark selected flag on ranked eligible candidates
            selected_hs2_set = {s["hs2"] for s in selected_sectors}
            for r in ranked_eligible:
                r["is_selected"] = r["hs2"] in selected_hs2_set

            results_by_dir[direction] = {
                "direction": direction,
                "total_candidates": len(candidate_codes),
                "eligible_count": len(eligible),
                "ineligible_count": len(ineligible),
                "p10_size_threshold_usd": p10_thresh,
                "selected_sectors": selected_sectors,
                "ranked_eligible": ranked_eligible,
                "ineligible": ineligible,
            }

            # Gather for Parquet output
            for r in ranked_eligible:
                rec = dict(r)
                all_metrics_records.append(rec)

            for inel in ineligible:
                rec = dict(inel)
                rec["is_selected"] = False
                rec["rank"] = None
                rec["volume_score"] = None
                rec["continuity_score"] = None
                rec["recent_relevance_score"] = None
                rec["forecastability_score"] = None
                rec["final_sector_score"] = None
                all_metrics_records.append(rec)

        return {
            "metadata": {
                "spec_version": "1.1",
                "selection_version": "1.1.0",
                "created_at": datetime.now(UTC).isoformat(),
                "development_period": {
                    "start": DEVELOPMENT_DATA_START,
                    "end": DEVELOPMENT_DATA_END,
                    "months_count": 156,
                },
                "holdout_period": {
                    "start": HOLDOUT_DATA_START,
                    "end": HOLDOUT_DATA_END,
                    "months_count": 12,
                    "status": "UNTOUCHED_ISOLATED",
                },
                "target_policy": {
                    "TR_RU": "Türkiye exporter-reported (792 -> 643, Flow X)",
                    "RU_TR": "Türkiye importer-reported (792 <- 643, Flow M)",
                },
            },
            "directions": results_by_dir,
            "all_metrics_records": all_metrics_records,
        }

    def export_artifacts(
        self,
        results: dict[str, Any],
        manifest_path: Path | str | None = None,
        metrics_parquet_path: Path | str | None = None,
        results_json_path: Path | str | None = None,
    ) -> tuple[Path, Path, Path]:
        """Export frozen YAML manifest, metrics parquet, and results json."""
        settings = get_settings()
        manifest_path = Path(manifest_path or settings.project_root / "configs" / "selected_sectors_v1_1.yaml")
        metrics_parquet_path = Path(
            metrics_parquet_path or settings.processed_data_dir / "sector_selection_metrics.parquet"
        )
        results_json_path = Path(
            results_json_path or settings.processed_data_dir / "sector_selection_results.json"
        )

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        results_json_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Build and export YAML Manifest
        meta = results["metadata"]
        manifest_data: dict[str, Any] = {
            "spec_version": meta["spec_version"],
            "selection_version": meta["selection_version"],
            "created_at": meta["created_at"],
            "development_period": f"{meta['development_period']['start']} to {meta['development_period']['end']}",
            "holdout_period": f"{meta['holdout_period']['start']} to {meta['holdout_period']['end']} (untouched)",
            "target_policy": meta["target_policy"],
            "total_modeling_series": 2 + sum(
                len(results["directions"][d]["selected_sectors"])
                for d in [TradeDirection.TR_RU.value, TradeDirection.RU_TR.value]
            ),
            "TR_RU": [
                {
                    "rank": s["rank"],
                    "hs2": s["hs2"],
                    "description": s["hs_description"],
                    "final_score": s["final_sector_score"],
                    "volume_score": s["volume_score"],
                    "continuity_score": s["continuity_score"],
                    "recent_relevance_score": s["recent_relevance_score"],
                    "forecastability_score": s["forecastability_score"],
                    "development_total_usd": s["development_total_trade_value"],
                    "coverage_pct": round(s["coverage_ratio"] * 100, 1),
                }
                for s in results["directions"]["TR_RU"]["selected_sectors"]
            ],
            "RU_TR": [
                {
                    "rank": s["rank"],
                    "hs2": s["hs2"],
                    "description": s["hs_description"],
                    "final_score": s["final_sector_score"],
                    "volume_score": s["volume_score"],
                    "continuity_score": s["continuity_score"],
                    "recent_relevance_score": s["recent_relevance_score"],
                    "forecastability_score": s["forecastability_score"],
                    "development_total_usd": s["development_total_trade_value"],
                    "coverage_pct": round(s["coverage_ratio"] * 100, 1),
                }
                for s in results["directions"]["RU_TR"]["selected_sectors"]
            ],
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.dump(manifest_data, f, sort_keys=False, allow_unicode=True)
        logger.info(f"Persisted frozen manifest to {manifest_path}")

        # 2. Export Metrics Parquet
        records = results["all_metrics_records"]
        metrics_df = pd.DataFrame(records)
        # Convert lists to strings for clean parquet storage
        if "classification_codes" in metrics_df.columns:
            metrics_df["classification_codes"] = metrics_df["classification_codes"].apply(
                lambda x: ",".join(x) if isinstance(x, list) else str(x)
            )
        if "eligibility_failure_reasons" in metrics_df.columns:
            metrics_df["eligibility_failure_reasons"] = metrics_df[
                "eligibility_failure_reasons"
            ].apply(lambda x: "; ".join(x) if isinstance(x, list) else str(x))

        metrics_df.to_parquet(metrics_parquet_path, index=False)
        logger.info(f"Persisted sector selection metrics to {metrics_parquet_path}")

        # 3. Export JSON Results
        # Convert non-serializable objects for JSON
        def _json_serializable(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            return obj

        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, default=_json_serializable, indent=2)
        logger.info(f"Persisted sector selection JSON results to {results_json_path}")

        return manifest_path, metrics_parquet_path, results_json_path
