"""Orchestration pipeline for mirror data ingestion, pairing, and bilateral QA.

Coordinates mirror data ingestion from UN Comtrade, raw snapshot storage,
standardization, conflict quarantine, deterministic pairing against canonical data,
and comprehensive coverage and gap analytics.
"""

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from tr_ru_forecast.config import Settings, get_settings
from tr_ru_forecast.core.constants import (
    MIRROR_FLOWS,
    MIRROR_MONTHLY_COLUMNS,
    START_PERIOD,
    TradeDirection,
)
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.data.ingestion.comtrade_client import ComtradeClient
from tr_ru_forecast.data.ingestion.pipeline import (
    TradeIngestionPipeline,
    generate_period_range,
    resolve_latest_complete_month,
)
from tr_ru_forecast.data.ingestion.transformer import STANDARDIZED_COLUMNS
from tr_ru_forecast.data.quality.coverage_analyzer import BilateralCoverageAnalyzer
from tr_ru_forecast.data.quality.mirror_pairing import pair_canonical_and_mirror
from tr_ru_forecast.data.quality.mirror_transformer import (
    deduplicate_mirror_records,
    transform_mirror_raw_record,
)
from tr_ru_forecast.data.storage.raw_storage import RawSnapshotStore

logger = get_logger("tr_ru_forecast.data.quality.mirror_pipeline")


class MirrorQualityPipeline:
    """End-to-end coordinator for mirror data ingestion, pairing, and quality analysis."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: ComtradeClient | None = None,
        snapshot_store: RawSnapshotStore | None = None,
        fail_closed: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or ComtradeClient(settings=self.settings)
        self.snapshot_store = snapshot_store or RawSnapshotStore(settings=self.settings)
        self.fail_closed = fail_closed

    def ingest_mirror_for_direction(
        self,
        direction: TradeDirection,
        start_period: str | None = None,
        end_period: str | None = None,
        use_cache: bool = False,
        fail_closed: bool | None = None,
        periods: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Ingest and standardize mirror import records for a bilateral direction.

        Raw snapshots are safely stored in 'mirror_{direction}' subdirectories
        to prevent any collision with canonical snapshots.
        """
        if periods:
            period_list = [p.replace("-", "").strip() for p in periods]
            start = period_list[0]
            end = period_list[-1]
        else:
            start = start_period or self.settings.start_period or START_PERIOD
            end = end_period or resolve_latest_complete_month()
            period_list = generate_period_range(start, end)
        mirror_info = MIRROR_FLOWS[direction]
        reporter_code = int(mirror_info["reporter_code"])
        partner_code = int(mirror_info["partner_code"])
        flow_code = str(mirror_info["flow_code"])

        raw_direction_key = f"mirror_{direction.value}"

        logger.info(
            "Starting mirror ingestion for %s (reporter=%d, partner=%d, flow=%s, periods=%d)",
            direction.value,
            reporter_code,
            partner_code,
            flow_code,
            len(period_list),
        )

        all_standardized: list[dict[str, Any]] = []
        raw_record_count = 0
        total_requests = 0
        successful_requests = 0
        failed_periods: dict[str, str] = {}
        successful_periods: set[str] = set()
        source_empty_periods: set[str] = set()

        # Identify missing mirror periods requiring API fetching
        missing_periods: list[str] = []
        for p in period_list:
            cached = self.snapshot_store.load_snapshot(raw_direction_key, p) if use_cache else None
            if cached is None:
                missing_periods.append(p)
            else:
                successful_periods.add(p)

        # Batch fetch missing mirror periods grouped by calendar year
        if missing_periods:
            years: dict[str, list[str]] = {}
            for p in missing_periods:
                yr = p[:4]
                years.setdefault(yr, []).append(p)

            for yr, yr_periods in sorted(years.items()):
                total_requests += 1
                batch_str = ",".join(yr_periods)
                try:
                    raw_response = self.client.fetch_monthly_flow(
                        period=batch_str,
                        reporter_code=reporter_code,
                        partner_code=partner_code,
                        flow_code=flow_code,
                    )
                    successful_requests += 1
                    batch_records = (
                        raw_response.get("data", []) if isinstance(raw_response, dict) else []
                    )
                    by_period: dict[str, list[dict[str, Any]]] = {p: [] for p in yr_periods}
                    for rec in batch_records:
                        rec_p = str(rec.get("period", ""))
                        if rec_p in by_period:
                            by_period[rec_p].append(rec)

                    for p, p_recs in by_period.items():
                        successful_periods.add(p)
                        if not p_recs:
                            source_empty_periods.add(p)
                        self.snapshot_store.save_snapshot(
                            direction=raw_direction_key,
                            period=p,
                            raw_response={"data": p_recs},
                            request_metadata={
                                "reporter_code": reporter_code,
                                "partner_code": partner_code,
                                "flow_code": flow_code,
                                "physical_direction": direction.value,
                                "batch_requested": True,
                            },
                        )
                except Exception as err:
                    logger.warning(
                        "Yearly mirror batch fetch failed for %s year %s: %s. Falling back to single period fetch.",
                        raw_direction_key,
                        yr,
                        err,
                    )
                    for p in yr_periods:
                        total_requests += 1
                        try:
                            raw_resp = self.client.fetch_monthly_flow(
                                period=p,
                                reporter_code=reporter_code,
                                partner_code=partner_code,
                                flow_code=flow_code,
                            )
                            successful_requests += 1
                            successful_periods.add(p)
                            p_recs = raw_resp.get("data", []) if isinstance(raw_resp, dict) else []
                            if not p_recs:
                                source_empty_periods.add(p)
                            self.snapshot_store.save_snapshot(
                                direction=raw_direction_key,
                                period=p,
                                raw_response=raw_resp,
                                request_metadata={
                                    "reporter_code": reporter_code,
                                    "partner_code": partner_code,
                                    "flow_code": flow_code,
                                    "physical_direction": direction.value,
                                    "batch_requested": False,
                                },
                            )
                        except Exception as single_err:
                            failed_periods[p] = str(single_err)
                            logger.error(
                                "Failed to fetch mirror %s for period %s: %s",
                                direction.value,
                                p,
                                single_err,
                            )

        # Process all periods from local snapshot store
        for period_str in period_list:
            cached = self.snapshot_store.load_snapshot(raw_direction_key, period_str)
            if cached and "response" in cached:
                records = cached["response"].get("data", [])
                raw_record_count += len(records)
                for raw_rec in records:
                    std_rec = transform_mirror_raw_record(
                        raw_record=raw_rec,
                        expected_direction=direction.value,
                    )
                    if std_rec is not None:
                        all_standardized.append(std_rec)

        # Deduplicate deterministically with quarantine policy
        dedup_result = deduplicate_mirror_records(
            all_standardized,
            fail_closed=self.fail_closed if fail_closed is None else fail_closed,
        )
        clean_records = dedup_result.clean_records

        if clean_records:
            df = pd.DataFrame(clean_records)
            df = df.reindex(columns=STANDARDIZED_COLUMNS)
            df.sort_values(
                by=["period", "is_total", "hs2"], ascending=[True, False, True], inplace=True
            )
            df.reset_index(drop=True, inplace=True)
        else:
            df = pd.DataFrame(columns=STANDARDIZED_COLUMNS)

        # Save mirror outputs
        self._save_mirror_outputs(
            df=df,
            direction=direction.value,
            quarantined_records=dedup_result.quarantined_records,
        )

        qa_info = {
            "direction": direction.value,
            "raw_record_count": raw_record_count,
            "standardized_count": len(df),
            "exact_duplicate_count": dedup_result.exact_duplicate_count,
            "conflict_count": dedup_result.conflict_count,
            "conflict_keys": dedup_result.conflict_keys,
            "quarantined_count": len(dedup_result.quarantined_records),
            "request_integrity": {
                "period_count_requested": len(period_list),
                "total_requests_made": total_requests,
                "successful_requests": successful_requests,
                "successful_periods_count": len(successful_periods),
                "source_empty_periods_count": len(source_empty_periods),
                "failed_periods_count": len(failed_periods),
                "failed_periods": failed_periods,
                "source_empty_periods": sorted(list(source_empty_periods)),
                "omitted_due_to_failure": len(failed_periods) > 0,
                "all_missing_confirmed_empty_by_source": len(failed_periods) == 0,
            },
        }

        return df, qa_info

    def pair_and_evaluate_direction(
        self,
        direction: TradeDirection,
        start_period: str | None = None,
        end_period: str | None = None,
        use_cache: bool = False,
        fail_closed: bool | None = None,
        periods: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Ingest mirror data and pair with canonical data for a given direction."""
        # 1. Ingest mirror data
        mirror_df, mirror_qa = self.ingest_mirror_for_direction(
            direction=direction,
            start_period=start_period,
            end_period=end_period,
            use_cache=use_cache,
            fail_closed=fail_closed,
            periods=periods,
        )

        # 2. Load or ingest canonical data
        canonical_path = (
            self.settings.processed_data_dir / f"trade_monthly_{direction.value}.parquet"
        )
        if canonical_path.exists() and use_cache and not periods:
            canonical_df = pd.read_parquet(canonical_path)
            logger.info(
                "Loaded canonical dataset from %s (%d rows)", canonical_path, len(canonical_df)
            )
        else:
            logger.info("Loading or refreshing canonical data for direction %s...", direction.value)
            can_pipeline = TradeIngestionPipeline(
                settings=self.settings,
                client=self.client,
                snapshot_store=self.snapshot_store,
                fail_closed=self.fail_closed if fail_closed is None else fail_closed,
            )
            canonical_df, _ = can_pipeline.run_ingestion_for_direction(
                direction=direction,
                start_period=start_period,
                end_period=end_period,
                use_cache=use_cache,
                fail_closed=fail_closed,
                periods=periods,
            )

        # 3. Pair datasets
        paired_df = pair_canonical_and_mirror(
            canonical_df=canonical_df,
            mirror_df=mirror_df,
            direction=direction,
        )

        return paired_df, mirror_qa

    def run_all(
        self,
        start_period: str | None = None,
        end_period: str | None = None,
        use_cache: bool = False,
        fail_closed: bool | None = None,
        periods: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run complete mirror pipeline for all bilateral trade directions and persist mirror_monthly."""
        all_paired: list[pd.DataFrame] = []
        ingestion_summaries: dict[str, Any] = {}

        for direction in [TradeDirection.TR_RU, TradeDirection.RU_TR]:
            paired_dir_df, qa_info = self.pair_and_evaluate_direction(
                direction=direction,
                start_period=start_period,
                end_period=end_period,
                use_cache=use_cache,
                fail_closed=fail_closed,
                periods=periods,
            )
            all_paired.append(paired_dir_df)
            ingestion_summaries[direction.value] = qa_info

        if all_paired:
            combined_df = pd.concat(all_paired, ignore_index=True)
            combined_df = combined_df.reindex(columns=MIRROR_MONTHLY_COLUMNS)
            combined_df.sort_values(by=["period", "direction", "hs2"], inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
        else:
            combined_df = pd.DataFrame(columns=MIRROR_MONTHLY_COLUMNS)

        # Persist unified mirror_monthly datasets
        self._save_mirror_monthly(combined_df)

        # Determine expected period count
        start = start_period or self.settings.start_period or START_PERIOD
        end = end_period or resolve_latest_complete_month()
        exp_periods = len(periods) if periods else len(generate_period_range(start, end))

        # Run Bilateral Coverage Analyzer with full history safety
        analyzer = BilateralCoverageAnalyzer(
            paired_df=combined_df,
            full_history_verified=bool(self.client.is_authenticated),
            expected_period_count=exp_periods,
        )
        audit_report = analyzer.generate_full_report()

        full_summary = {
            "retrieved_at": datetime.now(UTC).isoformat(),
            "pipeline": "M2_Mirror_Statistics",
            "verification_scope": "FULL" if self.client.is_authenticated else "PARTIAL",
            "full_history_verified": bool(self.client.is_authenticated),
            "expected_period_count": exp_periods,
            "observed_period_count": audit_report.get("observed_period_count", 0),
            "authentication_mode": "Subscription"
            if self.client.is_authenticated
            else "Public_Preview",
            "ingestion_summaries": ingestion_summaries,
            "quality_audit": audit_report,
        }

        # Save summary JSON
        summary_path = self.settings.processed_data_dir / "mirror_qa_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(full_summary, f, indent=2)
        logger.info("Persisted mirror QA summary to %s", summary_path)

        return combined_df, full_summary

    def _save_mirror_outputs(
        self,
        df: pd.DataFrame,
        direction: str,
        quarantined_records: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist standardized mirror datasets."""
        self.settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.interim_data_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = (
            self.settings.processed_data_dir / f"trade_monthly_mirror_{direction}.parquet"
        )
        csv_path = self.settings.interim_data_dir / f"trade_monthly_mirror_{direction}.csv"

        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        df.to_csv(csv_path, index=False)

        if quarantined_records:
            quarantine_path = self.settings.interim_data_dir / f"quarantine_mirror_{direction}.csv"
            pd.DataFrame(quarantined_records).to_csv(quarantine_path, index=False)
            logger.warning(
                "Persisted %d quarantined mirror records to %s",
                len(quarantined_records),
                quarantine_path,
            )

    def _save_mirror_monthly(self, df: pd.DataFrame) -> None:
        """Persist combined mirror_monthly dataset."""
        self.settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.interim_data_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = self.settings.processed_data_dir / "mirror_monthly.parquet"
        csv_path = self.settings.interim_data_dir / "mirror_monthly.csv"

        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        df.to_csv(csv_path, index=False)
        logger.info(
            "Saved unified mirror analytical dataset to %s and %s (%d rows)",
            parquet_path,
            csv_path,
            len(df),
        )
