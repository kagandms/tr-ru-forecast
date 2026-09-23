"""Ingestion pipeline coordinator for monthly bilateral trade flows.

Orchestrates fetching from UN Comtrade, raw snapshot persistence,
standardized normalization, deduplication, local Parquet/CSV storage,
and QA summary generation.
"""

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from tr_ru_forecast.config import Settings, get_settings
from tr_ru_forecast.core.constants import (
    CANONICAL_FLOWS,
    START_PERIOD,
    TradeDirection,
)
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.data.ingestion.comtrade_client import ComtradeClient
from tr_ru_forecast.data.ingestion.transformer import (
    STANDARDIZED_COLUMNS,
    deduplicate_records,
    transform_raw_record,
)
from tr_ru_forecast.data.storage.raw_storage import RawSnapshotStore

logger = get_logger("tr_ru_forecast.data.ingestion.pipeline")


def generate_period_range(start_period: str, end_period: str) -> list[str]:
    """Generate list of monthly periods formatted as YYYYMM.

    Args:
        start_period: Beginning month in 'YYYY-MM' or 'YYYYMM' format.
        end_period: Ending month in 'YYYY-MM' or 'YYYYMM' format.

    Returns:
        Sorted list of period strings in 'YYYYMM' format.
    """
    start_clean = start_period.replace("-", "")
    end_clean = end_period.replace("-", "")

    start_dt = pd.to_datetime(start_clean, format="%Y%m")
    end_dt = pd.to_datetime(end_clean, format="%Y%m")

    if start_dt > end_dt:
        raise ValueError(
            f"start_period ({start_period}) must not be later than end_period ({end_period})"
        )

    periods = pd.date_range(start=start_dt, end=end_dt, freq="MS").strftime("%Y%m").tolist()
    return periods


def resolve_latest_complete_month() -> str:
    """Determine the latest complete calendar month in YYYY-MM format.

    Uses current UTC date: if today is September 2026, the latest complete month is 2026-08.
    """
    now = datetime.now(UTC)
    if now.month == 1:
        return f"{now.year - 1:04d}-12"
    return f"{now.year:04d}-{now.month - 1:02d}"


class TradeIngestionPipeline:
    """Coordinator for UN Comtrade trade data ingestion."""

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

    def run_ingestion_for_direction(
        self,
        direction: TradeDirection,
        start_period: str | None = None,
        end_period: str | None = None,
        use_cache: bool = False,
        fail_closed: bool | None = None,
        periods: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Ingest trade records for a given bilateral direction.

        Args:
            direction: Canonical trade direction (TR_RU or RU_TR).
            start_period: Starting month (default from settings: '2012-01').
            end_period: Ending month (default: latest complete month).
            use_cache: If True, loads local raw snapshots when available before calling API.
            fail_closed: If True, aborts on duplicate conflict.
            periods: Optional explicit list of period strings ('YYYYMM' or 'YYYY-MM').

        Returns:
            tuple (standardized_df, qa_summary_dict)
        """
        if periods:
            period_list = [p.replace("-", "").strip() for p in periods]
            start = period_list[0]
            end = period_list[-1]
        else:
            start = start_period or self.settings.start_period or START_PERIOD
            end = end_period or resolve_latest_complete_month()
            period_list = generate_period_range(start, end)
        direction_info = CANONICAL_FLOWS[direction]
        reporter_code = int(direction_info["reporter_code"])
        partner_code = int(direction_info["partner_code"])
        flow_code = str(direction_info["flow_code"])

        logger.info(
            "Starting ingestion for direction %s: %s -> %s (reporter=%d, partner=%d, periods=%d)",
            direction.value,
            start,
            end,
            reporter_code,
            partner_code,
            len(period_list),
        )

        all_standardized: list[dict[str, Any]] = []
        raw_record_count = 0
        total_requests = 0
        successful_requests = 0
        failed_periods: dict[str, str] = {}
        successful_periods: set[str] = set()
        source_empty_periods: set[str] = set()

        # Identify periods requiring API fetching
        missing_periods: list[str] = []
        for p in period_list:
            cached = self.snapshot_store.load_snapshot(direction.value, p) if use_cache else None
            if cached is None:
                missing_periods.append(p)
            else:
                successful_periods.add(p)

        # Batch fetch missing periods grouped by calendar year
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
                            direction=direction.value,
                            period=p,
                            raw_response={"data": p_recs},
                            request_metadata={
                                "reporter_code": reporter_code,
                                "partner_code": partner_code,
                                "flow_code": flow_code,
                                "batch_requested": True,
                            },
                        )
                except Exception as err:
                    logger.warning(
                        "Yearly batch fetch failed for %s year %s: %s. Falling back to single period fetch.",
                        direction.value,
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
                                direction=direction.value,
                                period=p,
                                raw_response=raw_resp,
                                request_metadata={
                                    "reporter_code": reporter_code,
                                    "partner_code": partner_code,
                                    "flow_code": flow_code,
                                    "batch_requested": False,
                                },
                            )
                        except Exception as single_err:
                            failed_periods[p] = str(single_err)
                            logger.error(
                                "Failed to fetch %s for period %s: %s",
                                direction.value,
                                p,
                                single_err,
                            )

        # Process all periods from local snapshot store
        for period_str in period_list:
            cached = self.snapshot_store.load_snapshot(direction.value, period_str)
            if cached and "response" in cached:
                records = cached["response"].get("data", [])
                raw_record_count += len(records)
                for raw_rec in records:
                    std_rec = transform_raw_record(
                        raw_record=raw_rec,
                        expected_direction=direction.value,
                    )
                    if std_rec is not None:
                        all_standardized.append(std_rec)

        # Deduplicate deterministically using quarantine / fail-closed policy
        dedup_result = deduplicate_records(
            all_standardized,
            fail_closed=self.fail_closed if fail_closed is None else fail_closed,
        )
        clean_records = dedup_result.clean_records

        # Build DataFrame with guaranteed schema
        if clean_records:
            df = pd.DataFrame(clean_records)
            df = df.reindex(columns=STANDARDIZED_COLUMNS)
            df.sort_values(
                by=["period", "is_total", "hs2"], ascending=[True, False, True], inplace=True
            )
            df.reset_index(drop=True, inplace=True)
        else:
            df = pd.DataFrame(columns=STANDARDIZED_COLUMNS)

        # Save outputs (and quarantine file if any conflicts occurred)
        self._save_outputs(
            df=df,
            direction=direction.value,
            quarantined_records=dedup_result.quarantined_records,
        )

        # Generate QA Summary
        qa_summary = self._generate_qa_summary(
            direction=direction.value,
            df=df,
            raw_record_count=raw_record_count,
            duplicate_count=dedup_result.exact_duplicate_count,
            conflict_count=dedup_result.conflict_count,
            conflict_keys=dedup_result.conflict_keys,
            quarantined_count=len(dedup_result.quarantined_records),
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_periods=failed_periods,
            successful_periods=sorted(list(successful_periods)),
            source_empty_periods=sorted(list(source_empty_periods)),
            period_count_requested=len(period_list),
        )

        logger.info(
            "Completed ingestion for %s: raw=%d, standardized=%d, exact_duplicates=%d, conflicts=%d",
            direction.value,
            raw_record_count,
            len(df),
            dedup_result.exact_duplicate_count,
            dedup_result.conflict_count,
        )
        return df, qa_summary

    def _save_outputs(
        self,
        df: pd.DataFrame,
        direction: str,
        quarantined_records: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist standardized data to local Parquet and CSV files."""
        # Ensure directories exist
        self.settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.interim_data_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = self.settings.processed_data_dir / f"trade_monthly_{direction}.parquet"
        csv_path = self.settings.interim_data_dir / f"trade_monthly_{direction}.csv"

        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        df.to_csv(csv_path, index=False)

        if quarantined_records:
            quarantine_path = self.settings.interim_data_dir / f"quarantine_{direction}.csv"
            q_df = pd.DataFrame(quarantined_records)
            q_df.to_csv(quarantine_path, index=False)
            logger.warning(
                "Persisted %d quarantined conflicting records to %s",
                len(quarantined_records),
                quarantine_path,
            )

        logger.info("Saved analytical dataset to %s and %s", parquet_path, csv_path)

    def _generate_qa_summary(
        self,
        direction: str,
        df: pd.DataFrame,
        raw_record_count: int,
        duplicate_count: int,
        conflict_count: int = 0,
        conflict_keys: list[str] | None = None,
        quarantined_count: int = 0,
        total_requests: int = 0,
        successful_requests: int = 0,
        failed_periods: dict[str, str] | None = None,
        successful_periods: list[str] | None = None,
        source_empty_periods: list[str] | None = None,
        period_count_requested: int = 0,
    ) -> dict[str, Any]:
        """Compute minimum ingestion-level QA metrics."""
        conf_keys = conflict_keys or []
        failed = failed_periods or {}
        succ_periods = successful_periods or []
        empty_periods = source_empty_periods or []

        request_integrity = {
            "period_count_requested": period_count_requested,
            "total_requests_made": total_requests,
            "successful_requests": successful_requests,
            "successful_periods_count": len(succ_periods),
            "source_empty_periods_count": len(empty_periods),
            "failed_periods_count": len(failed),
            "failed_periods": failed,
            "source_empty_periods": empty_periods,
            "omitted_due_to_failure": len(failed) > 0,
            "all_missing_confirmed_empty_by_source": len(failed) == 0,
        }

        if df.empty:
            return {
                "direction": direction,
                "period_min": None,
                "period_max": None,
                "raw_record_count": raw_record_count,
                "standardized_record_count": 0,
                "total_record_count": 0,
                "hs2_record_count": 0,
                "duplicate_count": duplicate_count,
                "conflict_count": conflict_count,
                "conflict_keys": conf_keys,
                "quarantined_count": quarantined_count,
                "missing_trade_value_count": 0,
                "zero_trade_value_count": 0,
                "classification_values": [],
                "classification_codes": [],
                "request_integrity": request_integrity,
                "retrieved_at": datetime.now(UTC).isoformat(),
            }

        total_mask = df["is_total"] == True  # noqa: E712
        hs2_mask = df["is_total"] == False  # noqa: E712

        return {
            "direction": direction,
            "period_min": str(df["period"].min()),
            "period_max": str(df["period"].max()),
            "raw_record_count": raw_record_count,
            "standardized_record_count": len(df),
            "total_record_count": int(total_mask.sum()),
            "hs2_record_count": int(hs2_mask.sum()),
            "duplicate_count": duplicate_count,
            "conflict_count": conflict_count,
            "conflict_keys": conf_keys,
            "quarantined_count": quarantined_count,
            "missing_trade_value_count": int(df["trade_value_usd"].isna().sum()),
            "zero_trade_value_count": int((df["trade_value_usd"] == 0.0).sum()),
            "classification_values": sorted(list(df["classification"].dropna().unique())),
            "classification_codes": sorted(list(df["classification_code"].dropna().unique())),
            "request_integrity": request_integrity,
            "retrieved_at": datetime.now(UTC).isoformat(),
        }

    def run_all(
        self,
        start_period: str | None = None,
        end_period: str | None = None,
        use_cache: bool = False,
        fail_closed: bool | None = None,
        periods: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run ingestion for both canonical directions (TR_RU and RU_TR)."""
        summaries: dict[str, Any] = {}
        dfs: list[pd.DataFrame] = []

        for direction in [TradeDirection.TR_RU, TradeDirection.RU_TR]:
            df, summary = self.run_ingestion_for_direction(
                direction=direction,
                start_period=start_period,
                end_period=end_period,
                use_cache=use_cache,
                fail_closed=fail_closed,
                periods=periods,
            )
            summaries[direction.value] = summary
            if not df.empty:
                dfs.append(df)

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.sort_values(
                by=["period", "direction", "is_total", "hs2"],
                ascending=[True, True, False, True],
                inplace=True,
            )
            combined_parquet = self.settings.processed_data_dir / "trade_monthly_canonical.parquet"
            combined_csv = self.settings.interim_data_dir / "trade_monthly_canonical.csv"
            combined_df.to_parquet(combined_parquet, index=False, engine="pyarrow")
            combined_df.to_csv(combined_csv, index=False)
            logger.info("Saved combined canonical dataset to %s", combined_parquet)

        return summaries
