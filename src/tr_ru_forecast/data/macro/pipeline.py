"""Macroeconomic data pipeline orchestrator.

Ingests all 5 required macroeconomic indicators, saves raw snapshots with audit provenance,
normalizes into canonical tabular long format, persists macro_monthly.parquet,
and generates machine-readable audit artifacts.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.data.macro.client import FredMacroClient
from tr_ru_forecast.data.macro.registry import load_macro_source_registry
from tr_ru_forecast.data.macro.transformer import (
    build_linked_ru_ip_series,
    compute_indicator_coverage_audit,
    compute_ru_ip_linkage_diagnostics,
    parse_raw_fred_csv,
    parse_raw_rosstat_xlsx,
    transform_raw_macro_series,
)

logger = get_logger("tr_ru_forecast.data.macro.pipeline")


class MacroDataPipeline:
    """Orchestrates end-to-end macroeconomic data ingestion and normalization."""

    def __init__(
        self,
        registry_path: Path | str | None = None,
        client: FredMacroClient | None = None,
    ) -> None:
        self.registry = load_macro_source_registry(registry_path)
        self.client = client or FredMacroClient()
        self.settings = get_settings()

    def run_pipeline(
        self,
        output_parquet_path: Path | str | None = None,
        output_audit_path: Path | str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run the complete ingestion, normalization, and auditing workflow for all 5 indicators.

        Returns:
            Tuple of (unified_normalized_dataframe, audit_results_dict).
        """
        output_parquet_path = Path(
            output_parquet_path
            or self.settings.processed_data_dir / "macro_monthly.parquet"
        )
        output_audit_path = Path(
            output_audit_path
            or self.settings.processed_data_dir / "macro_data_audit.json"
        )

        all_dfs: list[pd.DataFrame] = []
        indicator_audits: dict[str, Any] = {}
        retrieval_summaries: dict[str, Any] = {}

        logger.info(
            f"Beginning M4 macroeconomic pipeline for {len(self.registry.indicators)} indicators..."
        )

        for ind_id, ind_meta in self.registry.indicators.items():
            logger.info(f"Processing indicator: {ind_id} ({ind_meta.source_series_id})")

            if ind_id == "ru_ip" and ind_meta.composite_structure:
                # 1. Fetch raw OECD data from provider
                raw_text, client_meta = self.client.fetch_series(
                    indicator_id=ind_id,
                    series_id="RUSPROINDMISMEI",
                    url=ind_meta.source_url_or_endpoint,
                    retrieval_method="fred_csv_download",
                )

                # 2. Persist raw snapshot
                raw_path, meta_path = self.client.save_raw_snapshot(
                    indicator_id=ind_id,
                    raw_text=raw_text,
                    audit_meta=client_meta,
                )

                retrieval_summaries[ind_id] = {
                    **client_meta,
                    "snapshot_file_path": str(raw_path),
                    "snapshot_meta_path": str(meta_path),
                }

                # 3. Parse OECD raw and Rosstat raw
                rosstat_xlsx = (
                    self.settings.raw_data_dir
                    / "macro"
                    / "ru_ip"
                    / "rosstat_investigation"
                    / "sezon_2023_07-2026.xlsx"
                )
                df_rosstat_raw = parse_raw_rosstat_xlsx(rosstat_xlsx)
                df_oecd_parsed = parse_raw_fred_csv(raw_text, ind_meta, client_meta["retrieved_at"])

                # 4. Compute and persist linkage diagnostics
                overlap_start = (
                    ind_meta.overlap_window.get("start_period", "2020-01")
                    if ind_meta.overlap_window
                    else "2020-01"
                )
                overlap_end = (
                    ind_meta.overlap_window.get("end_period", "2021-12")
                    if ind_meta.overlap_window
                    else "2021-12"
                )
                diagnostics = compute_ru_ip_linkage_diagnostics(
                    df_oecd=df_oecd_parsed,
                    df_rosstat=df_rosstat_raw,
                    overlap_start=overlap_start,
                    overlap_end=overlap_end,
                )
                diagnostics_path = self.settings.processed_data_dir / "ru_ip_linkage_diagnostics.json"
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                with open(diagnostics_path, "w", encoding="utf-8") as f:
                    json.dump(diagnostics, f, indent=2)
                logger.info(f"Persisted RU IP linkage diagnostics to {diagnostics_path}")

                # 5. Build linked series
                bridge_k = ind_meta.linkage_factor or diagnostics["ratio_diagnostics"]["mean_ratio"]
                switch_d = ind_meta.linkage_switch_date or "2022-01"
                norm_df = build_linked_ru_ip_series(
                    df_oecd_raw=df_oecd_parsed,
                    df_rosstat_raw=df_rosstat_raw,
                    source_meta=ind_meta,
                    retrieved_at=client_meta["retrieved_at"],
                    bridge_factor=bridge_k,
                    switch_date=switch_d,
                )
                all_dfs.append(norm_df)
            else:
                # 1. Fetch raw data from provider
                raw_text, client_meta = self.client.fetch_series(
                    indicator_id=ind_id,
                    series_id=ind_meta.source_series_id,
                    url=ind_meta.source_url_or_endpoint,
                    retrieval_method=ind_meta.retrieval_method,
                )

                # 2. Persist raw snapshot
                raw_path, meta_path = self.client.save_raw_snapshot(
                    indicator_id=ind_id,
                    raw_text=raw_text,
                    audit_meta=client_meta,
                )

                retrieval_summaries[ind_id] = {
                    **client_meta,
                    "snapshot_file_path": str(raw_path),
                    "snapshot_meta_path": str(meta_path),
                }

                # 3. Transform and normalize to canonical schema
                norm_df = transform_raw_macro_series(
                    raw_text=raw_text,
                    source_meta=ind_meta,
                    retrieved_at=client_meta["retrieved_at"],
                )
                all_dfs.append(norm_df)

            # 4. Audit coverage
            audit = compute_indicator_coverage_audit(
                df=norm_df,
                indicator_id=ind_id,
                dev_start=self.settings.development_data_start,
                dev_end=self.settings.development_data_end,
                holdout_start=self.settings.holdout_data_start,
                holdout_end=self.settings.holdout_data_end,
            )
            audit["canonical_unit"] = ind_meta.canonical_unit
            audit["origin_provider"] = ind_meta.origin_provider
            audit["distribution_provider"] = ind_meta.distribution_provider
            audit["underlying_national_source"] = ind_meta.underlying_national_source
            audit["source_provider"] = ind_meta.source_provider
            audit["series_id"] = ind_meta.source_series_id
            audit["quotation_convention"] = ind_meta.quotation_convention
            audit["status"] = ind_meta.status
            audit["fallback_lag_months"] = ind_meta.fallback_lag_months
            audit["release_lag_rule"] = ind_meta.release_lag_rule
            audit["notes"] = ind_meta.notes
            if ind_meta.composite_structure:
                audit["linkage_method"] = ind_meta.linkage_method
                audit["linkage_factor"] = ind_meta.linkage_factor
                audit["linkage_switch_date"] = ind_meta.linkage_switch_date
                audit["diagnostics_file_path"] = str(self.settings.processed_data_dir / "ru_ip_linkage_diagnostics.json")
            indicator_audits[ind_id] = audit

        # 5. Concatenate all normalized records
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # 6. Verify canonical key uniqueness (indicator_id + period)
        duplicates = combined_df[
            combined_df.duplicated(subset=["indicator_id", "period"], keep=False)
        ]
        if not duplicates.empty:
            raise ValueError(
                f"Canonical key uniqueness violation: Found {len(duplicates)} duplicated "
                f"(indicator_id, period) keys in normalized dataset."
            )

        # Sort deterministically
        combined_df.sort_values(by=["indicator_id", "period"], inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        # 7. Persist canonical parquet dataset
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(output_parquet_path, index=False)
        logger.info(
            f"Materialized {len(combined_df)} canonical macro records to {output_parquet_path}"
        )

        # 8. Build and persist machine-readable audit report
        audit_payload = {
            "metadata": {
                "spec_version": "1.1",
                "milestone": "M4",
                "revision": "Revision Gate",
                "created_at": datetime.now(UTC).isoformat(),
                "total_records": len(combined_df),
                "total_indicators": len(self.registry.indicators),
                "period_min": str(combined_df["period"].min()),
                "period_max": str(combined_df["period"].max()),
                "parquet_path": str(output_parquet_path),
            },
            "retrieval_audit": retrieval_summaries,
            "coverage_audit": indicator_audits,
        }

        output_audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_audit_path, "w", encoding="utf-8") as f:
            json.dump(audit_payload, f, indent=2)
        logger.info(f"Persisted macro data audit artifact to {output_audit_path}")

        return combined_df, audit_payload
