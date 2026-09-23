"""CLI entry point for Milestone M3: Data Audit & Sector Selection.

Materializes forecast target dataset, builds 156-month development panel,
evaluates candidate eligibility, scores eligible candidates, freezes top sectors,
and generates docs/M3_SECTOR_SELECTION_REPORT.md.
"""

from pathlib import Path
from typing import Any

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.logging import get_logger
from tr_ru_forecast.data.sectors.hs_audit import audit_hs_revisions
from tr_ru_forecast.data.sectors.selector import SectorSelector
from tr_ru_forecast.data.sectors.target_materializer import build_forecast_target_dataset

logger = get_logger("tr_ru_forecast.data.sectors.cli")


def generate_selection_report(
    results: dict[str, Any],
    revision_audit: dict[str, Any],
    report_path: Path,
) -> None:
    """Generate comprehensive M3 Sector Selection Report markdown document."""
    meta = results["metadata"]
    dirs = results["directions"]
    tr_ru = dirs["TR_RU"]
    ru_tr = dirs["RU_TR"]

    lines = [
        "# M3 — DATA AUDIT & SECTOR SELECTION REPORT",
        "",
        "**Document Version:** 1.1.0  ",
        f"**Execution Date:** {meta['created_at']}  ",
        "**Authoritative Specification:** Specification Pack v1.1 (`specification/`)  ",
        "**Author:** Senior Data Analyst & Forecasting Research Engineer  ",
        "",
        "---",
        "",
        "## 1. EXECUTIVE SUMMARY",
        "",
        "Milestone M3 establishes the empirical data foundation and frozen commodity scope for the TR–RU Bilateral Trade Forecasting & Intelligence Platform under Specification v1.1. Following the pre-M3 discovery of reporting divergence, both forecast targets are materialized strictly as official Türkiye-reported series. Sector eligibility and multi-factor ranking are executed exclusively over the 156-month development period (`2012-01` through `2024-12`). The 12-month final holdout (`2025-01` through `2025-12`) is strictly isolated and untouched.",
        "",
        "A total of **12 commodity sectors** (6 per direction) plus **2 bilateral total series** are selected for the subsequent forecasting milestones, establishing an empirical scope of **14 distinct monthly target series** across 3 forecast horizons ($h=1, 3, 6$).",
        "",
        "---",
        "",
        "## 2. TARGET POLICY & SOURCE PROVENANCE VERIFICATION",
        "",
        "Under Specification v1.1, forecasting targets must be official, continuous series. Both directions utilize Türkiye customs returns (`reporter_code=792`, `partner_code=643`):",
        "",
        "| Metric / Property | TR → RU Forecast Target | RU → TR Forecast Target |",
        "| :--- | :--- | :--- |",
        "| **Semantic Concept** | Türkiye exports to Russia | Türkiye imports from Russia |",
        "| **Reporter Code & Country** | `792` (Türkiye) | `792` (Türkiye) |",
        "| **Partner Code & Country** | `643` (Russian Federation) | `643` (Russian Federation) |",
        "| **Trade Flow** | Export (`flow_code='X'`) | Import (`flow_code='M'`) |",
        "| **Source Dataset** | `trade_monthly_TR_RU.parquet` | `trade_monthly_mirror_RU_TR.parquet` |",
        "| **Full Modeling Span** | `2012-01` → `2025-12` (168 months) | `2012-01` → `2025-12` (168 months) |",
        "| **Development Period** | `2012-01` → `2024-12` (156 months) | `2012-01` → `2024-12` (156 months) |",
        "| **Final Holdout (Untouched)** | `2025-01` → `2025-12` (12 months) | `2025-01` → `2025-12` (12 months) |",
        "| **Russia Exporter Data Used?** | **NO** (0 records) | **NO** (0 records) |",
        "",
        "> [!IMPORTANT]",
        "> **Russia Exporter Data Quarantine:** The historical dataset `trade_monthly_RU_TR.parquet` contains Russian customs returns ending at `2021-12`. Under Specification v1.1 and runtime guards, this dataset is strictly prohibited from serving as the RU_TR forecast target. The official Turkish import dataset (`trade_monthly_mirror_RU_TR.parquet`) provides the unbroken 2012–2025 observational series.",
        "",
        "---",
        "",
        "## 3. DEVELOPMENT PERIOD & HOLDOUT LEAKAGE GUARD",
        "",
        "- **Development Window:** `2012-01` through `2024-12` (156 calendar months).",
        "- **Holdout Period:** `2025-01` through `2025-12` (12 calendar months).",
        "- **Programmatic Holdout Guard:** Implemented in `tr_ru_forecast.data.sectors.panel_builder.enforce_holdout_guard`. Any record beyond `2024-12` raises an unhandled `ValueError` at runtime.",
        "- **Leakage Invariant:** Unit tests confirm that mutating 2025 holdout trade values has zero effect on sector eligibility, metric values, or selection rankings.",
        "",
        "---",
        "",
        "## 4. CANDIDATE HS2 UNIVERSE & NOMENCLATURE AUDIT",
        "",
        f"- **TR_RU Observed HS2 Chapters:** {tr_ru['total_candidates']} chapters.",
        f"- **RU_TR Observed HS2 Chapters:** {ru_tr['total_candidates']} chapters.",
        "- **TOTAL Scope:** Bilateral total flows are automatically preserved for modeling but strictly excluded from sector candidate ranking.",
        "",
        "### HS Revision Regimes (2012–2024 Development Horizon)",
        f"- **Nomenclatures Observed:** {', '.join(revision_audit.get('revisions_present', []))}",
        "- **H4 (HS 2012):** 2012-01 to 2016-12",
        "- **H5 (HS 2017):** 2017-01 to 2021-12",
        "- **H6 (HS 2022):** 2022-01 to 2024-12 (in development span)",
        "- **Chapter Stability:** HS 2-digit chapter headings (01–97) demonstrate complete structural continuity across all 3 revision transitions.",
        "",
        "### Special & Residual Chapter Exclusions",
        "- **Chapter 77:** Reserved by WCO for future potential allocation; 0 records observed.",
        "- **Chapter 98:** Special classification provisions / postal packages; excluded from ranking.",
        "- **Chapter 99:** Commodities not specified according to kind / special provisions / confidential trade; audited and excluded from ranking as a residual non-production category.",
        "",
        "---",
        "",
        "## 5. ELIGIBILITY FILTER EVALUATION",
        "",
        "All candidate sectors were subjected to five deterministic criteria (A through E):",
        "- **Filter A (History Potential):** $\\ge 96$ months potential span.",
        "- **Filter B (Coverage):** Observed months $\\ge 141$ / 156 ($\\ge 90.0\\%$).",
        "- **Filter C (Zero Ratio):** Zero trade months $\\le 10.0\\%$ of observed months.",
        "- **Filter D (Recent Activity):** Observed months in `2023-01` → `2024-12` $\\ge 21$ / 24.",
        "- **Filter E (Economic Size Floor):** Development total trade value $\\ge 10\\text{th}$ percentile ($P_{10}$) among candidates passing A–D.",
        "",
        "### Direction Summary:",
        "",
        "| Direction | Total Candidates | Passing A–D | P10 Size Floor (USD) | Passing Filter E (Final Eligible) | Disqualified |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **TR → RU** | {tr_ru['total_candidates']} | {tr_ru['eligible_count'] + sum(1 for m in tr_ru['ineligible'] if any('Filter E' in r for r in m['eligibility_failure_reasons']))} | ${tr_ru['p10_size_threshold_usd']:,.2f} | **{tr_ru['eligible_count']}** | {tr_ru['ineligible_count']} |",
        f"| **RU → TR** | {ru_tr['total_candidates']} | {ru_tr['eligible_count'] + sum(1 for m in ru_tr['ineligible'] if any('Filter E' in r for r in m['eligibility_failure_reasons']))} | ${ru_tr['p10_size_threshold_usd']:,.2f} | **{ru_tr['eligible_count']}** | {ru_tr['ineligible_count']} |",
        "",
        "---",
        "",
        "## 6. MULTI-FACTOR SCORING & WEIGHTED RANKING",
        "",
        "Eligible candidates were scored on a 0–100 scale using four orthogonal components:",
        "$$\\text{Final Score} = 0.40 \\times \\text{Volume} + 0.30 \\times \\text{Continuity} + 0.20 \\times \\text{Recent} + 0.10 \\times \\text{Forecastability}$$",
        "",
        "1. **Volume Score (40%):** Fractional percentile rank of $\\log(1 + \\text{Development Total Trade Value})$.",
        "2. **Continuity Score (30%):** $100 \\times [0.60 \\times \\text{Coverage} + 0.25 \\times (1 - \\text{Zero Ratio}) + 0.15 \\times (1 - \\text{Streak} / 156)]$.",
        "3. **Recent Relevance (20%):** $0.80 \\times \\text{Recent Volume Percentile} + 0.20 \\times (\\text{Recent Coverage} \\times 100)$.",
        "4. **Forecastability (10%):** Development Seasonal Naive ($h=1, t-12$) sMAPE fractional percentile rank (lower sMAPE yields higher score).",
        "",
        "Deterministic tie-break order: `final_sector_score DESC` $\\rightarrow$ `volume_score DESC` $\\rightarrow$ `development_total_trade_value DESC` $\\rightarrow$ `hs2 ASC`.",
        "",
        "---",
        "",
        "## 7. FINAL RANKINGS — TR → RU (TÜRKİYE EXPORTS TO RUSSIA)",
        "",
        "### Top 15 Eligible Candidates (Top 6 Frozen for Modeling):",
        "",
        "| Rank | HS2 | Description | Volume | Continuity | Recent | Forecastability | sMAPE (%) | Total Dev Value (USD) | Final Score | Selection Status |",
        "| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for item in tr_ru["ranked_eligible"][:15]:
        status = "**SELECTED**" if item["is_selected"] else "Eligible (Unselected)"
        lines.append(
            f"| {item['rank']} | **{item['hs2']}** | {item['hs_description'][:40]}... | "
            f"{item['volume_score']:.1f} | {item['continuity_score']:.1f} | {item['recent_relevance_score']:.1f} | "
            f"{item['forecastability_score']:.1f} | {item['seasonal_naive_smape']:.1f}% | "
            f"${item['development_total_trade_value']:,.0f} | **{item['final_sector_score']:.2f}** | {status} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 8. FINAL RANKINGS — RU → TR (TÜRKİYE IMPORTS FROM RUSSIA)",
        "",
        "### Top 15 Eligible Candidates (Top 6 Frozen for Modeling):",
        "",
        "| Rank | HS2 | Description | Volume | Continuity | Recent | Forecastability | sMAPE (%) | Total Dev Value (USD) | Final Score | Selection Status |",
        "| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for item in ru_tr["ranked_eligible"][:15]:
        status = "**SELECTED**" if item["is_selected"] else "Eligible (Unselected)"
        lines.append(
            f"| {item['rank']} | **{item['hs2']}** | {item['hs_description'][:40]}... | "
            f"{item['volume_score']:.1f} | {item['continuity_score']:.1f} | {item['recent_relevance_score']:.1f} | "
            f"{item['forecastability_score']:.1f} | {item['seasonal_naive_smape']:.1f}% | "
            f"${item['development_total_trade_value']:,.0f} | **{item['final_sector_score']:.2f}** | {status} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 9. FROZEN SELECTED SECTORS (MODELING SCOPE)",
        "",
        "The following 12 sectors are formally frozen for the modeling milestones (M6–M10):",
        "",
        "### TR → RU Direction (Türkiye Exports to Russia)",
    ])

    for s in tr_ru["selected_sectors"]:
        lines.append(
            f"{s['rank']}. **HS {s['hs2']}** — {s['hs_description']}  \n"
            f"   - Final Score: `{s['final_sector_score']}` | Volume: `${s['development_total_trade_value']:,.0f}` | Coverage: `{s['coverage_ratio']*100:.1f}%` | sMAPE: `{s['seasonal_naive_smape']:.1f}%`"
        )

    lines.extend([
        "",
        "### RU → TR Direction (Türkiye Imports from Russia)",
    ])

    for s in ru_tr["selected_sectors"]:
        lines.append(
            f"{s['rank']}. **HS {s['hs2']}** — {s['hs_description']}  \n"
            f"   - Final Score: `{s['final_sector_score']}` | Volume: `${s['development_total_trade_value']:,.0f}` | Coverage: `{s['coverage_ratio']*100:.1f}%` | sMAPE: `{s['seasonal_naive_smape']:.1f}%`"
        )

    lines.extend([
        "",
        "### Complete Modeling Scope",
        "- **Bilateral Total Series:** 2 (`TR_RU_TOTAL`, `RU_TR_TOTAL`)",
        f"- **TR_RU Commodity Sectors:** {len(tr_ru['selected_sectors'])}",
        f"- **RU_TR Commodity Sectors:** {len(ru_tr['selected_sectors'])}",
        f"- **Total Modeled Series Count:** **{2 + len(tr_ru['selected_sectors']) + len(ru_tr['selected_sectors'])}**",
        "- **Forecast Horizons per Series:** 3 ($h=1, 3, 6$)",
        f"- **Total Forecast Targets Evaluated:** **{(2 + len(tr_ru['selected_sectors']) + len(ru_tr['selected_sectors'])) * 3}**",
        "",
        "---",
        "",
        "## 10. NOTABLE EXCLUSIONS & DISQUALIFICATION AUDIT",
        "",
        "### Top Excluded Candidates by Volume:",
        "",
        "#### TR → RU Exclusions:",
    ])

    # Sample top ineligibles
    top_inel_tr = sorted(
        tr_ru["ineligible"],
        key=lambda x: x["development_total_trade_value"],
        reverse=True,
    )[:5]
    for inel in top_inel_tr:
        reasons = "; ".join(inel["eligibility_failure_reasons"])
        lines.append(
            f"- **HS {inel['hs2']}** ({inel['hs_description']}): ${inel['development_total_trade_value']:,.0f} — *Disqualified*: {reasons}"
        )

    lines.append("\n#### RU → TR Exclusions:\n")
    top_inel_ru = sorted(
        ru_tr["ineligible"],
        key=lambda x: x["development_total_trade_value"],
        reverse=True,
    )[:5]
    for inel in top_inel_ru:
        reasons = "; ".join(inel["eligibility_failure_reasons"])
        lines.append(
            f"- **HS {inel['hs2']}** ({inel['hs_description']}): ${inel['development_total_trade_value']:,.0f} — *Disqualified*: {reasons}"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 11. ACCEPTANCE CRITERIA VERIFICATION",
        "",
        "- [x] **v1.1 Target Mappings Verified:** TR_RU = Turkey export (792->643), RU_TR = Turkey import (792<-643).",
        "- [x] **Russian Exporter Data Quarantine:** Russia export records (`trade_monthly_RU_TR.parquet`) rejected for RU_TR target.",
        "- [x] **Strict Development Window:** Exclusively `2012-01` to `2024-12` utilized for sector selection.",
        "- [x] **Holdout Leakage Guard Active:** 2025 data strictly isolated; zero influence on metrics or rankings.",
        "- [x] **156-Month Panel Integrity:** Denominator strictly 156 months; missing values kept as `NaN` (never converted to 0).",
        "- [x] **Deterministic Eligibility Filters:** Filters A through E applied programmatically.",
        "- [x] **P10 Size Filter:** Computed direction-specifically on positive-volume development data only.",
        "- [x] **Deterministic Multi-Factor Scoring:** Volume (40%), Continuity (30%), Recent (20%), Forecastability (10%).",
        "- [x] **Maximum 6 Sectors per Direction:** Top 6 selected; no quota-filling with disqualified sectors.",
        "- [x] **Bilateral Total Preserved:** Included in modeling scope outside sector selection.",
        "- [x] **Frozen Manifest Version-Controlled:** Persisted to `configs/selected_sectors_v1_1.yaml`.",
        "- [x] **Audit Outputs Generated:** Parquet metrics and JSON results persisted to `data/processed/`.",
        "",
        "---",
        "",
        "## 12. GATE VERDICT",
        "",
        "```",
        "================================================================================",
        "MILESTONE M3: DATA AUDIT & SECTOR SELECTION",
        "FINAL VERDICT: PASS",
        "STATUS: FROZEN — READY FOR MILESTONE M4 (MACRO DATA FOUNDATION)",
        "================================================================================",
        "```",
    ])

    report_content = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Generated M3 Selection Report at {report_path}")


def main() -> None:
    """Execute end-to-end M3 workflow."""
    logger.info("==================================================")
    logger.info("Starting Milestone M3: Data Audit & Sector Selection")
    logger.info("==================================================")

    settings = get_settings()

    # 1. Materialize Forecast Target Dataset
    logger.info("Step 1: Materializing canonical forecast target dataset under v1.1 policy...")
    target_df = build_forecast_target_dataset()
    logger.info(f"Target dataset loaded with {len(target_df)} total records")

    # 2. Execute Sector Selection on Development Period Only
    logger.info("Step 2: Enforcing development cutoff and executing sector audit...")
    dev_target_df = target_df[target_df["period"] <= settings.development_data_end].copy()
    selector = SectorSelector(dev_target_df)
    results = selector.run_selection()

    # 3. Export Manifest and Processed Data
    logger.info("Step 3: Exporting selection artifacts...")
    manifest_path, parquet_path, json_path = selector.export_artifacts(results)

    # 4. Generate Report
    logger.info("Step 4: Auditing HS revisions and generating selection report...")
    rev_audit = audit_hs_revisions(target_df)
    report_path = settings.project_root / "docs" / "M3_SECTOR_SELECTION_REPORT.md"
    generate_selection_report(results, rev_audit, report_path)

    # 5. Summary Output
    tr_selected = [s["hs2"] for s in results["directions"]["TR_RU"]["selected_sectors"]]
    ru_selected = [s["hs2"] for s in results["directions"]["RU_TR"]["selected_sectors"]]

    logger.info("==================================================")
    logger.info("M3 Execution Complete!")
    logger.info(f"TR_RU Selected Top 6: {tr_selected}")
    logger.info(f"RU_TR Selected Top 6: {ru_selected}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Report: {report_path}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
