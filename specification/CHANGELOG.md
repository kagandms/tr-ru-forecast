# Specification Changelog
# TR–RU Bilateral Trade Forecasting & Intelligence Platform

All major amendments, methodological revisions, and version freezes for the specification pack are recorded here.

---

## [v1.1] — 2026-09-21

### Context & Reason for Amendment
Execution of the Pre-M3 Data Access Gate using authenticated UN Comtrade API access (`scripts/verify_full_history.py`) empirically audited the complete 176-month calendar universe (`2012-01` through `2026-08`). The verification confirmed:
- **`TR_RU` Exporter-Reported (Türkiye):** Continuous 168 complete months (`2012-01` → `2025-12`).
- **`RU_TR` Exporter-Reported (Russia):** Only 108 observed months (`2013-01` → `2021-12`). No observations exist in UN Comtrade after `2021-12`.
- **`RU_TR` Importer-Reported (Türkiye):** Continuous 168 complete months (`2012-01` → `2025-12`).

Consequently, the v1.0 policy of defining both forecast targets strictly as exporter-reported canonical series renders the `RU_TR` direction obsolete for contemporary forecasting and holdout evaluation.

### Key Amendments in v1.1

#### 1. Forecast Target Source Policy
- **TR → RU Forecast Target:**
  - Reporter: Türkiye (`792`)
  - Partner: Russian Federation (`643`)
  - Flow: Export (`X`)
  - Semantic: Türkiye'nin Rusya'ya ihracatı.
- **RU → TR Forecast Target:**
  - Reporter: Türkiye (`792`)
  - Partner: Russian Federation (`643`)
  - Flow: Import (`M`)
  - Semantic: Türkiye'nin Rusya'dan ithalatı.
- Both forecasting targets are now official, uninterrupted Türkiye-reported bilateral trade series.

#### 2. Reference / Mirror Series Policy
- **TR → RU Reference:** Russia reporter (`643`), Türkiye partner (`792`), Import (`M`).
- **RU → TR Reference:** Russia reporter (`643`), Türkiye partner (`792`), Export (`X`).
- Reference series are maintained as a historical data-quality and comparative research layer.
- Reference series are **NOT** forecast targets, **NOT** imputations, **NOT** automatic replacements, and **NOT** averaging sources.

#### 3. Terminology Disambiguation
Analytical and modeling documentation strictly distinguishes:
- **SOURCE SERIES:** Raw data records published by national customs agencies to UN Comtrade.
- **FORECAST TARGET:** The primary official dependent series modeled in forecasting experiments.
- **MIRROR / REFERENCE SERIES:** Secondary counter-reporting flows used for bilateral discrepancy and data quality analysis.

#### 4. Model Data Cutoff Definition
- Defined `LATEST COMPLETE AVAILABLE REPORTING MONTH`: The most recent month for which official trade returns have been submitted and verified (`2025-12` in the verified dataset).
- Modeling Dataset Horizon: `2012-01 → 2025-12` (168 complete monthly observations).
- Periods `2026-01` through `2026-08` have not yet been released by the reporting agency and are excluded from the modeling dataset.

#### 5. Holdout Period Protocol
- The frozen experiment protocol is preserved: **FINAL HOLDOUT LENGTH = 12 MONTHS**.
- **Development Period:** `2012-01 → 2024-12` (156 months).
- **Final Untouched Holdout:** `2025-01 → 2025-12` (12 months).
- Proposal for a 24-month holdout from pre-M3 reports is explicitly rejected; 12 months remains strictly authoritative.

#### 6. M3 Sector Selection Scope
- Sector eligibility filters and ranking algorithms are applied strictly on the **Development Period (`2012-01 → 2024-12`)** over the designated **Forecast Target** series (`TR_RU` export, `RU_TR` import).
- The final 12-month holdout (`2025-01 → 2025-12`) is strictly untouched during sector selection.

#### 7. Neutral Research Interpretation
- The reporting gap is stated objectively: *"No Russia-reported RU→TR bilateral export observations are available in the verified UN Comtrade dataset after 2021-12."*
- Causal or speculative claims regarding the administrative reasons for absence are strictly prohibited.

---

## [v1.0] — 2026-09-20

### Initial Frozen Specification
- Established repository engineering foundation, M0-M15 roadmap, and stop/go gates.
- Defined primary source as UN Comtrade, monthly frequency, HS2 product granularity, and 1/3/6-month horizons.
- Defined mandatory model families: Seasonal Naive, SARIMA, SARIMAX, XGBoost (direct).
- Defined expanding walk-forward validation and untouched 12-month final holdout.
- Initial target definition used exporter-reported canonical flows for both directions.

## M10 implementation record — 2026-10-14 (protocol v1.1 unchanged)

- Added deterministic CV-only champion selection, 168-candidate leaderboard, 42-task registry, operational fallback references, eligibility audit and content provenance.
- Reused the frozen 281 monthly-origin calendar (96/94/91 by horizon), sMAPE primary / MAE tie-break and strictly <2% relative-gain parsimony rule. The pasted five-fold/RMSE-first proposal was not adopted because it conflicts with the authoritative protocol.
- Negative point predictions invalidate their task/model candidate; observed scores remain available for audit. Worst-origin stability is diagnostic and does not override the frozen champion rule.
- M9 A2 maintenance findings and the recorded M8 validation-manifest hash discrepancy remain open. Row-level CV calendar alignment is independently checked. M10 implementation does not close those prerequisite reviews or authorize M11 holdout access.

## M11 pre-score amendment — 2026-10-14

The user explicitly authorized retrospective latest-vintage evaluation without external publication/vintage archives, monthly expanding holdout refits and chronological holdout residual updates after a CV seed. Experiment Protocol §19 defines the departure from the former development-only refit and CV-only calibration wording before scoring. Champion assignments, frozen orders/hyperparameters, features, metrics, interval levels and the 12-month holdout remain unchanged.

## M12 artifact API implementation — 2026-10-16

- The user's M12 prompt selects read-only disk-artifact serving of frozen M10/M11 and mirror/history/macro outputs. Added typed Pydantic v2 contracts, validated GET routes, immutable repository/service projections, configured CORS, load-time checksums and structured degraded readiness. No numerical protocol/model/artifact changes.
- Global and task CV/holdout metrics, governed fallback intervals and actual XGBoost SHAP are available to M13. M2 exporter/importer gap semantics and missing post-2022 mirror pairs remain explicit.
- Future inference model serialization, PostgreSQL and M13 UI remain separate work; current M12 does not claim a production deployment or real-time vintage fidelity.

## M10–M13 acceptance synchronization — 2026-10-16

- M10 and M11 acceptance reports record completion of the frozen CV-only champion registry and the authorized retrospective latest-vintage 2025 holdout evaluation under deterministic lag rules. The latter is not historical vintage replay or proof of decision-time availability.
- M12 acceptance records the typed, read-only, checksum-verified local artifact API. M13 acceptance records the five-module Next.js dashboard, API/snapshot modes, deterministic bundled snapshot and local build/browser checks. Neither milestone asserts hosted deployment or real-user UAT.
- The accepted M11 scope retains 1,512 role predictions and 9,072 local SHAP rows. SHAP is predictive attribution; mirror observations are paired reporter flows supplied through UN Comtrade, not three independently verified national feeds.
- The M13 frontend snapshot remains pinned to SHA-256 `867a3d8230ca62c23a2f7bc4e7021e0f9f8eba0fe8a727469f1c11b07bebf574`; the 102-file processed-artifact ledger remains the integrity reference.

## M14 portfolio documentation — 2026-10-16

- Rebuilt the repository README as an evidence-led portfolio entry point with results, architecture, methodology limits, API/UI guide, reproducibility commands and M3–M14 report links. Corrected the mirror-source description to match the local artifacts. Verification is in `docs/M14_PORTFOLIO_ACCEPTANCE_REPORT.md`.
- Synchronized milestone status and specification index. M14 remains in progress because the broader research package (working paper, abstracts, data dictionary and release materials) is not complete; M15 deployment/release acceptance remains separate.
