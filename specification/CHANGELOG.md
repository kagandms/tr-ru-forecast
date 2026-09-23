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
