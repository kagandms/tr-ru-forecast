# MILESTONES.md
# Implementation Roadmap & Acceptance Gates

**Version:** 1.1  
**Status:** M0–M13 implementation/acceptance recorded; M14 portfolio packaging in progress (v1.1)
**Previous Version:** 1.0 (Frozen initial release)  
**Changelog:** [`CHANGELOG.md`](CHANGELOG.md)

Rule: coding agent receives **one milestone at a time**.

## M0 — Repository & Engineering Foundation
**Goal:** codeable project skeleton.

Deliverables:
- repo structure
- environment/dependency config
- `.env.example`
- tests
- logging skeleton
- config structure
- README skeleton
- optional CI basics

Acceptance:
- clean setup works
- no secrets
- tests runnable
- layer boundaries clear
- no business/model logic yet

**STOP:** do not continue to M1 automatically.

## M1 — Trade Data Ingestion
Goal: reproducible UN Comtrade canonical monthly ingestion.

Scope:
- TR→RU export
- RU→TR export
- TOTAL + HS2
- 2012-01 → latest complete month

Acceptance:
- direction mapping correct
- TOTAL/HS2 separated
- `YYYY-MM`
- deterministic duplicates
- missing not converted to zero
- source/retrieved_at stored

## M2 — Mirror Statistics
Goal: pair canonical and mirror data.

Acceptance:
- TR export ↔ RU import
- RU export ↔ TR import
- safe gap calculations
- canonical never overwritten
- missing/mismatch flags

## M3 — Data Audit & Sector Selection (Specification v1.1)
Goal: audit target series continuity and select top forecastable sectors.

Target Series Policy:
- TR_RU: Türkiye exports to Russia target series (`reporter=792`, `flow=X`).
- RU_TR: Türkiye imports from Russia target series (`reporter=792`, `flow=M`).
- Time span: **Development period `2012-01 → 2024-12`** (156 months). Final 12-month holdout (`2025-01 → 2025-12`) is strictly untouched.

Required filters:
- >=96 months
- >=90% coverage
- <=10% zero ratio
- >=21/24 recent observed months (2023-01 to 2024-12)
- non-trivial size

Ranking:
- volume 40
- continuity 30
- recent relevance 20
- forecastability 10

Acceptance:
- strictly no holdout data used (2025 holdout untouched)
- reproducible scores
- max 6 per direction
- failure reasons logged
- sector list versioned

**GATE:** sectors frozen.

## M4 — Macro Data Foundation
Initial:
- USD/TRY
- RUB/USD
- Brent/energy
- TR industrial production
- RU industrial production

Acceptance:
- source registry
- monthly normalized table
- release-lag metadata
- no future actuals
- fallback lag policy supported

## M5 — Feature Engineering
Features:
- lags 1/2/3/6/12
- rolling mean/std 3/6/12
- MoM/YoY
- month/quarter/year
- approved macro lags

Acceptance:
- h1/h3/h6 alignment tested
- backward-only rolling
- holdout isolation
- leakage tests

## M6 — Seasonal Naive Baseline
Acceptance:
- seasonal offset correct
- expanding walk-forward
- sMAPE/MAE/RMSE
- per series/horizon results
- holdout untouched

## M7 — SARIMA
Acceptance:
- order strategy documented
- same folds as baseline
- convergence/failure logged
- 80/95 intervals

## M8 — SARIMAX
Acceptance:
- publication-safe exogenous use
- no future macro actual
- deterministic missing-exog handling
- same folds/metrics
- intervals

## M9 — XGBoost Direct Forecasting
Deliver:
- h1/h3/h6 separate models
- controlled time-aware tuning
- SHAP
- calibrated intervals

Acceptance:
- no recursive forecast
- no holdout tuning
- fold-safe preprocessing
- SHAP not causal
- 80/95 intervals from development residuals/conformal method

## M10 — Champion Selection & Registry — ACCEPTED (2026-10-14)
Rule:
1. lowest validation sMAPE
2. MAE tie-break
3. <2% relative gain → simpler model

Acceptance:
- deterministic registry
- no manual cherry-picking
- run metadata complete
- holdout still closed

**GATE:** experiment design fully frozen.

Acceptance evidence: [`docs/M10_ACCEPTANCE_REPORT.md`](../docs/M10_ACCEPTANCE_REPORT.md) and [`docs/M10_CHAMPION_SELECTION_REPORT.md`](../docs/M10_CHAMPION_SELECTION_REPORT.md). Selection uses the frozen 281-origin calendar and does not open the holdout. The acceptance record retains the legacy M8 validation-manifest discrepancy and its explicit equivalence check.

## M11 — Final Holdout Evaluation — ACCEPTED (2026-10-14)
Goal: evaluate champion models on the untouched 12-month holdout (`2025-01 → 2025-12`).

Acceptance:
- strictly untouched during development and sector selection
- champion unchanged after results
- validation vs holdout separate
- negative/null results retained
- reproducible final report

Evaluation scope: **retrospective latest-vintage holdout evaluation under deterministic lag rules**. No historical publication-time archive or macro vintage snapshots were used; real-time decision-time availability is not established. See [`docs/M11_ACCEPTANCE_REPORT.md`](../docs/M11_ACCEPTANCE_REPORT.md) and [`docs/M11_FINAL_HOLDOUT_REPORT.md`](../docs/M11_FINAL_HOLDOUT_REPORT.md).

## M12 — Backend API — ACCEPTED (2026-10-16)
Minimum semantics:
- overview
- trade history
- forecasts
- model metrics
- sector selection
- mirror statistics
- economic indicators
- methodology metadata

Acceptance:
- typed/validated responses
- unavailable forecast status explicit
- data freshness metadata
- canonical/mirror distinction preserved

Accepted scope is the read-only, checksum-verified API over frozen local artifacts. It does not provide online inference, PostgreSQL persistence, live data refresh or a production deployment. Evidence: [`docs/M12_ACCEPTANCE_REPORT.md`](../docs/M12_ACCEPTANCE_REPORT.md).

## M13 — Frontend Dashboard — ACCEPTED (2026-10-16)
Modules:
- Overview
- Trade Explorer
- Forecasts
- Sector Analysis
- Model Lab
- Mirror Statistics
- Economic Indicators
- Methodology

Acceptance:
- TR/EN/RU
- direction/sector/horizon selectors
- intervals visible
- champion visible
- responsive core use
- methodology/uncertainty not hidden

Delivered scope: five dashboard modules, strict TypeScript, live-API/verified-snapshot modes, deterministic static snapshot and local production build. Local browser/build evidence is not hosted deployment or real-user acceptance. The interface discloses the retrospective latest-vintage deterministic-lag scope. Evidence: [`docs/M13_ACCEPTANCE_REPORT.md`](../docs/M13_ACCEPTANCE_REPORT.md).

## M14 — Research & Portfolio Packaging — IN PROGRESS
Deliverables:
- final README
- architecture diagram
- methodology
- data dictionary
- model results
- limitations
- English research paper
- Russian abstract
- Turkish project summary
- reproducibility instructions

The README/specification showcase portion is recorded in [`docs/M14_PORTFOLIO_ACCEPTANCE_REPORT.md`](../docs/M14_PORTFOLIO_ACCEPTANCE_REPORT.md). The working paper, Turkish summary, Russian abstract, data dictionary and source/licensing package remain open.

Paper:
1. Abstract
2. Introduction
3. Bilateral Trade Background
4. Literature Review
5. Data
6. Mirror Statistics and Data Quality
7. Methodology
8. Statistical Models
9. Machine Learning
10. Experimental Design
11. Results
12. Explainability
13. Discussion
14. Limitations
15. Conclusion

## M15 — Final QA / Release Gate
Must pass:
- all MUST requirements
- tests
- current data-quality report
- versioned selected sectors
- reproducible holdout report
- API/frontend alignment
- TR/EN/RU core strings
- no secrets
- current README/methodology/research report

## Mandatory prompt template
Every coding-agent prompt must contain:
1. Role
2. Current milestone
3. Relevant frozen spec
4. Exact task
5. Files/modules allowed to change
6. Non-goals
7. Acceptance criteria
8. Tests required
9. Output/report format
10. Stop condition

The agent must not implement the next milestone without a separate prompt.
