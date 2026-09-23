# MILESTONES.md
# Implementation Roadmap & Acceptance Gates

**Version:** 1.0  
**Status:** READY FOR PROMPT GENERATION

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

## M3 — Data Audit & Sector Selection
Required filters:
- >=96 months
- >=90% coverage
- <=10% zero ratio
- >=21/24 recent observed months
- non-trivial size

Ranking:
- volume 40
- continuity 30
- recent relevance 20
- forecastability 10

Acceptance:
- no holdout use
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

## M10 — Champion Selection & Registry
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

## M11 — Final Holdout Evaluation
Goal: last 12 complete months.

Acceptance:
- previously untouched
- champion unchanged after results
- validation vs holdout separate
- negative/null results retained
- reproducible final report

## M12 — Backend API
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

## M13 — Frontend Dashboard
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

## M14 — Research & Portfolio Packaging
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
