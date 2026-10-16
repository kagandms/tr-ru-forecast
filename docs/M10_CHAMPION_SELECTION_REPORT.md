# M10 Champion Selection Report

**Generated:** 2026-10-14T14:27:47.380449+00:00
**Specification:** v1.1
**Status:** CV selection frozen; M11 acceptance gate remains closed.

## 1. Executive Summary & Acceptance Verdict

Computed 168 candidate entries and 42 deterministic champion/fallback decisions from the existing development CV artifacts. Selection implementation is complete; full experiment release is conditional on the open M9 A2 review and the legacy M8 provenance discrepancy. No holdout evaluation, model refit, API implementation or causal interpretation is included.

## 2. Selection Protocol & Governance Rules

The frozen calendar contains 96/94/91 monthly rolling origins for h1/h3/h6 (281 shared folds), not five folds. The pasted five-fold/RMSE-first proposal is superseded by the explicitly authoritative protocol. Each task requires complete finite, nonnegative predictions and successful converged fits. Failed candidates remain in the leaderboard. Seasonal Naive must be eligible, otherwise selection fails closed. No silent fallback or clipping repairs CV evidence.

Rank by mean sMAPE, MAE, diagnostic RMSE, population standard deviation, then complexity tier. Starting from the raw winner, select the simplest eligible model whose relative sMAPE loss against that winner is strictly below 2%, using the simpler model as the denominator (protocol §8). Exactly 2% retains the complex winner. This deterministic tolerance is not a test of statistical indistinguishability.

A worst-origin ratio above 1.25 against an eligible candidate in the same parsimony window flags instability; it does not override the frozen selection rule. The cap is a configurable descriptive threshold, not a fitted parameter. Fallback is the best ranked lower-complexity candidate for SARIMAX/XGBoost champions; otherwise the best other eligible candidate. If only the baseline is eligible, fallback explicitly repeats the baseline.

sMAPE/MAE are means over origins; std uses ddof=0. `mean_rmse` preserves the M6–M9 pooled RMSE (square root of mean squared error), since the mean of single-observation RMSEs would equal MAE. Skill is (naive−candidate)/naive, with percentage multiplied by 100. A zero baseline uses skill 0 for an exact tie and −1 for a positive-error candidate, an explicit finite sentinel. Missing-candidate metrics remain null. Finite negative forecasts remain in descriptive scores while invalidating their candidate.

Model references identify frozen refit specifications, not serialized serving models. Seasonal Naive intervals are explicitly unavailable until a separate approved method is supplied; native classical and M9 prequential interval policies are retained.

## 3. Global Model Family Win Share

| Family | h1 | h3 | h6 | Overall | Share | Mean skill % | Median skill % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seasonal_naive | 0 | 0 | 0 | 0 | 0.0% | — | — |
| sarima | 11 | 13 | 9 | 33 | 78.6% | 30.44 | 29.76 |
| sarimax | 0 | 0 | 0 | 0 | 0.0% | — | — |
| xgboost | 3 | 1 | 5 | 9 | 21.4% | 30.76 | 24.48 |

## 4. Complete 42-Task Champion Matrix

| Sector | h | Champion | Fallback | CV sMAPE % | Std | Worst | Naive % | Skill % | Reason |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| RU_TR:10 | 1 | sarima | xgboost | 33.49 | 28.00 | 136.66 | 53.11 | 36.94 | CLEAR_SMAPE_WINNER |
| RU_TR:10 | 3 | sarima | xgboost | 41.89 | 29.46 | 136.74 | 53.78 | 22.10 | CLEAR_SMAPE_WINNER |
| RU_TR:10 | 6 | xgboost | sarima | 42.57 | 29.49 | 115.69 | 53.99 | 21.15 | CLEAR_SMAPE_WINNER |
| RU_TR:23 | 1 | sarima | xgboost | 22.59 | 18.29 | 83.73 | 31.36 | 27.98 | CLEAR_SMAPE_WINNER |
| RU_TR:23 | 3 | sarima | xgboost | 25.23 | 20.75 | 91.37 | 31.11 | 18.89 | CLEAR_SMAPE_WINNER |
| RU_TR:23 | 6 | sarima | xgboost | 25.55 | 21.03 | 85.72 | 31.59 | 19.12 | CLEAR_SMAPE_WINNER |
| RU_TR:27 | 1 | sarima | xgboost | 18.97 | 16.58 | 74.34 | 43.93 | 56.82 | CLEAR_SMAPE_WINNER |
| RU_TR:27 | 3 | sarima | xgboost | 27.46 | 23.63 | 128.37 | 44.57 | 38.37 | CLEAR_SMAPE_WINNER |
| RU_TR:27 | 6 | sarima | xgboost | 35.99 | 30.17 | 136.31 | 45.04 | 20.10 | CLEAR_SMAPE_WINNER |
| RU_TR:72 | 1 | sarima | xgboost | 23.31 | 18.50 | 109.14 | 38.16 | 38.91 | CLEAR_SMAPE_WINNER |
| RU_TR:72 | 3 | sarima | xgboost | 26.87 | 19.98 | 97.97 | 38.26 | 29.76 | CLEAR_SMAPE_WINNER |
| RU_TR:72 | 6 | sarima | xgboost | 32.05 | 22.54 | 94.26 | 38.37 | 16.48 | CLEAR_SMAPE_WINNER |
| RU_TR:74 | 1 | sarima | seasonal_naive | 46.69 | 42.26 | 199.98 | 65.40 | 28.62 | CLEAR_SMAPE_WINNER |
| RU_TR:74 | 3 | sarima | xgboost | 50.10 | 43.42 | 199.99 | 64.76 | 22.63 | CLEAR_SMAPE_WINNER |
| RU_TR:74 | 6 | xgboost | sarima | 53.22 | 42.39 | 199.99 | 65.57 | 18.84 | CLEAR_SMAPE_WINNER |
| RU_TR:76 | 1 | sarima | xgboost | 27.52 | 24.04 | 120.73 | 40.61 | 32.24 | CLEAR_SMAPE_WINNER |
| RU_TR:76 | 3 | xgboost | sarima | 29.88 | 25.94 | 135.47 | 40.09 | 25.47 | CLEAR_SMAPE_WINNER |
| RU_TR:76 | 6 | sarima | xgboost | 38.14 | 28.11 | 122.65 | 40.15 | 5.00 | CLEAR_SMAPE_WINNER |
| RU_TR:TOTAL | 1 | sarima | xgboost | 14.07 | 12.62 | 63.42 | 31.33 | 55.08 | CLEAR_SMAPE_WINNER |
| RU_TR:TOTAL | 3 | sarima | xgboost | 19.17 | 14.38 | 63.64 | 31.78 | 39.68 | CLEAR_SMAPE_WINNER |
| RU_TR:TOTAL | 6 | sarima | xgboost | 25.77 | 19.06 | 70.76 | 32.13 | 19.79 | CLEAR_SMAPE_WINNER |
| TR_RU:08 | 1 | sarima | xgboost | 22.20 | 21.37 | 97.20 | 34.17 | 35.03 | CLEAR_SMAPE_WINNER |
| TR_RU:08 | 3 | sarima | seasonal_naive | 25.33 | 20.19 | 107.08 | 33.45 | 24.26 | CLEAR_SMAPE_WINNER |
| TR_RU:08 | 6 | sarima | seasonal_naive | 27.09 | 21.37 | 110.57 | 33.24 | 18.50 | CLEAR_SMAPE_WINNER |
| TR_RU:39 | 1 | sarima | xgboost | 15.94 | 13.79 | 64.15 | 34.29 | 53.51 | CLEAR_SMAPE_WINNER |
| TR_RU:39 | 3 | sarima | sarimax | 21.58 | 17.98 | 89.91 | 34.73 | 37.86 | CLEAR_SMAPE_WINNER |
| TR_RU:39 | 6 | sarima | xgboost | 27.28 | 24.54 | 117.49 | 35.03 | 22.13 | CLEAR_SMAPE_WINNER |
| TR_RU:73 | 1 | xgboost | sarima | 17.49 | 14.83 | 64.72 | 29.98 | 41.64 | CLEAR_SMAPE_WINNER |
| TR_RU:73 | 3 | sarima | sarimax | 22.23 | 17.21 | 81.26 | 29.86 | 25.54 | CLEAR_SMAPE_WINNER |
| TR_RU:73 | 6 | xgboost | sarima | 23.23 | 18.01 | 75.92 | 29.69 | 21.74 | CLEAR_SMAPE_WINNER |
| TR_RU:84 | 1 | xgboost | sarima | 17.00 | 14.33 | 67.36 | 36.49 | 53.42 | CLEAR_SMAPE_WINNER |
| TR_RU:84 | 3 | sarima | xgboost | 20.43 | 18.02 | 80.00 | 36.61 | 44.21 | CLEAR_SMAPE_WINNER |
| TR_RU:84 | 6 | sarima | sarimax | 25.48 | 24.46 | 116.79 | 36.67 | 30.52 | CLEAR_SMAPE_WINNER |
| TR_RU:85 | 1 | sarima | xgboost | 22.19 | 17.09 | 83.38 | 41.87 | 47.02 | PARSIMONY_PREFERENCE |
| TR_RU:85 | 3 | sarima | xgboost | 26.94 | 20.19 | 78.01 | 42.63 | 36.80 | PARSIMONY_PREFERENCE |
| TR_RU:85 | 6 | xgboost | sarima | 31.94 | 24.97 | 113.17 | 42.29 | 24.48 | CLEAR_SMAPE_WINNER |
| TR_RU:87 | 1 | sarima | xgboost | 21.05 | 20.05 | 116.53 | 33.37 | 36.93 | CLEAR_SMAPE_WINNER |
| TR_RU:87 | 3 | sarima | xgboost | 25.46 | 24.30 | 124.49 | 33.68 | 24.40 | CLEAR_SMAPE_WINNER |
| TR_RU:87 | 6 | sarima | xgboost | 30.01 | 26.18 | 128.50 | 33.10 | 9.35 | CLEAR_SMAPE_WINNER |
| TR_RU:TOTAL | 1 | xgboost | sarima | 14.83 | 12.87 | 65.13 | 31.17 | 52.42 | CLEAR_SMAPE_WINNER |
| TR_RU:TOTAL | 3 | sarima | xgboost | 21.79 | 19.20 | 80.06 | 31.10 | 29.96 | PARSIMONY_PREFERENCE |
| TR_RU:TOTAL | 6 | xgboost | sarima | 25.44 | 21.21 | 100.79 | 30.89 | 17.66 | CLEAR_SMAPE_WINNER |

## 5. Eligibility Audit

| Family | Eligible / 42 | Negative forecasts | Ineligible tasks |
| --- | ---: | ---: | --- |
| seasonal_naive | 42 | 0 | none |
| sarima | 42 | 0 | none |
| sarimax | 27 | 27 | TR_RU:08/h1, TR_RU:08/h3, TR_RU:84/h1, TR_RU:87/h6, RU_TR:10/h1, RU_TR:10/h3, RU_TR:10/h6, RU_TR:76/h1, RU_TR:76/h3, RU_TR:76/h6, RU_TR:23/h1, RU_TR:23/h3, RU_TR:74/h1, RU_TR:74/h3, RU_TR:74/h6 |
| xgboost | 41 | 1 | RU_TR:74/h1 |

## 6. Economic & Domain Interpretation

Exogenous-capable families win 9 tasks: RU_TR:10/h6 (xgboost), RU_TR:74/h6 (xgboost), RU_TR:76/h3 (xgboost), TR_RU:73/h1 (xgboost), TR_RU:73/h6 (xgboost), TR_RU:84/h1 (xgboost), TR_RU:85/h6 (xgboost), TR_RU:TOTAL/h1 (xgboost), TR_RU:TOTAL/h6 (xgboost). These are observed predictive comparisons; XGBoost also uses trade lags/calendar features, so its wins do not isolate macro contributions. SARIMAX uses unchanged univariate orders with a constant origin-known macro path; longer horizons cannot benefit from future macro realizations. SARIMA wins are consistent with useful within-series dynamics and seasonality, but do not establish a causal economic mechanism.

TOTAL aggregates and HS2 components have different volatility and seasonal structure. Fruit exports (08), energy imports (27), and machinery/metal flows are economically different series; the table supports sector/horizon heterogeneity without attributing shocks to specific causes. Worst-origin and dispersion diagnostics retain large development errors, including origins around 2022. No break-specific effect estimate or statistical significance claim is made.

The comparison uses retrospective latest-vintage development artifacts. Approximate publication lags do not establish historical macro vintage fidelity or actual trade decision-time availability. Winner shares and CV skill are selection-period results; independent 2025 performance remains unknown. Negative SARIMAX/XGBoost point forecasts are retained as eligibility evidence.

## 7. Reproducibility & Provenance Evidence

Git commit: `c7c8edc9e750d49d6de8aad676f1553f51c22387`; dirty working tree: `True`. Source fingerprint: `c336392369e17e92f22f0a2abcde988741806d89c97aa8811d710f94bf3f0479`.

Commands: `make m10-select`; reproduce independently with `make m10-select M10_OUTPUT_DIR=/tmp/m10-reproduction`. Existing output files are never overwritten. Compare champion decisions separately from run timestamps and Git dirty-state metadata.

Artifacts: `configs/champion_registry_v1_1.yaml`, `data/processed/m10_cv_leaderboard.parquet` (168 rows), `m10_cv_fold_details.parquet` (15,736 observed CV rows), `m10_champion_matrix.parquet` (42 rows), `m10_cv_leaderboard.csv`, and `m10_selection_run.json`. Registry holds input/output checksums; run JSON also hashes the registry and report.

The M9 accepted rerun config, prediction and fit-log hashes are verified against its run metadata. Every candidate is checked against the frozen calendar and shared baseline actuals. M10 reads only allowlisted CV artifacts, configuration, run metadata and source files; raw trade/target/macro/feature datasets are not opened. This boundary does not establish that earlier milestones never read 2025 data.

Legacy discrepancies: m8_sarimax: recorded validation YAML hash differs; exact row-level CV calendar independently verified.

Executed check results and test counts are recorded in `docs/M10_ACCEPTANCE_REPORT.md`; the generator does not assert unexecuted checks.
