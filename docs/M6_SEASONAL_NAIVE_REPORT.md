# M6 Seasonal Naive Baseline Report

**Document Version:** 1.1.0  
**Execution Date:** 2026-09-22T17:20:37.987960+00:00  
**Authoritative Specification:** Specification Pack v1.1 (`specification/`)  
**Author:** Senior Forecasting Research Engineer & Backtesting Specialist  
**Status:** **READY FOR EXTERNAL REVIEW**  

---

## Specification / Scope

- **Milestone:** M6 — Seasonal Naive Baseline.
- **Specification Pack:** v1.1 (authoritative source of truth).
- **Scope Boundary:** Establishes the frozen benchmark forecasting result across the 156-month development period (`2012-01` to `2024-12`) for all 14 target series across horizons $h \in \{1, 3, 6\}$.
- **Strict Non-Goals Enforced:** No SARIMA, SARIMAX, or XGBoost model fitting; no hyperparameter tuning; no champion model selection; strictly no opening of the 2025 trade-target holdout.

---

## Frozen Inputs

| Artifact Role | File Path | Status / Notes |
| :--- | :--- | :--- |
| **Development Features** | `data/processed/model_features_development.parquet` | Milestone M5 Final Pass (2,184 rows, 31 columns) |
| **Selected Sectors Manifest** | `configs/selected_sectors_v1_1.yaml` | Milestone M3 Final Pass (12 HS2 sectors + 2 TOTALs) |
| **Feature Specification** | `configs/feature_spec_v1_1.yaml` | Milestone M5 Final Pass |
| **Macro Data Foundation** | `data/processed/macro_monthly.parquet` | Milestone M4 Final Pass (not consumed by Seasonal Naive) |

> [!NOTE]
> **Input Data Integrity:** Seasonal Naive consumes strictly the historical `target` values from `model_features_development.parquet` for the 14 modeling series. Zero macroeconomic or engineered features are utilized.

---

## Modeling Universe

The modeling universe consists of exactly **14 monthly target series** across the two bilateral directions:
- **TR → RU Direction (7 series):** `TOTAL`, `08`, `84`, `39`, `87`, `73`, `85` (Türkiye exports to Russia, reporter 792, partner 643, Flow X).
- **RU → TR Direction (7 series):** `TOTAL`, `27`, `72`, `10`, `76`, `23`, `74` (Türkiye imports from Russia, reporter 792, partner 643, Flow M).

Both forecasting targets are official Türkiye-reported series. Russia-reported counterflows remain mirror/reference statistics only.

---

## Validation Protocol

- **Strategy:** Expanding walk-forward / rolling-origin validation.
- **Cadence:** Monthly forecast origins.
- **Development Horizon:** `2012-01` to `2024-12` (156 calendar months).
- **Final Untouched Holdout:** `2025-01` to `2025-12` (12 calendar months; strictly quarantined).
- **Minimum Initial History:** 60 months (`2012-01` through `2016-12` inclusive).
- **First Validation Origin:** **`2016-12`** (empirically derived as the earliest month with 60 observed monthly target values through the origin).
- **Training Window Semantics:** For origin $t$, `train_start = '2012-01'` and `train_end = t` (expanding monotonically).
- **Horizon-Specific Terminal Origins:**
  - **$h=1$:** First origin `2016-12`, Last origin **`2024-11`** (forecast period: `2024-12`), **96 origins**.
  - **$h=3$:** First origin `2016-12`, Last origin **`2024-09`** (forecast period: `2024-12`), **94 origins**.
  - **$h=6$:** First origin `2016-12`, Last origin **`2024-06`** (forecast period: `2024-12`), **91 origins**.

---

## Seasonal Naive Definition

$$\text{Forecast}(t+h) = \text{Actual}(t+h-12)$$

For each forecast origin $t$ and horizon $h \in \{1, 3, 6\}$:
- `forecast_period` = $t + h$ calendar months.
- `seasonal_source_period` = $t + h - 12$ calendar months.
- **Origin Safety Invariant:** $(t+h-12) \le t$ strictly holds for all $h \le 6$. Every prediction is formed using only historical information available on or before the forecast origin.
- **Exact Calendar Offsets:** Offsets are based strictly on contiguous monthly calendar dates, not naive row slicing.
- **Series Isolation:** Forecasts query historical observations strictly within the same bilateral direction and HS2 commodity code.

---

## Validation Fold Registry

- Manifest YAML: `configs/validation_folds_v1_1.yaml`
- Row-Level Table: `data/processed/validation_origins.parquet`
- Total Shared Validation Folds: **281 folds** ($96 + 94 + 91$).
- Reusability: This fold schedule is frozen and will be reused without modification by M7 SARIMA, M8 SARIMAX, and M9 XGBoost.

---

## Prediction Audit

- **Total Prediction Rows:** $3,934$ ($14\text{ series} \times 281\text{ folds}$).
- **Valid Prediction Rows:** $3,934$ (100% valid; zero missing predictions).
- **Missing / Invalid Predictions:** $0$.
- **Forecast Period Range:** `2017-01` to `2024-12` (zero 2025 holdout observations targeted).
- **Seasonal Source Period Range:** `2016-01` to `2023-12` (all strictly $\le$ corresponding forecast origin).

---

## Metric Definitions

For each prediction $i$ with actual $y_i$ and prediction $\hat{y}_i$:
- **Absolute Error:** $|y_i - \hat{y}_i|$
- **Squared Error:** $(y_i - \hat{y}_i)^2$
- **sMAPE Component:**
  $$\text{sMAPE}_i = \begin{cases} 0.0 & \text{if } y_i = 0 \text{ and } \hat{y}_i = 0 \\ \frac{2 |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|} \times 100 & \text{otherwise} \end{cases}$$
- **Aggregated Group Metrics:**
  - **sMAPE (Primary):** $\frac{1}{N} \sum_{i=1}^N \text{sMAPE}_i$
  - **MAE (Secondary):** $\frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$
  - **RMSE (Diagnostic):** $\sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$

---

## Results by Series and Horizon

The canonical 42 metric groups ($14\text{ series} \times 3\text{ horizons}$) are recorded below:

| Direction | HS2 | Horizon | Origins | sMAPE (%) | MAE (USD) | RMSE (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `RU_TR` | `10` | $h=1$ | 96 | **53.11%** | $85,327,521 | $110,645,197 |
| `RU_TR` | `10` | $h=3$ | 94 | **53.78%** | $86,919,936 | $111,798,061 |
| `RU_TR` | `10` | $h=6$ | 91 | **53.99%** | $89,046,596 | $113,550,390 |
| `RU_TR` | `23` | $h=1$ | 96 | **31.36%** | $12,494,194 | $17,163,047 |
| `RU_TR` | `23` | $h=3$ | 94 | **31.11%** | $12,486,238 | $17,146,701 |
| `RU_TR` | `23` | $h=6$ | 91 | **31.59%** | $12,769,189 | $17,398,651 |
| `RU_TR` | `27` | $h=1$ | 96 | **43.93%** | $710,838,536 | $1,059,749,004 |
| `RU_TR` | `27` | $h=3$ | 94 | **44.57%** | $723,659,745 | $1,070,782,524 |
| `RU_TR` | `27` | $h=6$ | 91 | **45.04%** | $739,917,102 | $1,087,463,530 |
| `RU_TR` | `72` | $h=1$ | 96 | **38.16%** | $116,403,608 | $144,433,992 |
| `RU_TR` | `72` | $h=3$ | 94 | **38.26%** | $117,349,572 | $145,337,197 |
| `RU_TR` | `72` | $h=6$ | 91 | **38.37%** | $118,727,871 | $146,776,064 |
| `RU_TR` | `74` | $h=1$ | 96 | **65.40%** | $47,552,902 | $72,382,297 |
| `RU_TR` | `74` | $h=3$ | 94 | **64.76%** | $47,845,312 | $72,974,712 |
| `RU_TR` | `74` | $h=6$ | 91 | **65.57%** | $48,864,161 | $74,090,765 |
| `RU_TR` | `76` | $h=1$ | 96 | **40.61%** | $33,071,414 | $41,374,291 |
| `RU_TR` | `76` | $h=3$ | 94 | **40.09%** | $33,032,700 | $41,486,422 |
| `RU_TR` | `76` | $h=6$ | 91 | **40.15%** | $33,420,646 | $41,875,036 |
| `RU_TR` | `TOTAL` | $h=1$ | 96 | **31.33%** | $845,670,102 | $1,223,943,925 |
| `RU_TR` | `TOTAL` | $h=3$ | 94 | **31.78%** | $860,683,333 | $1,236,716,455 |
| `RU_TR` | `TOTAL` | $h=6$ | 91 | **32.13%** | $879,366,800 | $1,255,789,912 |
| `TR_RU` | `08` | $h=1$ | 96 | **34.17%** | $19,838,200 | $24,541,735 |
| `TR_RU` | `08` | $h=3$ | 94 | **33.45%** | $19,864,601 | $24,599,573 |
| `TR_RU` | `08` | $h=6$ | 91 | **33.24%** | $20,302,195 | $24,920,645 |
| `TR_RU` | `39` | $h=1$ | 96 | **34.29%** | $10,997,853 | $18,757,962 |
| `TR_RU` | `39` | $h=3$ | 94 | **34.73%** | $11,210,940 | $18,955,903 |
| `TR_RU` | `39` | $h=6$ | 91 | **35.03%** | $11,507,027 | $19,261,065 |
| `TR_RU` | `73` | $h=1$ | 96 | **29.98%** | $4,174,094 | $5,574,094 |
| `TR_RU` | `73` | $h=3$ | 94 | **29.86%** | $4,239,162 | $5,630,291 |
| `TR_RU` | `73` | $h=6$ | 91 | **29.69%** | $4,324,512 | $5,711,376 |
| `TR_RU` | `84` | $h=1$ | 96 | **36.49%** | $33,932,325 | $57,039,755 |
| `TR_RU` | `84` | $h=3$ | 94 | **36.61%** | $34,550,901 | $57,638,951 |
| `TR_RU` | `84` | $h=6$ | 91 | **36.67%** | $35,462,366 | $58,567,888 |
| `TR_RU` | `85` | $h=1$ | 96 | **41.87%** | $13,278,882 | $21,145,971 |
| `TR_RU` | `85` | $h=3$ | 94 | **42.63%** | $13,549,913 | $21,369,494 |
| `TR_RU` | `85` | $h=6$ | 91 | **42.29%** | $13,831,486 | $21,697,656 |
| `TR_RU` | `87` | $h=1$ | 96 | **33.37%** | $11,215,514 | $14,459,763 |
| `TR_RU` | `87` | $h=3$ | 94 | **33.68%** | $11,408,365 | $14,606,994 |
| `TR_RU` | `87` | $h=6$ | 91 | **33.10%** | $11,509,798 | $14,764,404 |
| `TR_RU` | `TOTAL` | $h=1$ | 96 | **31.17%** | $161,699,432 | $236,370,722 |
| `TR_RU` | `TOTAL` | $h=3$ | 94 | **31.10%** | $164,137,868 | $238,769,281 |
| `TR_RU` | `TOTAL` | $h=6$ | 91 | **30.89%** | $167,442,241 | $242,343,353 |

---

## Direction-Level Descriptive Summary

| Direction | Horizon | Mean sMAPE | Median sMAPE | Mean MAE | Mean RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `RU_TR` | $h=1$ | 43.42% | 40.61% | $264,479,754 | $381,384,536 |
| `RU_TR` | $h=3$ | 43.48% | 40.09% | $268,853,834 | $385,177,439 |
| `RU_TR` | $h=6$ | 43.83% | 40.15% | $274,587,481 | $390,992,050 |
| `TR_RU` | $h=1$ | 34.48% | 34.17% | $36,448,043 | $53,984,286 |
| `TR_RU` | $h=3$ | 34.58% | 33.68% | $36,994,536 | $54,510,069 |
| `TR_RU` | $h=6$ | 34.42% | 33.24% | $37,768,518 | $55,323,770 |

---

## Horizon-Level Descriptive Summary

| Horizon | Total Groups | Mean sMAPE | Median sMAPE | Mean MAE | Mean RMSE |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$h=1$** | 14 | **38.95%** | 35.39% | $150,463,898 | $217,684,411 |
| **$h=3$** | 14 | **39.03%** | 35.67% | $152,924,185 | $219,843,754 |
| **$h=6$** | 14 | **39.12%** | 35.85% | $156,177,999 | $223,157,910 |

---

## Missing / Zero Handling

- **Reported Zeros:** Preserved as genuine numerical zero observations. In the empirical development dataset, zero trade is observed in specific minor monthly sectors without triggering division-by-zero errors.
- **Missing Policy:** Missing values remain strictly `NaN`. Because the 14 selected development series have 100% complete records across 2012–2024, zero missing seasonal sources or targets were encountered.
- **No Imputation:** No interpolation, forward-fill, backward-fill, or mirror replacement was performed.

---

## Holdout Isolation

> [!IMPORTANT]
> **Holdout Isolation Guarantee:**
> - All validation origins satisfy $t \le \text{2024-11}$.
> - All forecast periods satisfy $t+h \le \text{2024-12}$.
> - Not a single observation from the 2025 holdout (`2025-01` to `2025-12`) is accessed.
> - **Holdout Mutation Test:** Radically mutating all 2025 trade targets confirmed that predictions, validation folds, and metrics remain 100% numerically identical.

---

## Reproducibility Metadata

- **Run ID:** `run_m6_seasonal_naive_v1_1`
- **Execution Timestamp:** `2026-09-22T17:20:37.987960+00:00`
- **Code Version:** `1.1.0`
- **Hyperparameters:** `seasonal_period_months = 12` (deterministic; random seed not applicable)
- **Artifacts Persisted:**
  - Validation Manifest: `/Users/kagansmtdms/Downloads/tr-ru_forecast/configs/validation_folds_v1_1.yaml`
  - Validation Origins: `/Users/kagansmtdms/Downloads/tr-ru_forecast/data/processed/validation_origins.parquet`
  - Predictions: `/Users/kagansmtdms/Downloads/tr-ru_forecast/data/processed/m6_seasonal_naive_predictions.parquet`
  - Metrics: `/Users/kagansmtdms/Downloads/tr-ru_forecast/data/processed/m6_seasonal_naive_metrics.parquet`
  - Run Metadata: `/Users/kagansmtdms/Downloads/tr-ru_forecast/data/processed/m6_seasonal_naive_run.json`

---

## Tests

The automated test suite `tests/unit/test_seasonal_naive.py` executes 50 comprehensive unit and acceptance tests covering:
1. Complete 14-series universe and M3/M5 manifest integrity.
2. Deterministic walk-forward origin schedule and 60-month history threshold.
3. Horizon terminal bounds ($h=1 \le \text{2024-11}$, $h=3 \le \text{2024-09}$, $h=6 \le \text{2024-06}$).
4. Seasonal Naive calendar offset formula and origin safety invariant ($t+h-12 \le t$).
5. Series isolation (no cross-series leakage).
6. sMAPE both-zero safeguard ($0.0$) and finite evaluation.
7. Complete 42-group metric coverage without duplicates.
8. Holdout mutation tests on predictions, metrics, and fold registry.
9. Non-goal enforcement: no scalers, no tuning, no model selection, no M7+ artifacts.

---

## Validation

- Full test suite: **232+ / 232+ tests passing**.
- Linter: **Clean (`ruff check .` passes with 0 errors)**.
- Sector manifest hash: **Identical to M3 Final Pass**.
- Feature manifest hash: **Identical to M5 Final Pass**.

---

## Deviations From Specification

**None.** All implementations strictly comply with Specification Pack v1.1 and M6 milestone instructions.

---

## Issues / Limitations

- **Pure Persistence Limitation:** Seasonal Naive strictly assumes that historical trade patterns from month $t+h-12$ persist unchanged into month $t+h$. It cannot anticipate structural shifts, sanctions, trade rerouting, or macroeconomic shocks.
- **Exogenous Independence:** The baseline does not incorporate exchange rate or commodity price signals. It serves as an uninformative benchmark to evaluate whether subsequent econometric and machine learning models (SARIMAX, XGBoost) add genuine predictive value.
- **Development Estimates Only:** Validation metrics are historical estimates over 2016–2024. The final 2025 holdout remains untouched.

---

## Acceptance Criteria

- [x] Exactly 14 frozen target series evaluated
- [x] Only development data used
- [x] 2025 holdout remains untouched
- [x] Seasonal Naive formula exactly equals Actual(t+h-12)
- [x] Exact calendar offsets used
- [x] Same-series isolation enforced
- [x] Monthly expanding walk-forward origins generated
- [x] Minimum initial history = 60 months enforced
- [x] Origin schedule deterministic
- [x] Horizon terminal origins correct (h1: 2024-11, h3: 2024-09, h6: 2024-06)
- [x] Shared validation fold registry versioned (`configs/validation_folds_v1_1.yaml`)
- [x] Fold registry reusable by M7/M8/M9 (`data/processed/validation_origins.parquet`)
- [x] sMAPE correctly implemented (with both-zero safeguard)
- [x] MAE correctly implemented
- [x] RMSE correctly implemented
- [x] Both-zero sMAPE rule correct
- [x] Metrics computed per direction/sector/horizon
- [x] Exactly 42 expected metric groups
- [x] Prediction rows fully auditable (3,934 rows)
- [x] Holdout mutation test passes
- [x] Rerun determinism passes
- [x] M3/M5 frozen manifests unchanged
- [x] Machine-readable run metadata produced (`data/processed/m6_seasonal_naive_run.json`)
- [x] Full pytest suite passes
- [x] Lint passes (`ruff` clean)
- [x] M7+ not implemented

---

## Ready for M7?
**YES** — Milestone M6 Seasonal Naive Baseline is complete, verified, and reproducible. Ready for external review and authorization of M7 (SARIMA).
