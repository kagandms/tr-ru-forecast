# M8 SARIMAX Forecasting Report

**Generated**: 2026-09-23 16:14:50 UTC  
**Run ID**: `run_m8_sarimax_v1_1`  
**Specification**: `Pack v1.1`  
**Model Type**: `SARIMAX (Multivariate Statistical Forecasting with Origin-Known Exogenous Persistence)`  
**Status**: `FINAL CANDIDATE (AWAITING EXTERNAL REVIEW)`  

---

## 1. Specification / Scope

Milestone M8 evaluates whether adding publication-safe macroeconomic regressors to the existing statistical forecasting framework provides incremental forecasting accuracy relative to univariate SARIMA across the 14 frozen target series. The model is evaluated using the exact 281 expanding walk-forward validation folds established in M6/M7.

Key scope boundaries enforced:
- **Strict Order Reuse**: Structural ARIMA orders $(p,d,q)$ and seasonal orders $(P,D,Q,12)$ are reused verbatim from `configs/sarima_orders_v1_1.yaml`. No new AIC/BIC grid search or order re-tuning was performed.
- **Exact 5 Exogenous Regressors**: `usd_try_lagged`, `rub_usd_lagged`, `brent_lagged`, `tr_ip_lagged`, and `ru_ip_lagged`. No additional macro transformations, logs, or indicators.
- **Frozen Future-Exogenous Strategy**: `origin_known_persistence`. The 5-dimensional exogenous vector legally knowable at forecast origin snapshot $t$ is held strictly constant across all future steps ($t+1, \dots, t+h$). Future macro actuals are strictly prohibited.
- **Publication-Safe Availability**: All macro observations satisfy `availability_date_or_approximation <= canonical_origin_snapshot_date`.
- **Deterministic Missing-Exog Policy**: Fail closed loudly without interpolation, mean-filling, zero-substitution, regressor dropping, or silent fallback.
- **Native Multi-Step Uncertainty**: Native 80% ($\alpha=0.20$) and 95% ($\alpha=0.05$) prediction intervals condition on the deterministic persistence path.
- **Strict Holdout Quarantine**: Calendar year 2025 (`2025-01` to `2025-12`) remains completely sealed and untouched.
- **No Champion Selection**: Champion designation is explicitly deferred to M10.

---

## 2. Frozen Inputs

| Manifest / Artifact | Path | Checksum (SHA-256) | Status |
| :--- | :--- | :--- | :--- |
| Selected Sectors Manifest | `configs/selected_sectors_v1_1.yaml` | `c6e8f524657d8e54...` | FROZEN (M3) |
| Feature Spec Manifest | `configs/feature_spec_v1_1.yaml` | `873f98023b0b7e1e...` | FROZEN (M5) |
| Macro Sources Manifest | `configs/macro_sources_v1_1.yaml` | `bca72a83fc594529...` | FROZEN (M4) |
| Validation Folds Manifest | `configs/validation_folds_v1_1.yaml` | `0dc666c284991d49...` | FROZEN (M6) |
| SARIMA Orders Manifest | `configs/sarima_orders_v1_1.yaml` | `b9b2671e542e751a...` | FROZEN (M7) |
| Development Target Features (2,184 rows × 31 cols, 14 series × 156 months) | `data/processed/model_features_development.parquet` | `0328922bf4e2e246...` | FROZEN (M5) |

---

## 3. Validation Fold Integrity

SARIMAX evaluated the exact 281 expanding walk-forward folds from M6/M7:
- **Total Folds**: `281`
- **Horizon $h=1$**: `96` origins (`2016-12` to `2024-11`, forecast periods `2017-01` to `2024-12`)
- **Horizon $h=3$**: `94` origins (`2016-12` to `2024-09`, forecast periods `2017-03` to `2024-12`)
- **Horizon $h=6$**: `91` origins (`2016-12` to `2024-06`, forecast periods `2017-06` to `2024-12`)
- **Total Expected Predictions**: $14 \text{ series} \times (96 + 94 + 91) = 3,934$ rows (Actual: `3934`).

---

## 4. Reused Structural Orders & Exogenous Regressors

### Frozen Orders (Reused Verbatim from M7)

| Direction | HS2 | Order $(p,d,q)$ | Seasonal $(P,D,Q)_{12}$ | Trend | Source Policy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TR_RU` | `08` | `[1, 1, 1]` | `[1, 1, 0, 12]` | `n` | M7 Freeze (No Search) |
| `TR_RU` | `84` | `[0, 1, 0]` | `[1, 1, 1, 12]` | `n` | M7 Freeze (No Search) |
| `TR_RU` | `39` | `[1, 1, 0]` | `[1, 1, 1, 12]` | `n` | M7 Freeze (No Search) |
| `TR_RU` | `87` | `[1, 1, 0]` | `[0, 1, 1, 12]` | `n` | M7 Freeze (No Search) |
| `TR_RU` | `73` | `[1, 1, 0]` | `[0, 1, 1, 12]` | `n` | M7 Freeze (No Search) |
| `TR_RU` | `85` | `[1, 1, 1]` | `[0, 1, 1, 12]` | `n` | M7 Freeze (No Search) |
| `TR_RU` | `TOTAL` | `[1, 1, 1]` | `[0, 1, 0, 12]` | `n` | M7 Freeze (No Search) |
| `RU_TR` | `27` | `[1, 1, 0]` | `[0, 1, 1, 12]` | `n` | M7 Freeze (No Search) |
| `RU_TR` | `72` | `[0, 1, 1]` | `[0, 1, 1, 12]` | `n` | M7 Freeze (No Search) |
| `RU_TR` | `10` | `[0, 1, 1]` | `[0, 1, 1, 12]` | `n` | M7 Freeze (No Search) |
| `RU_TR` | `76` | `[1, 1, 1]` | `[0, 1, 1, 12]` | `n` | M7 Freeze (No Search) |
| `RU_TR` | `23` | `[0, 1, 1]` | `[0, 1, 1, 12]` | `n` | M7 Freeze (No Search) |
| `RU_TR` | `74` | `[0, 1, 1]` | `[0, 1, 1, 12]` | `n` | M7 Freeze (No Search) |
| `RU_TR` | `TOTAL` | `[0, 1, 1]` | `[0, 1, 1, 12]` | `n` | M7 Freeze (No Search) |

### Exogenous Regressors
- `usd_try_lagged`: OECD USD/TRY monthly average (M+1 Day 1 release lag)
- `rub_usd_lagged`: OECD RUB/USD (USD/RUB quotation) monthly average (M+1 Day 1 release lag)
- `brent_lagged`: IMF Brent crude spot price monthly average (M+1 Day 1 release lag)
- `tr_ip_lagged`: TÜİK Türkiye industrial production index (M+2 Day 15 release lag)
- `ru_ip_lagged`: Rosstat Russia industrial production index (M+2 Day 15 release lag)

---

## 5. Walk-Forward Backtest & Convergence Audit

- **Total Unique Fits**: `1344` (14 series × 96 unique origins)
- **Successful Fits**: `1344` (100.0%)
- **Converged Fits**: `1344` (100.0%)
- **Primary Optimizer (`lbfgs`) Used**: `1344`
- **Retry Optimizer (`powell`) Used**: `0`
- **Failed Fits Logged**: `0`
- **Valid Predictions Generated**: `3934` / `3934` (100.0%)

---

## 6. Prediction Intervals & Invariants

Native statistical prediction intervals generated under origin-known persistence:
- **80% Intervals** ($\alpha = 0.20$): $z_{0.90} \approx 1.282$
- **95% Intervals** ($\alpha = 0.05$): $z_{0.975} \approx 1.960$

Interval Invariant Checks:
- $\text{lower}_{95} \le \text{lower}_{80} \le \text{upper}_{80} \le \text{upper}_{95}$: **VERIFIED (0 violations)**
- Negative point forecasts: `27` (raw values preserved)
- Negative 80% lower bounds: `502` (raw values preserved)
- Negative 95% lower bounds: `1126` (raw values preserved)

---

## 7. Validation Performance Metrics

| Direction | HS2 | Horizon | SARIMAX sMAPE (%) | SARIMAX MAE ($) | SARIMAX RMSE ($) | Cov 80% | Cov 95% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `RU_TR` | `10` | `h=1` | 37.15% | $54,627,287 | $75,624,460 | 72.9% | 88.5% |
| `RU_TR` | `10` | `h=3` | 49.66% | $81,194,736 | $107,881,810 | 68.1% | 90.4% |
| `RU_TR` | `10` | `h=6` | 52.25% | $90,500,993 | $120,419,716 | 79.1% | 92.3% |
| `RU_TR` | `23` | `h=1` | 27.69% | $10,352,584 | $13,906,753 | 75.0% | 90.6% |
| `RU_TR` | `23` | `h=3` | 30.96% | $12,068,854 | $16,740,306 | 75.5% | 88.3% |
| `RU_TR` | `23` | `h=6` | 33.93% | $13,539,607 | $18,066,517 | 85.7% | 93.4% |
| `RU_TR` | `27` | `h=1` | 21.66% | $358,004,821 | $511,881,450 | 78.1% | 86.5% |
| `RU_TR` | `27` | `h=3` | 30.02% | $468,779,122 | $649,656,787 | 78.7% | 88.3% |
| `RU_TR` | `27` | `h=6` | 38.43% | $610,412,935 | $864,616,351 | 80.2% | 87.9% |
| `RU_TR` | `72` | `h=1` | 24.15% | $72,115,760 | $92,389,943 | 82.3% | 94.8% |
| `RU_TR` | `72` | `h=3` | 30.72% | $92,914,397 | $117,306,922 | 80.9% | 95.7% |
| `RU_TR` | `72` | `h=6` | 36.96% | $112,432,275 | $138,209,575 | 82.4% | 97.8% |
| `RU_TR` | `74` | `h=1` | 57.04% | $40,177,904 | $58,277,215 | 63.5% | 79.2% |
| `RU_TR` | `74` | `h=3` | 65.11% | $46,435,933 | $67,552,546 | 71.3% | 81.9% |
| `RU_TR` | `74` | `h=6` | 76.36% | $60,494,527 | $91,085,857 | 69.2% | 83.5% |
| `RU_TR` | `76` | `h=1` | 31.57% | $25,638,479 | $34,756,827 | 82.3% | 93.8% |
| `RU_TR` | `76` | `h=3` | 34.40% | $27,351,554 | $35,645,936 | 85.1% | 96.8% |
| `RU_TR` | `76` | `h=6` | 48.75% | $35,856,093 | $45,056,904 | 82.4% | 100.0% |
| `RU_TR` | `TOTAL` | `h=1` | 16.89% | $436,666,924 | $624,062,858 | 81.2% | 90.6% |
| `RU_TR` | `TOTAL` | `h=3` | 22.64% | $586,568,658 | $771,124,532 | 76.6% | 87.2% |
| `RU_TR` | `TOTAL` | `h=6` | 28.30% | $728,652,465 | $985,682,141 | 73.6% | 85.7% |
| `TR_RU` | `08` | `h=1` | 27.01% | $16,108,007 | $22,712,123 | 77.1% | 88.5% |
| `TR_RU` | `08` | `h=3` | 34.65% | $20,627,049 | $26,431,632 | 80.9% | 94.7% |
| `TR_RU` | `08` | `h=6` | 35.31% | $23,961,878 | $31,368,499 | 90.1% | 98.9% |
| `TR_RU` | `39` | `h=1` | 17.34% | $4,374,469 | $6,161,953 | 69.8% | 83.3% |
| `TR_RU` | `39` | `h=3` | 24.18% | $7,136,682 | $10,842,819 | 71.3% | 89.4% |
| `TR_RU` | `39` | `h=6` | 33.23% | $10,663,950 | $17,112,930 | 71.4% | 82.4% |
| `TR_RU` | `73` | `h=1` | 21.56% | $2,822,732 | $3,690,785 | 80.2% | 90.6% |
| `TR_RU` | `73` | `h=3` | 25.35% | $3,706,614 | $4,839,183 | 86.2% | 95.7% |
| `TR_RU` | `73` | `h=6` | 30.10% | $4,434,548 | $5,653,666 | 89.0% | 97.8% |
| `TR_RU` | `84` | `h=1` | 20.93% | $16,114,840 | $23,751,326 | 64.6% | 79.2% |
| `TR_RU` | `84` | `h=3` | 24.16% | $24,201,266 | $35,869,637 | 70.2% | 81.9% |
| `TR_RU` | `84` | `h=6` | 28.06% | $33,704,960 | $56,872,852 | 75.8% | 81.3% |
| `TR_RU` | `85` | `h=1` | 23.61% | $5,784,930 | $7,998,632 | 75.0% | 89.6% |
| `TR_RU` | `85` | `h=3` | 29.49% | $8,428,424 | $12,414,200 | 77.7% | 86.2% |
| `TR_RU` | `85` | `h=6` | 38.45% | $12,318,487 | $19,528,049 | 78.0% | 81.3% |
| `TR_RU` | `87` | `h=1` | 23.27% | $7,984,026 | $10,287,561 | 93.8% | 97.9% |
| `TR_RU` | `87` | `h=3` | 28.99% | $9,286,805 | $12,435,622 | 96.8% | 98.9% |
| `TR_RU` | `87` | `h=6` | 36.00% | $11,731,787 | $15,251,438 | 97.8% | 100.0% |
| `TR_RU` | `TOTAL` | `h=1` | 18.93% | $89,845,753 | $118,324,493 | 62.5% | 81.2% |
| `TR_RU` | `TOTAL` | `h=3` | 23.60% | $124,022,395 | $173,840,773 | 69.1% | 83.0% |
| `TR_RU` | `TOTAL` | `h=6` | 29.72% | $176,839,170 | $265,833,588 | 64.8% | 80.2% |

---

## 8. Descriptive Model Comparison (Validation Backtest)

Mean sMAPE across all 14 series by horizon:

| Horizon | Seasonal Naive sMAPE | SARIMA sMAPE | SARIMAX sMAPE | SARIMAX vs SARIMA (sMAPE diff) |
| :--- | :--- | :--- | :--- | :--- |
| `h=1` | 38.95% | 22.98% | 26.34% | +3.36% |
| `h=3` | 39.03% | 27.50% | 32.42% | +4.92% |
| `h=6` | 39.12% | 32.64% | 38.99% | +6.34% |
| **Overall Mean** | **39.03%** | **27.71%** | **32.59%** | **+4.88%** |

Key empirical observations:
- SARIMAX comfortably outperforms the Seasonal Naive baseline across all horizons ($26.34\%$ vs $38.95\%$ at $h=1$).
- However, under the publication-safe origin-known persistence policy, adding the 5 macro regressors increases validation sMAPE relative to univariate SARIMA ($26.34\%$ vs $22.98\%$ at $h=1$, and $32.59\%$ vs $27.71\%$ overall).
- In accordance with project principles, these negative/null econometric increments are transparently documented rather than obscured.
- Champion selection is deferred to M10; no model selection decisions are made at M8.

---

## 9. Full-Development Residual Diagnostics (2012-01 to 2024-12)

| Direction | HS2 | Nobs | Converged | AIC | BIC | Resid Mean ($) | Resid Std ($) | Ljung-Box p-val (lag 12) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `RU_TR` | `10` | 156 | True | 5568.2 | 5591.9 | $-435,689 | $59,550,694 | 0.1116 |
| `RU_TR` | `23` | 156 | True | 5101.7 | 5125.4 | $169,312 | $16,775,268 | 0.3474 |
| `RU_TR` | `27` | 156 | True | 6105.6 | 6129.3 | $-22,898,527 | $409,960,166 | 0.5698 |
| `RU_TR` | `72` | 156 | True | 5661.5 | 5685.2 | $-6,292,121 | $126,025,625 | 0.6063 |
| `RU_TR` | `74` | 156 | True | 5487.2 | 5510.9 | $1,369,267 | $45,371,001 | 0.0280 |
| `RU_TR` | `76` | 156 | True | 5376.9 | 5403.6 | $-2,351,206 | $34,767,218 | 0.4246 |
| `RU_TR` | `TOTAL` | 156 | True | 6174.3 | 6198.0 | $-37,640,801 | $559,997,915 | 0.6621 |
| `TR_RU` | `08` | 156 | True | 5242.9 | 5269.6 | $-354,152 | $39,705,153 | 0.0057 |
| `TR_RU` | `39` | 156 | True | 4857.5 | 4884.2 | $-75,914 | $6,356,038 | 0.0745 |
| `TR_RU` | `73` | 156 | True | 4714.8 | 4738.5 | $33,983 | $3,225,322 | 0.6671 |
| `TR_RU` | `84` | 156 | True | 5215.9 | 5239.6 | $-32,804 | $20,757,022 | 0.5552 |
| `TR_RU` | `85` | 156 | True | 4949.9 | 4976.5 | $-291,319 | $8,852,983 | 0.0529 |
| `TR_RU` | `87` | 156 | True | 5090.6 | 5114.3 | $-586,640 | $14,318,621 | 0.7005 |
| `TR_RU` | `TOTAL` | 156 | True | 5685.5 | 5709.2 | $-1,007,073 | $162,861,031 | 0.0004 |

---

## 10. Exogenous Provenance Audit

- **Total Provenance Rows Recorded**: `480` (96 origins × 5 indicators)
- **Publication Safety Invariant Violations**: `0` (0 violations)
- **Audit Artifact**: `data/processed/m8_sarimax_exog_audit.parquet`

---

## 11. Limitations & Methodological Disclaimers

1. **Deterministic Persistence Scenario**: The future macro path is an origin-known deterministic persistence scenario, not a forecast of future realized macroeconomic conditions.
2. **Conditional Prediction Intervals**: Native SARIMAX intervals condition on that exogenous path and therefore do not represent uncertainty in the future macro trajectory itself.
3. **No Causal Assertions**: Exogenous coefficients represent statistical association under simultaneous equations in statespace form; they do not imply structural causal effects.
4. **No Champion Selection**: Model comparison is strictly descriptive. Formal champion determination will be executed under the frozen M10 decision rules.
5. **Holdout Quarantine**: Calendar year 2025 (`2025-01` to `2025-12`) was strictly sealed and never accessed.

---

## 12. Acceptance Criteria Checklist

- [x] Exactly 14 frozen target series modeled
- [x] 2025 holdout remains strictly untouched
- [x] Exact M6/M7 validation folds reused (281 folds)
- [x] Fold registry unchanged
- [x] Frozen structural orders from M7 reused verbatim (no re-search)
- [x] Exactly 5 publication-safe exogenous variables consumed jointly
- [x] Future-exogenous strategy: origin_known_persistence strictly enforced
- [x] Future actual macro values strictly forbidden
- [x] Deterministic missing-exog policy enforced (fail closed loudly)
- [x] Native multi-step $h=1, 3, 6$ forecasts produced
- [x] 80% and 95% native prediction intervals generated
- [x] Interval invariants pass without violation
- [x] Convergence status and fit logs preserved
- [x] Full-development residual diagnostics generated (Ljung-Box lag 12)
- [x] Exogenous provenance audit artifact generated (`data/processed/m8_sarimax_exog_audit.parquet`)
- [x] Methodological manifest versioned (`configs/sarimax_spec_v1_1.yaml`)
- [x] Prediction artifact generated (`data/processed/m8_sarimax_predictions.parquet`)
- [x] Metrics artifact generated (`data/processed/m8_sarimax_metrics.parquet`)
- [x] Diagnostics artifact generated (`data/processed/m8_sarimax_residual_diagnostics.parquet`)
- [x] Fit log artifact generated (`data/processed/m8_sarimax_fit_log.parquet`)
- [x] Reproducibility run metadata generated (`data/processed/m8_sarimax_run.json`)
- [x] Future-macro mutation test passes
- [x] 2025 macro quarantine test passes
- [x] 2025 target mutation test passes
- [x] Full pytest test suite passes cleanly
- [x] Ruff linter passes cleanly (0 errors)
- [x] M9+ work strictly prohibited

---

## 13. Ready for External Review?

**YES**. M8 implementation is complete and ready for external review.

