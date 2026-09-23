# M7 SARIMA Baseline Report

**Generated**: 2026-09-22 17:58:30 UTC  
**Run ID**: `run_m7_sarima_v1_1`  
**Specification**: `Pack v1.1`  
**Model Type**: `SARIMA (Univariate Statistical Baseline)`  
**Status**: `FINAL CANDIDATE (READY FOR EXTERNAL REVIEW)`  

---

## 1. Specification / Scope

Milestone M7 implements the project's first fitted statistical forecasting baseline: univariate Seasonal Autoregressive Integrated Moving Average (SARIMA) across all 14 frozen target series. The model is evaluated using the exact expanding walk-forward validation fold registry established in M6.

Key scope boundaries enforced:
- **Strictly Univariate**: `exog=None`. No macro regressors, M5 engineered predictors, or exogenous indicators are consumed.
- **Strict Anti-Leakage Order Selection**: Order selection is conducted strictly on the initial 60-month training window (`2012-01` to `2016-12`). Validation data (`2017-01` to `2024-12`) is strictly isolated.
- **Controlled Candidate Search Grid**: Exactly $2^6 = 64$ candidate orders evaluated per series ($p,d,q,P,D,Q \in \{0, 1\}, s=12$).
- **Deterministic Tie-Breaking**: Winning specification chosen by lowest AIC, then lowest BIC, structural complexity, and lexicographic tuple.
- **Native Multi-Step Uncertainty**: 80% ($\alpha=0.20$) and 95% ($\alpha=0.05$) prediction intervals computed from the model's native forecast distribution.
- **Strict Holdout Quarantine**: The 2025 calendar year (`2025-01` to `2025-12`) remains completely sealed and untouched.
- **No Champion Selection**: No model selection or champion designation is performed (deferred to M10).

---

## 2. Frozen Inputs

| Manifest / Artifact | Path | Checksum (SHA-256) | Status |
| :--- | :--- | :--- | :--- |
| Selected Sectors Manifest | `configs/selected_sectors_v1_1.yaml` | `c6e8f524657d8e54...` | FROZEN (M3) |
| Feature Spec Manifest | `configs/feature_spec_v1_1.yaml` | `873f98023b0b7e1e...` | FROZEN (M5) |
| Validation Folds Manifest | `configs/validation_folds_v1_1.yaml` | `dede0d85e8cd1c89...` | FROZEN (M6) |
| SARIMA Orders Manifest | `configs/sarima_orders_v1_1.yaml` | `b9b2671e542e751a...` | FROZEN (M7) |
| Development Target Features | `data/processed/model_features_development.parquet` | `dev_features_0328922bf4e2` | FROZEN (M5) |

---

## 3. Validation Fold Integrity

M7 loads and evaluates the exact 281 expanding walk-forward folds from M6:
- **Total Folds**: `281`
- **Horizon $h=1$**: `96` origins (`2016-12` to `2024-11`, forecast periods `2017-01` to `2024-12`)
- **Horizon $h=3$**: `94` origins (`2016-12` to `2024-09`, forecast periods `2017-03` to `2024-12`)
- **Horizon $h=6$**: `91` origins (`2016-12` to `2024-06`, forecast periods `2017-06` to `2024-12`)
- **Total Expected Predictions**: $14 \text{ series} \times (96 + 94 + 91) = 3,934$ rows.

---

## 4. Order Selection Protocol & Selected Orders

Order selection was conducted strictly on the initial 60-month window (`2012-01` through `2016-12`).
A controlled grid of 64 candidates was evaluated per series ($p,d,q,P,D,Q \in \{0, 1\}, s=12$) with trend policy `trend='c' if d==0 and D==0 else 'n'`.

### Frozen Selected Orders (14 Series)

| Direction | HS2 | Description | Order $(p,d,q)$ | Seasonal $(P,D,Q)_{12}$ | Trend | AIC | BIC | Valid / Total Cand. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `TR_RU` | `08` | HS2 08 | `(1, 1, 1)` | `(1, 1, 0, 12)` | `n` | 1698.84 | 1706.24 | 64 / 64 |
| `TR_RU` | `84` | HS2 84 | `(0, 1, 0)` | `(1, 1, 1, 12)` | `n` | 1637.81 | 1643.36 | 64 / 64 |
| `TR_RU` | `39` | HS2 39 | `(1, 1, 0)` | `(1, 1, 1, 12)` | `n` | 1546.15 | 1553.55 | 64 / 64 |
| `TR_RU` | `87` | HS2 87 | `(1, 1, 0)` | `(0, 1, 1, 12)` | `n` | 1696.48 | 1702.03 | 64 / 64 |
| `TR_RU` | `73` | HS2 73 | `(1, 1, 0)` | `(0, 1, 1, 12)` | `n` | 1539.08 | 1544.63 | 64 / 64 |
| `TR_RU` | `85` | HS2 85 | `(1, 1, 1)` | `(0, 1, 1, 12)` | `n` | 1614.92 | 1622.32 | 64 / 64 |
| `TR_RU` | `TOTAL` | Total Bilateral Flow | `(1, 1, 1)` | `(0, 1, 0, 12)` | `n` | 1815.33 | 1820.88 | 64 / 64 |
| `RU_TR` | `27` | HS2 27 | `(1, 1, 0)` | `(0, 1, 1, 12)` | `n` | 1967.76 | 1973.31 | 63 / 64 |
| `RU_TR` | `72` | HS2 72 | `(0, 1, 1)` | `(0, 1, 1, 12)` | `n` | 1833.48 | 1839.03 | 64 / 64 |
| `RU_TR` | `10` | HS2 10 | `(0, 1, 1)` | `(0, 1, 1, 12)` | `n` | 1789.00 | 1794.56 | 64 / 64 |
| `RU_TR` | `76` | HS2 76 | `(1, 1, 1)` | `(0, 1, 1, 12)` | `n` | 1751.00 | 1758.41 | 64 / 64 |
| `RU_TR` | `23` | HS2 23 | `(0, 1, 1)` | `(0, 1, 1, 12)` | `n` | 1641.61 | 1647.16 | 64 / 64 |
| `RU_TR` | `74` | HS2 74 | `(0, 1, 1)` | `(0, 1, 1, 12)` | `n` | 1723.58 | 1729.13 | 64 / 64 |
| `RU_TR` | `TOTAL` | Total Bilateral Flow | `(0, 1, 1)` | `(0, 1, 1, 12)` | `n` | 2005.74 | 2011.29 | 64 / 64 |

---

## 5. Walk-Forward Backtest & Convergence Audit

- **Unique Models Fitted**: 14 series × 96 unique origins = `1,344` fitted series/origin models (100.0% converged with `lbfgs`)
- **Total Horizon Prediction Rows Derived**: `3,934`
- **Valid Predictions**: `3,934` (100.0%)
- **Failed Predictions**: `0`
- **Primary Optimizer (`lbfgs`) Converged**: `3,934`
- **Optimizer Retries (`powell`)**: `0`
- **Residual Failures Logged**: `0`

---

## 6. Prediction Intervals & Invariants

Native statistical prediction intervals were generated for all valid folds:
- 80% Intervals ($\alpha = 0.20$): $z_{0.90} \approx 1.282$
- 95% Intervals ($\alpha = 0.05$): $z_{0.975} \approx 1.960$

Interval Invariant Checks:
- $\text{lower}_{95} \le \text{lower}_{80} \le \hat{y} \le \text{upper}_{80} \le \text{upper}_{95}$: **VERIFIED (0 violations)**
- $\text{width}_{95} \ge \text{width}_{80} \ge 0$: **VERIFIED (0 violations)**

Negative Value Preservation:
- Negative point forecasts: `0` (raw values preserved)
- Negative 80% lower bounds: `346` (raw values preserved)
- Negative 95% lower bounds: `989` (raw values preserved)

---

## 7. Results by Series and Horizon (All 42 Groups)

| Direction | HS2 | Horizon | Valid / Exp | sMAPE (%) | MAE (USD) | RMSE (USD) | 80% Cov | 95% Cov | Mean Width 80 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `RU_TR` | `10` | $h=1$ | 96/96 | 33.49% | $54,959,249 | $76,778,969 | 72.9% | 86.5% | $154,705,798 |
| `RU_TR` | `10` | $h=3$ | 94/94 | 41.89% | $70,698,386 | $94,096,922 | 73.4% | 88.3% | $186,468,449 |
| `RU_TR` | `10` | $h=6$ | 91/91 | 43.62% | $74,458,585 | $98,784,367 | 75.8% | 90.1% | $222,316,911 |
| `RU_TR` | `23` | $h=1$ | 96/96 | 22.59% | $9,251,388 | $13,234,930 | 79.2% | 92.7% | $29,555,835 |
| `RU_TR` | `23` | $h=3$ | 94/94 | 25.23% | $10,407,958 | $14,746,408 | 79.8% | 90.4% | $32,718,377 |
| `RU_TR` | `23` | $h=6$ | 91/91 | 25.55% | $10,339,477 | $14,421,865 | 84.6% | 95.6% | $36,920,821 |
| `RU_TR` | `27` | $h=1$ | 96/96 | 18.97% | $324,956,698 | $479,024,946 | 85.4% | 89.6% | $998,289,259 |
| `RU_TR` | `27` | $h=3$ | 94/94 | 27.46% | $465,434,281 | $654,900,715 | 81.9% | 90.4% | $1,392,386,065 |
| `RU_TR` | `27` | $h=6$ | 91/91 | 35.99% | $614,710,550 | $900,907,739 | 81.3% | 86.8% | $1,839,565,679 |
| `RU_TR` | `72` | $h=1$ | 96/96 | 23.31% | $68,734,314 | $86,936,240 | 86.5% | 95.8% | $251,165,053 |
| `RU_TR` | `72` | $h=3$ | 94/94 | 26.87% | $81,403,793 | $101,225,388 | 78.7% | 97.9% | $282,890,130 |
| `RU_TR` | `72` | $h=6$ | 91/91 | 32.05% | $97,491,932 | $118,883,074 | 83.5% | 96.7% | $323,733,949 |
| `RU_TR` | `74` | $h=1$ | 96/96 | 46.69% | $34,753,132 | $54,336,838 | 71.9% | 88.5% | $79,928,640 |
| `RU_TR` | `74` | $h=3$ | 94/94 | 50.10% | $38,799,402 | $60,659,962 | 72.3% | 83.0% | $88,433,304 |
| `RU_TR` | `74` | $h=6$ | 91/91 | 58.81% | $47,018,800 | $75,679,627 | 70.3% | 85.7% | $98,500,582 |
| `RU_TR` | `76` | $h=1$ | 96/96 | 27.52% | $23,703,209 | $32,321,731 | 85.4% | 93.8% | $84,218,356 |
| `RU_TR` | `76` | $h=3$ | 94/94 | 30.61% | $25,864,972 | $33,859,062 | 85.1% | 94.7% | $95,731,120 |
| `RU_TR` | `76` | $h=6$ | 91/91 | 38.14% | $32,174,751 | $39,315,133 | 86.8% | 98.9% | $113,454,495 |
| `RU_TR` | `TOTAL` | $h=1$ | 96/96 | 14.07% | $387,854,303 | $553,476,091 | 84.4% | 92.7% | $1,360,347,264 |
| `RU_TR` | `TOTAL` | $h=3$ | 94/94 | 19.17% | $537,670,028 | $713,620,687 | 77.7% | 94.7% | $1,688,057,369 |
| `RU_TR` | `TOTAL` | $h=6$ | 91/91 | 25.77% | $726,850,795 | $991,389,847 | 75.8% | 87.9% | $2,070,109,062 |
| `TR_RU` | `08` | $h=1$ | 96/96 | 22.20% | $13,655,009 | $19,954,105 | 82.3% | 91.7% | $45,538,060 |
| `TR_RU` | `08` | $h=3$ | 94/94 | 25.33% | $16,107,375 | $21,535,362 | 85.1% | 95.7% | $59,204,869 |
| `TR_RU` | `08` | $h=6$ | 91/91 | 27.09% | $18,106,503 | $23,705,280 | 84.6% | 95.6% | $70,527,289 |
| `TR_RU` | `39` | $h=1$ | 96/96 | 15.94% | $4,272,935 | $6,136,911 | 70.8% | 87.5% | $10,626,712 |
| `TR_RU` | `39` | $h=3$ | 94/94 | 21.58% | $6,920,420 | $10,789,672 | 75.5% | 89.4% | $15,799,170 |
| `TR_RU` | `39` | $h=6$ | 91/91 | 27.28% | $10,240,279 | $17,217,194 | 74.7% | 84.6% | $21,147,797 |
| `TR_RU` | `73` | $h=1$ | 96/96 | 18.16% | $2,526,976 | $3,439,217 | 81.2% | 94.8% | $9,010,933 |
| `TR_RU` | `73` | $h=3$ | 94/94 | 22.23% | $3,363,893 | $4,600,405 | 87.2% | 95.7% | $12,703,590 |
| `TR_RU` | `73` | $h=6$ | 91/91 | 26.78% | $3,983,697 | $5,252,918 | 87.9% | 96.7% | $16,956,723 |
| `TR_RU` | `84` | $h=1$ | 96/96 | 17.50% | $14,241,282 | $21,667,454 | 72.9% | 84.4% | $31,933,404 |
| `TR_RU` | `84` | $h=3$ | 94/94 | 20.33% | $20,403,309 | $31,052,239 | 78.7% | 88.3% | $54,417,864 |
| `TR_RU` | `84` | $h=6$ | 91/91 | 25.32% | $28,618,579 | $48,383,142 | 78.0% | 84.6% | $74,931,816 |
| `TR_RU` | `85` | $h=1$ | 96/96 | 22.19% | $5,626,193 | $7,784,877 | 82.3% | 87.5% | $16,425,475 |
| `TR_RU` | `85` | $h=3$ | 94/94 | 26.94% | $8,119,783 | $12,230,336 | 79.8% | 88.3% | $23,752,228 |
| `TR_RU` | `85` | $h=6$ | 91/91 | 32.83% | $11,727,266 | $19,358,816 | 79.1% | 83.5% | $31,428,520 |
| `TR_RU` | `87` | $h=1$ | 96/96 | 21.05% | $7,597,786 | $10,132,926 | 93.8% | 97.9% | $38,612,180 |
| `TR_RU` | `87` | $h=3$ | 94/94 | 25.46% | $8,979,435 | $12,138,818 | 96.8% | 100.0% | $58,950,198 |
| `TR_RU` | `87` | $h=6$ | 91/91 | 30.01% | $11,068,211 | $14,633,019 | 97.8% | 100.0% | $81,157,144 |
| `TR_RU` | `TOTAL` | $h=1$ | 96/96 | 18.07% | $88,347,979 | $119,477,553 | 64.6% | 79.2% | $185,726,057 |
| `TR_RU` | `TOTAL` | $h=3$ | 94/94 | 21.79% | $121,566,519 | $179,880,768 | 72.3% | 83.0% | $269,950,299 |
| `TR_RU` | `TOTAL` | $h=6$ | 91/91 | 27.80% | $177,274,246 | $276,672,293 | 70.3% | 79.1% | $358,279,483 |

---

## 8. Descriptive Seasonal Naive Comparison (Non-Champion)

Note: This comparison is purely descriptive. Champion selection is strictly reserved for M10. A negative $\Delta$ indicates SARIMA had lower error than Seasonal Naive.

| Direction | HS2 | Horizon | SARIMA sMAPE | M6 sMAPE | $\Delta$ sMAPE | SARIMA MAE | M6 MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `RU_TR` | `10` | $h=1$ | 33.49% | 53.11% | **-19.62%** | $54,959,249 | $85,327,521 |
| `RU_TR` | `10` | $h=3$ | 41.89% | 53.78% | **-11.89%** | $70,698,386 | $86,919,936 |
| `RU_TR` | `10` | $h=6$ | 43.62% | 53.99% | **-10.37%** | $74,458,585 | $89,046,596 |
| `RU_TR` | `23` | $h=1$ | 22.59% | 31.36% | **-8.77%** | $9,251,388 | $12,494,194 |
| `RU_TR` | `23` | $h=3$ | 25.23% | 31.11% | **-5.88%** | $10,407,958 | $12,486,238 |
| `RU_TR` | `23` | $h=6$ | 25.55% | 31.59% | **-6.04%** | $10,339,477 | $12,769,189 |
| `RU_TR` | `27` | $h=1$ | 18.97% | 43.93% | **-24.96%** | $324,956,698 | $710,838,536 |
| `RU_TR` | `27` | $h=3$ | 27.46% | 44.57% | **-17.10%** | $465,434,281 | $723,659,745 |
| `RU_TR` | `27` | $h=6$ | 35.99% | 45.04% | **-9.06%** | $614,710,550 | $739,917,102 |
| `RU_TR` | `72` | $h=1$ | 23.31% | 38.16% | **-14.85%** | $68,734,314 | $116,403,608 |
| `RU_TR` | `72` | $h=3$ | 26.87% | 38.26% | **-11.38%** | $81,403,793 | $117,349,572 |
| `RU_TR` | `72` | $h=6$ | 32.05% | 38.37% | **-6.32%** | $97,491,932 | $118,727,871 |
| `RU_TR` | `74` | $h=1$ | 46.69% | 65.40% | **-18.72%** | $34,753,132 | $47,552,902 |
| `RU_TR` | `74` | $h=3$ | 50.10% | 64.76% | **-14.65%** | $38,799,402 | $47,845,312 |
| `RU_TR` | `74` | $h=6$ | 58.81% | 65.57% | **-6.76%** | $47,018,800 | $48,864,161 |
| `RU_TR` | `76` | $h=1$ | 27.52% | 40.61% | **-13.09%** | $23,703,209 | $33,071,414 |
| `RU_TR` | `76` | $h=3$ | 30.61% | 40.09% | **-9.48%** | $25,864,972 | $33,032,700 |
| `RU_TR` | `76` | $h=6$ | 38.14% | 40.15% | **-2.01%** | $32,174,751 | $33,420,646 |
| `RU_TR` | `TOTAL` | $h=1$ | 14.07% | 31.33% | **-17.26%** | $387,854,303 | $845,670,102 |
| `RU_TR` | `TOTAL` | $h=3$ | 19.17% | 31.78% | **-12.61%** | $537,670,028 | $860,683,333 |
| `RU_TR` | `TOTAL` | $h=6$ | 25.77% | 32.13% | **-6.36%** | $726,850,795 | $879,366,800 |
| `TR_RU` | `08` | $h=1$ | 22.20% | 34.17% | **-11.97%** | $13,655,009 | $19,838,200 |
| `TR_RU` | `08` | $h=3$ | 25.33% | 33.45% | **-8.12%** | $16,107,375 | $19,864,601 |
| `TR_RU` | `08` | $h=6$ | 27.09% | 33.24% | **-6.15%** | $18,106,503 | $20,302,195 |
| `TR_RU` | `39` | $h=1$ | 15.94% | 34.29% | **-18.35%** | $4,272,935 | $10,997,853 |
| `TR_RU` | `39` | $h=3$ | 21.58% | 34.73% | **-13.15%** | $6,920,420 | $11,210,940 |
| `TR_RU` | `39` | $h=6$ | 27.28% | 35.03% | **-7.75%** | $10,240,279 | $11,507,027 |
| `TR_RU` | `73` | $h=1$ | 18.16% | 29.98% | **-11.81%** | $2,526,976 | $4,174,094 |
| `TR_RU` | `73` | $h=3$ | 22.23% | 29.86% | **-7.62%** | $3,363,893 | $4,239,162 |
| `TR_RU` | `73` | $h=6$ | 26.78% | 29.69% | **-2.91%** | $3,983,697 | $4,324,512 |
| `TR_RU` | `84` | $h=1$ | 17.50% | 36.49% | **-18.98%** | $14,241,282 | $33,932,325 |
| `TR_RU` | `84` | $h=3$ | 20.33% | 36.61% | **-16.28%** | $20,403,309 | $34,550,901 |
| `TR_RU` | `84` | $h=6$ | 25.32% | 36.67% | **-11.35%** | $28,618,579 | $35,462,366 |
| `TR_RU` | `85` | $h=1$ | 22.19% | 41.87% | **-19.69%** | $5,626,193 | $13,278,882 |
| `TR_RU` | `85` | $h=3$ | 26.94% | 42.63% | **-15.68%** | $8,119,783 | $13,549,913 |
| `TR_RU` | `85` | $h=6$ | 32.83% | 42.29% | **-9.46%** | $11,727,266 | $13,831,486 |
| `TR_RU` | `87` | $h=1$ | 21.05% | 33.37% | **-12.32%** | $7,597,786 | $11,215,514 |
| `TR_RU` | `87` | $h=3$ | 25.46% | 33.68% | **-8.22%** | $8,979,435 | $11,408,365 |
| `TR_RU` | `87` | $h=6$ | 30.01% | 33.10% | **-3.09%** | $11,068,211 | $11,509,798 |
| `TR_RU` | `TOTAL` | $h=1$ | 18.07% | 31.17% | **-13.10%** | $88,347,979 | $161,699,432 |
| `TR_RU` | `TOTAL` | $h=3$ | 21.79% | 31.10% | **-9.32%** | $121,566,519 | $164,137,868 |
| `TR_RU` | `TOTAL` | $h=6$ | 27.80% | 30.89% | **-3.09%** | $177,274,246 | $167,442,241 |

---

## 9. Full-Development Residual Diagnostics

Diagnostic model fits on complete development sample (`2012-01` to `2024-12`, 156 observations):

| Direction | HS2 | Order | Residual Mean | Residual Std | LB Stat (lag 12) | LB p-value | Converged |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `TR_RU` | `08` | `(1, 1, 1) (1, 1, 0, 12)` | $-1,111,534 | $18,816,087 | 8.14 | 0.7741 | YES |
| `TR_RU` | `84` | `(0, 1, 0) (1, 1, 1, 12)` | $-171,705 | $17,451,207 | 30.95 | 0.0020 | YES |
| `TR_RU` | `39` | `(1, 1, 0) (1, 1, 1, 12)` | $-45,142 | $5,341,393 | 21.96 | 0.0380 | YES |
| `TR_RU` | `87` | `(1, 1, 0) (0, 1, 1, 12)` | $-418,417 | $13,305,156 | 8.73 | 0.7261 | YES |
| `TR_RU` | `73` | `(1, 1, 0) (0, 1, 1, 12)` | $-99,282 | $3,254,163 | 9.66 | 0.6455 | YES |
| `TR_RU` | `85` | `(1, 1, 1) (0, 1, 1, 12)` | $-261,559 | $8,006,941 | 21.23 | 0.0471 | YES |
| `TR_RU` | `TOTAL` | `(1, 1, 1) (0, 1, 0, 12)` | $-1,982,994 | $109,562,961 | 36.60 | 0.0003 | YES |
| `RU_TR` | `27` | `(1, 1, 0) (0, 1, 1, 12)` | $-2,898,656 | $422,422,823 | 9.50 | 0.6593 | YES |
| `RU_TR` | `72` | `(0, 1, 1) (0, 1, 1, 12)` | $-4,298,759 | $77,850,194 | 8.53 | 0.7425 | YES |
| `RU_TR` | `10` | `(0, 1, 1) (0, 1, 1, 12)` | $-2,512,703 | $60,999,006 | 16.66 | 0.1627 | YES |
| `RU_TR` | `76` | `(1, 1, 1) (0, 1, 1, 12)` | $-2,153,520 | $29,911,857 | 15.26 | 0.2275 | YES |
| `RU_TR` | `23` | `(0, 1, 1) (0, 1, 1, 12)` | $-96,675 | $11,497,906 | 19.27 | 0.0823 | YES |
| `RU_TR` | `74` | `(0, 1, 1) (0, 1, 1, 12)` | $-75,358 | $42,232,612 | 27.41 | 0.0067 | YES |
| `RU_TR` | `TOTAL` | `(0, 1, 1) (0, 1, 1, 12)` | $-17,171,639 | $527,542,541 | 11.02 | 0.5273 | YES |

---

## 10. Holdout Isolation & Mutation Verification

The platform strictly distinguishes between structural isolation and explicit upstream mutation verification:

### A. Structural Holdout Isolation
- All training data strictly terminates at `forecast_origin <= 2024-11`.
- All forecast target evaluation periods strictly satisfy `forecast_period <= 2024-12`.
- Zero 2025 observations appear in predictions, metrics, diagnostics, or validation folds.

### B. Explicit Upstream Holdout Mutation Verification
- All `1,987` holdout observations for 2025 in `data/processed/forecast_target_monthly.parquet` were radically mutated in memory: $y_{\text{mutated}} = y \times 999.0 + 10^{12}$.
- The accepted M5 development feature pipeline (`build_development_features`) was re-executed against the mutated upstream source.
- Reconstructed development features: **100% IDENTICAL** to canonical `model_features_development.parquet`.
- The M7 SARIMA pipeline was re-executed in temporary scratch storage against the rebuilt features:
  - Selected SARIMA Orders: **100% IDENTICAL**
  - Validation Folds: **100% IDENTICAL**
  - Point Predictions & Prediction Intervals: **100% NUMERICALLY IDENTICAL**
  - 42-Group Validation Metrics: **100% NUMERICALLY IDENTICAL**
  - Full-Development Residual Diagnostics: **100% NUMERICALLY IDENTICAL**
  - Fit Failures: **EXACTLY 0**

---

## 11. Limitations & Deviations

1. **Gaussian Prediction Intervals**: SARIMA predictive distributions are mathematically unbounded Gaussians, which can produce negative lower confidence bounds on volatile or zero-adjacent series. In compliance with the specification, bounds are preserved as raw values without artificial clipping.
2. **Controlled Search Grid**: Search space was deliberately constrained to 64 candidate specifications ($p,d,q,P,D,Q \in \{0, 1\}$) to prevent overfitting and combinatorial explosion.
3. **Fixed Specification Over Expanding Windows**: The single winning specification chosen on the pre-validation 60-month window (`2012-01` to `2016-12`) is held fixed across all subsequent validation origins. Structural breaks (such as the 2022 geopolitical sanctions) may alter true optimal dynamics.
4. **Development-Only Validation**: All metrics evaluate development origins only (`2017-01` to `2024-12`). The 2025 holdout remains strictly sealed.

---

## 12. Acceptance Criteria Checklist

- [x] Exactly 14 frozen target series modeled
- [x] 2025 holdout remains strictly untouched
- [x] Exact M6 validation folds reused (281 folds)
- [x] Fold registry unchanged
- [x] Order-selection strategy documented and auditable
- [x] Order search strictly restricted to initial 60-month training window (2012-01 to 2016-12)
- [x] Later validation data cannot alter selected orders
- [x] Controlled search grid exactly enforced (64 candidates per series)
- [x] One fixed SARIMA specification per series
- [x] Native multi-step $h=1, 3, 6$ used
- [x] No exogenous or macro variables used (`exog=None`)
- [x] sMAPE, MAE, RMSE match M6 metric definitions
- [x] 80% native prediction intervals generated
- [x] 95% native prediction intervals generated
- [x] Interval invariants pass without violation
- [x] Convergence status logged
- [x] Candidate search failures logged
- [x] Fold fit failures logged (pipeline fails gracefully)
- [x] No hidden Seasonal Naive fallback
- [x] Full-development residual diagnostics generated (Ljung-Box lag 12)
- [x] Order manifest versioned (`configs/sarima_orders_v1_1.yaml`)
- [x] Order search audit persisted (`data/processed/m7_sarima_order_search.parquet`)
- [x] Prediction artifact generated (`data/processed/m7_sarima_predictions.parquet`)
- [x] Metrics artifact generated (`data/processed/m7_sarima_metrics.parquet`)
- [x] Diagnostics artifact generated (`data/processed/m7_sarima_diagnostics.parquet`)
- [x] Failure artifact generated (`data/processed/m7_sarima_failures.parquet`)
- [x] Reproducibility run metadata generated (`data/processed/m7_sarima_run.json`)
- [x] Holdout mutation test passes
- [x] Order-selection leakage test passes
- [x] Prior frozen manifests remain intact
- [x] Full pytest test suite passes cleanly
- [x] Ruff linter passes cleanly (0 errors)
- [x] M8+ artifacts strictly prohibited

---

## 13. Ready for M8?

**YES**. All M7 requirements, constraints, invariants, and artifacts are verified and complete.

