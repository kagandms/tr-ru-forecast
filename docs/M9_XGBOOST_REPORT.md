# M9 XGBoost Direct Forecasting Report

**Generated**: 2026-10-14T11:02:01.908846+00:00  
**Run ID**: `run_m9_xgboost_v1_1_72c6bf10bc56`  
**Specification**: `Pack v1.1`  
**Model Type**: `xgboost_direct`  
**Status**: `DEVELOPMENT VALIDATION RUN`  
**Git Commit**: `9d539b8124cff22bc56b32d34b211ce087b8ec9b`  
**Working Tree Dirty**: `True`  
**Source Fingerprint**: `586d27f8cb7fe680a9a9617061d3a6119c73d01cd2bb51aadc7008a78f548eb3`  

---

## 1. Specification / Scope

Milestone M9 implements the project's primary machine-learning direct forecasting model: gradient-boosted decision trees using XGBoost across all 14 frozen target series for horizons $h=1$, $h=3$, and $h=6$.

Key scope boundaries enforced:
- **Direct Forecasting Architecture**: Separate models for each horizon ($XGB_{h1}$, $XGB_{h3}$, $XGB_{h6}$). Zero recursive forecasting.
- **Strict 21-Predictor Whitelist**: Only the exact 21 engineered features from M5 are consumed. Target labels, metadata, and future features are strictly excluded.
- **Critical Label-Availability Embargo**: Training feature rows must satisfy `training_period + h <= forecast_origin`. Labels knowable only after forecast origin are strictly prohibited from training fits.
- **Pre-Validation Time-Aware Tuning**: Hyperparameters selected strictly on calendar year 2016 targets (pre-dating the 2017–2024 official validation window). No 2017–2024 validation data influenced tuning.
- **Shared Horizon Parameter Topology**: Exactly one parameter set selected per horizon, shared across all 14 series. Zero uncontrolled per-series tuning.
- **No Early Stopping**: Fixed iteration counts prevent hidden tuning loops.
- **Prequential Conformal Uncertainty**: Development out-of-sample absolute residual calibration for 80% and 95% prediction intervals. Minimum sample size $n \ge 19$. Strict residual availability embargo ($s + h \le t$).
- **Fold-Local TreeSHAP**: Exact fold model explained for each of the 3,934 predictions with strict additive consistency verification. Global importance aggregated by series/horizon group.
- **Non-Causal Attribution**: SHAP values documented strictly as predictive model drivers, never causal relationships.
- **Holdout Exclusion**: 2025 outcomes are outside the M5 development feature table and are not scored or used by this M9 run.
- **No Champion Selection**: Model comparison is strictly descriptive; champion selection is reserved for M10.

---

## 2. Frozen Inputs & Checksums

| Manifest / Artifact | Path | Checksum (SHA-256) | Status |
| :--- | :--- | :--- | :--- |
| Selected Sectors Manifest | `configs/selected_sectors_v1_1.yaml` | `c6e8f524657d8e542468e78dce36a73f71ffc6e594fdf2429fb2144d25522734` | FROZEN (M3) |
| Feature Spec Manifest | `configs/feature_spec_v1_1.yaml` | `873f98023b0b7e1e23475f82a747862e3dc51e030ed28a2f1a76924d3b7126bb` | FROZEN (M5) |
| Macro Sources Manifest | `configs/macro_sources_v1_1.yaml` | `bca72a83fc59452973b9b6400760a1aed3152fd828c46387a2952b1b52833568` | FROZEN (M4) |
| Validation Folds Manifest | `configs/validation_folds_v1_1.yaml` | `0dc666c284991d494575a44cba18a2d93be2b3195bca01e8ae01a9f7191cded3` | FROZEN (M6) |
| Validation Folds Artifact | `data/processed/validation_origins.parquet` | `442647cedab2e8f7b516172f8ad81d047f474fecf9fe8998e75e2e5375cc9e44` | FROZEN (M6) |
| Development Target Features | `data/processed/model_features_development.parquet` | `0328922bf4e2e24694697369b1d4af92d75e59c479aacfcc72c08fd719dc88e0` | FROZEN (M5) |

Output artifact SHA-256 checksums:
- `xgboost_spec_manifest` (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/xgboost_spec_v1_1.yaml`): `48e6105d3c3e9e45fe2b690feb3cb5c75a761a50357b8f36698c58243ba8ef1d`
- `tuning_results` (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_tuning_results.parquet`): `5ff47cfd992186c7efb8ce7a92d9a10cb7306f7c14d5267a0626a3a2b15afa28`
- `predictions` (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_predictions.parquet`): `73993371fe45a0af234c0f889f09742bef41f53a9b6745bb10df2a53aaf1f567`
- `metrics` (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_metrics.parquet`): `50475f51e534cc2769a9c2b107b7b5b90e7a2fddf8bb743bc1a21fde8ae0cad1`
- `interval_audit` (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_interval_audit.parquet`): `aeb7c353a69d07de8cd216c3aac7f8e825f77f91d673bd10eb87822161bfee35`
- `shap_values` (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_shap_values.parquet`): `8df59ef271199d425cb1df986460fc0d9ff368a6d6ae59a781ab72e8fede38e4`
- `shap_global` (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_shap_global.parquet`): `dbc1f782c4d4fa33b8e7912004b66b3ded2215f4e093c2b64270c38d29bc951d`
- `fit_log` (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_fit_log.parquet`): `63f99bc71965d70af213f5971a7d45a3d3f0f436867c1c43e58c8ead7e38f207`

---

## 3. Exact 21-Predictor Whitelist

The 21 features consumed by all XGBoost models:
```text
- lag_1
- lag_2
- lag_3
- lag_6
- lag_12
- rolling_mean_3
- rolling_mean_6
- rolling_mean_12
- rolling_std_3
- rolling_std_6
- rolling_std_12
- mom_growth
- yoy_growth
- month
- quarter
- year
- usd_try_lagged
- rub_usd_lagged
- brent_lagged
- tr_ip_lagged
- ru_ip_lagged
```

Explicitly excluded from predictor set:
- Target columns: `target`, `target_h1`, `target_h3`, `target_h6`
- Metadata columns: `period`, `direction`, `hs2`, `hs_description`, `is_total`, `feature_snapshot_date`

---

## 4. Pre-Validation Tuning Design & Selected Hyperparameters

- **Tuning Evaluation Target Period**: `2016-01` to `2016-12` (12 months)
- **Candidate Grid**: 16 candidates ($2 \times 2 \times 2 \times 2$)
  - `max_depth`: [2, 3]
  - `min_child_weight`: [1, 5]
  - `learning_rate`: [0.03, 0.10]
  - `n_estimators`: [150, 300]
- **Fixed Parameters**: `objective='reg:squarederror'`, `booster='gbtree'`, `tree_method='hist'`, `subsample=1.0`, `colsample_bytree=1.0`, `reg_alpha=0.0`, `reg_lambda=1.0`, `gamma=0.0`, `random_state=42`, `n_jobs=1`
- **Selection Metric**: Arithmetic mean of the 14 series-level sMAPEs across 12 tuning predictions.
- **Total Tuning Fits**: 3 horizons $\times$ 16 candidates $\times$ 14 series $\times$ 12 origins = `8,064` fits (100% converged, 0 failures).

### Winning Hyperparameter Sets per Horizon

| Horizon | Candidate ID | max_depth | min_child_weight | learning_rate | n_estimators | Mean Tuning sMAPE | Median Tuning sMAPE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $h=1$ | `cand_05` | 2 | 5 | 0.03 | 300 | 36.83% | 33.92% |
| $h=3$ | `cand_07` | 2 | 5 | 0.10 | 300 | 45.55% | 42.52% |
| $h=6$ | `cand_01` | 2 | 1 | 0.03 | 300 | 47.30% | 49.42% |

---

## 5. Outer Validation Backtest Integrity

- **Folds Source**: `configs/validation_folds_v1_1.yaml` (281 folds)
- **Total Unique Fits**: 14 series $\times$ 281 folds = `3,934` fits
- **Successful Fits**: `3,934` (100.0%)
- **Failed Fits**: `0`
- **Valid Predictions Generated**: `3,934` / `3,934`

---

## 6. Prediction Intervals & Conformal Availability

- **Calibration Method**: Prequential out-of-sample absolute residual conformal calibration.
- **Residual Eligibility**: Earlier residual $(s, h)$ eligible at origin $t$ strictly if $s + h \le t$.
- **Minimum Calibration Pool**: $n \ge 19$ required for finite 95% intervals ($k_{95} = \lceil 20 \times 0.95 \rceil = 19$).
- **Interval-Enabled Rows**: `3,038` / `3,934` (77 origins for h1, 73 for h3, 67 for h6 per series). Exactly matches theoretical expectations.
- **Interval Invariant Checks**: $\text{lower}_{95} \le \text{lower}_{80} \le \hat{y} \le \text{upper}_{80} \le \text{upper}_{95}$: **VERIFIED (0 violations)**.
- **Raw, Unclipped Conformal Method**: Non-negativity is NOT artificially enforced. Negative point predictions and negative lower bounds are preserved raw without zero-clipping.
- **Observed Negative Values**: point predictions: `1`, lower 80% bounds: `68`, lower 95% bounds: `671`.

### Empirical Coverage Summary (with Denominators)

| Horizon | Interval-Eligible Forecasts ($n \ge 19$) | Nominal 80% | Empirical 80% Coverage | Nominal 95% | Empirical 95% Coverage |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $h=1$ | 1078 | 80.0% | **65.5%** (706/1078) | 95.0% | **91.0%** (981/1078) |
| $h=3$ | 1022 | 80.0% | **62.0%** (634/1022) | 95.0% | **89.3%** (913/1022) |
| $h=6$ | 938 | 80.0% | **62.9%** (590/938) | 95.0% | **87.5%** (821/938) |
| **Total** | **3038** | **80.0%** | **63.5%** (1930/3038) | **95.0%** | **89.4%** (2715/3038) |

---

## 7. Validation Results by Series and Horizon (All 42 Groups)

| Direction | HS2 | Horizon | Valid Folds | XGBoost sMAPE | MAE (USD) | RMSE (USD) | Cov 80% | Cov 95% | Mean Width 80 | Mean Width 95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `RU_TR` | `10` | $h=1$ | 96 | **36.83%** | $59,319,666 | $76,435,999 | 61.0% | 88.3% | $144,674,247 | $258,428,250 |
| `RU_TR` | `10` | $h=3$ | 94 | **46.37%** | $76,729,055 | $97,005,862 | 61.6% | 87.7% | $179,944,400 | $328,386,340 |
| `RU_TR` | `10` | $h=6$ | 91 | **42.57%** | $69,305,900 | $90,023,138 | 73.1% | 91.0% | $197,006,318 | $348,569,250 |
| `RU_TR` | `23` | $h=1$ | 96 | **26.13%** | $10,069,893 | $14,284,232 | 71.4% | 85.7% | $26,858,825 | $45,949,357 |
| `RU_TR` | `23` | $h=3$ | 94 | **30.60%** | $12,576,741 | $17,390,215 | 58.9% | 87.7% | $25,845,452 | $60,013,063 |
| `RU_TR` | `23` | $h=6$ | 91 | **25.91%** | $10,480,816 | $15,238,904 | 61.2% | 88.1% | $25,329,200 | $52,738,852 |
| `RU_TR` | `27` | $h=1$ | 96 | **19.37%** | $339,519,426 | $498,681,216 | 63.6% | 92.2% | $736,124,271 | $2,073,604,790 |
| `RU_TR` | `27` | $h=3$ | 94 | **28.65%** | $496,640,471 | $718,163,789 | 57.5% | 90.4% | $971,181,004 | $2,836,457,779 |
| `RU_TR` | `27` | $h=6$ | 91 | **36.28%** | $655,673,558 | $915,458,350 | 52.2% | 83.6% | $1,228,909,395 | $3,252,606,716 |
| `RU_TR` | `72` | $h=1$ | 96 | **23.62%** | $70,208,817 | $87,896,425 | 84.4% | 97.4% | $225,820,001 | $413,179,366 |
| `RU_TR` | `72` | $h=3$ | 94 | **29.17%** | $89,167,442 | $112,417,907 | 74.0% | 94.5% | $261,822,098 | $476,408,145 |
| `RU_TR` | `72` | $h=6$ | 91 | **34.01%** | $103,362,531 | $127,959,086 | 77.6% | 92.5% | $332,284,426 | $585,022,511 |
| `RU_TR` | `74` | $h=1$ | 96 | **47.13%** | $36,708,162 | $59,846,094 | 57.1% | 85.7% | $60,160,368 | $147,974,709 |
| `RU_TR` | `74` | $h=3$ | 94 | **51.71%** | $40,982,644 | $62,562,174 | 50.7% | 80.8% | $69,108,152 | $147,444,509 |
| `RU_TR` | `74` | $h=6$ | 91 | **53.22%** | $41,356,030 | $69,728,161 | 65.7% | 79.1% | $78,582,915 | $175,356,758 |
| `RU_TR` | `76` | $h=1$ | 96 | **29.10%** | $24,521,128 | $31,362,260 | 76.6% | 94.8% | $72,138,299 | $137,135,606 |
| `RU_TR` | `76` | $h=3$ | 94 | **29.88%** | $24,887,111 | $33,830,531 | 76.7% | 95.9% | $73,298,262 | $182,608,963 |
| `RU_TR` | `76` | $h=6$ | 91 | **38.81%** | $31,746,727 | $41,299,968 | 79.1% | 98.5% | $120,843,673 | $197,206,368 |
| `RU_TR` | `TOTAL` | $h=1$ | 96 | **15.19%** | $419,823,538 | $580,691,096 | 62.3% | 92.2% | $969,034,728 | $2,637,191,617 |
| `RU_TR` | `TOTAL` | $h=3$ | 94 | **22.43%** | $613,184,433 | $781,205,694 | 57.5% | 94.5% | $1,518,478,998 | $3,154,331,414 |
| `RU_TR` | `TOTAL` | $h=6$ | 91 | **26.44%** | $766,840,557 | $1,000,789,411 | 50.7% | 91.0% | $1,592,582,545 | $4,016,776,216 |
| `TR_RU` | `08` | $h=1$ | 96 | **25.35%** | $16,452,546 | $22,777,315 | 71.4% | 94.8% | $44,791,922 | $125,109,207 |
| `TR_RU` | `08` | $h=3$ | 94 | **35.46%** | $22,581,871 | $30,172,190 | 78.1% | 94.5% | $64,370,910 | $168,381,714 |
| `TR_RU` | `08` | $h=6$ | 91 | **34.66%** | $22,321,065 | $27,732,619 | 76.1% | 97.0% | $61,505,616 | $127,883,195 |
| `TR_RU` | `39` | $h=1$ | 96 | **17.09%** | $4,829,072 | $7,447,038 | 57.1% | 92.2% | $9,418,820 | $26,658,270 |
| `TR_RU` | `39` | $h=3$ | 94 | **24.31%** | $7,714,607 | $13,038,830 | 52.1% | 87.7% | $12,131,031 | $38,245,076 |
| `TR_RU` | `39` | $h=6$ | 91 | **27.94%** | $9,585,266 | $15,990,664 | 47.8% | 76.1% | $14,196,406 | $44,251,313 |
| `TR_RU` | `73` | $h=1$ | 96 | **17.49%** | $2,468,600 | $3,513,909 | 62.3% | 93.5% | $6,607,446 | $16,031,408 |
| `TR_RU` | `73` | $h=3$ | 94 | **25.89%** | $3,593,930 | $4,866,640 | 67.1% | 91.8% | $8,930,771 | $19,954,398 |
| `TR_RU` | `73` | $h=6$ | 91 | **23.23%** | $3,632,178 | $5,038,103 | 64.2% | 85.1% | $8,534,789 | $19,198,143 |
| `TR_RU` | `84` | $h=1$ | 96 | **17.00%** | $14,369,448 | $22,078,206 | 53.2% | 84.4% | $26,395,217 | $73,309,663 |
| `TR_RU` | `84` | $h=3$ | 94 | **22.05%** | $20,119,503 | $33,821,446 | 53.4% | 84.9% | $31,594,071 | $119,694,384 |
| `TR_RU` | `84` | $h=6$ | 91 | **29.79%** | $27,365,144 | $47,739,741 | 58.2% | 85.1% | $37,528,386 | $134,259,312 |
| `TR_RU` | `85` | $h=1$ | 96 | **21.95%** | $5,974,333 | $9,355,788 | 63.6% | 89.6% | $12,106,111 | $29,549,384 |
| `TR_RU` | `85` | $h=3$ | 94 | **26.74%** | $8,028,415 | $13,085,469 | 58.9% | 87.7% | $14,345,032 | $42,290,717 |
| `TR_RU` | `85` | $h=6$ | 91 | **31.94%** | $10,541,743 | $17,502,707 | 50.7% | 82.1% | $15,555,674 | $49,511,575 |
| `TR_RU` | `87` | $h=1$ | 96 | **21.17%** | $7,792,408 | $10,131,175 | 66.2% | 92.2% | $19,268,565 | $40,191,543 |
| `TR_RU` | `87` | $h=3$ | 94 | **28.93%** | $10,917,375 | $14,896,956 | 53.4% | 83.6% | $27,076,096 | $49,314,145 |
| `TR_RU` | `87` | $h=6$ | 91 | **31.94%** | $11,757,905 | $14,521,978 | 58.2% | 86.6% | $27,396,045 | $47,813,579 |
| `TR_RU` | `TOTAL` | $h=1$ | 96 | **14.83%** | $73,200,105 | $104,959,817 | 66.2% | 90.9% | $180,462,544 | $449,031,730 |
| `TR_RU` | `TOTAL` | $h=3$ | 94 | **21.38%** | $104,776,019 | $158,721,721 | 68.5% | 89.0% | $270,338,785 | $706,004,457 |
| `TR_RU` | `TOTAL` | $h=6$ | 91 | **25.44%** | $129,631,287 | $197,126,969 | 65.7% | 89.6% | $280,836,453 | $842,327,987 |

---

## 8. Descriptive Model Comparison (Non-Champion)

> [!NOTE]
> Champion selection is strictly reserved for Milestone M10. This table presents descriptive backtest comparisons only.

| Horizon | Seasonal Naive sMAPE | SARIMA sMAPE | SARIMAX sMAPE | XGBoost sMAPE | XGB vs S-Naive | XGB vs SARIMA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $h=1$ | 38.95% | 22.98% | 26.34% | **23.73%** | -15.21% | +0.76% |
| $h=3$ | 39.03% | 27.51% | 32.42% | **30.26%** | -8.77% | +2.75% |
| $h=6$ | 39.12% | 32.66% | 38.99% | **33.01%** | -6.11% | +0.36% |
| **Overall Mean** | **39.03%** | **27.71%** | **32.59%** | **29.00%** | **-10.03%** | **+1.29%** |

---

## 9. TreeSHAP Feature Attributions (Global & Local)

- **Additivity Invariant**: `base_val + sum(shap_vals) ≈ prediction` within float32 numerical precision (`rtol=1e-4, atol=1.0`) verified across all `3,934` forecasts (0 violations).
- **Global Importance Methodology**: Mean absolute SHAP is defined and reported separately for each `(direction, hs2, horizon)` group across all 42 groups (882 rows total).
- **Cross-Series Pooling Prohibition**: Universal cross-series rankings pooling raw dollar SHAP values are prohibited because series scales differ by orders of magnitude.

### Series-Specific Global Feature Importance Examples

Top-3 predictive features by mean absolute SHAP for selected canonical series groups:

| Direction | HS2 | Horizon | Top Feature 1 (Mean |SHAP|) | Top Feature 2 (Mean |SHAP|) | Top Feature 3 (Mean |SHAP|) |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `TR_RU` | `TOTAL` | $h=1$ | `usd_try_lagged` ($72,858,103) | `brent_lagged` ($47,374,901) | `rolling_mean_12` ($39,253,314) |
| `TR_RU` | `TOTAL` | $h=3$ | `usd_try_lagged` ($100,175,371) | `rolling_mean_12` ($68,901,226) | `month` ($18,059,489) |
| `TR_RU` | `TOTAL` | $h=6$ | `usd_try_lagged` ($91,885,523) | `rolling_mean_6` ($30,697,865) | `lag_6` ($27,117,650) |
| `RU_TR` | `TOTAL` | $h=1$ | `rolling_mean_3` ($314,159,426) | `usd_try_lagged` ($147,304,933) | `rolling_mean_6` ($128,232,752) |
| `RU_TR` | `TOTAL` | $h=3$ | `rolling_mean_3` ($353,469,535) | `tr_ip_lagged` ($254,155,271) | `usd_try_lagged` ($130,724,448) |
| `RU_TR` | `TOTAL` | $h=6$ | `usd_try_lagged` ($495,776,223) | `tr_ip_lagged` ($141,358,825) | `ru_ip_lagged` ($129,733,501) |
| `TR_RU` | `84` | $h=1$ | `lag_1` ($21,811,744) | `usd_try_lagged` ($12,623,495) | `rolling_mean_3` ($3,016,367) |
| `RU_TR` | `27` | $h=1$ | `usd_try_lagged` ($339,367,162) | `brent_lagged` ($105,044,051) | `tr_ip_lagged` ($73,566,620) |

### Latest Development Forecast SHAP Drivers (Local Examples)

| Horizon | Origin | Target | Direction | HS2 | Point Forecast ($) | Base Value ($) | Top Feature 1 | Top Feature 2 | Top Feature 3 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| $h=1$ | `2024-11` | `2024-12` | `TR_RU` | `TOTAL` | $735,691,456 | $485,250,720 | `usd_try_lagged` (+198,614,496) | `lag_1` (+39,214,560) | `yoy_growth` (-36,741,636) |
| $h=3$ | `2024-09` | `2024-12` | `TR_RU` | `TOTAL` | $739,756,096 | $480,970,912 | `usd_try_lagged` (+204,270,560) | `month` (+45,445,008) | `rolling_mean_12` (+30,082,178) |
| $h=6$ | `2024-06` | `2024-12` | `TR_RU` | `TOTAL` | $759,399,424 | $473,250,880 | `usd_try_lagged` (+217,792,416) | `tr_ip_lagged` (+29,586,124) | `month` (+27,956,528) |
| $h=1$ | `2024-11` | `2024-12` | `RU_TR` | `TOTAL` | $3,715,975,168 | $2,415,661,056 | `rolling_mean_3` (+598,911,680) | `tr_ip_lagged` (+235,619,712) | `rolling_mean_6` (+216,788,480) |
| $h=3$ | `2024-09` | `2024-12` | `RU_TR` | `TOTAL` | $3,626,480,128 | $2,407,390,720 | `rolling_mean_3` (+706,009,280) | `tr_ip_lagged` (+403,967,488) | `yoy_growth` (-216,071,776) |
| $h=6$ | `2024-06` | `2024-12` | `RU_TR` | `TOTAL` | $4,187,836,416 | $2,389,712,896 | `usd_try_lagged` (+1,146,104,320) | `tr_ip_lagged` (+211,761,104) | `ru_ip_lagged` (+140,634,192) |

---

## 10. Software Verification Evidence

The M9 model run produces the artifacts documented above. Unit-test and mutation-test results, including the exact command and pass count, are recorded separately in `M9_ACCEPTANCE_REPORT.md`.

---

## 11. Limitations & Methodological Disclaimers

1. **Non-Causal Interpretation**: SHAP attributions reflect statistical associations in the fitted decision trees; they do not represent economic causality.
2. **Short Early-History Tuning Blocks**: Early tuning fits operate on fewer than 60 observations due to the label-availability embargo.
3. **Prequential Uncertainty Burn-In**: Conformal prediction intervals require $n \ge 19$ historical residuals. Because residuals become available only when $s + h \le t$, the burn-in period varies by horizon: 19 unavailable origins for $h=1$ (77 available), 21 for $h=3$ (73 available), and 24 for $h=6$ (67 available) per series (totaling 896 unavailable and 3,038 enabled intervals across all 14 series).
4. **Development Run Boundary**: This run uses the frozen development feature artifact, which ends in 2024-12; this report does not establish that no earlier workflow inspected 2025 data.
5. **No Champion Selection**: Model selection decisions are strictly deferred to Milestone M10.

---

## 12. Model-Run Contract

- [x] Exactly 14 frozen target series modeled
- [x] M3 selection unchanged
- [x] M5 feature family unchanged
- [x] Exact 21 predictors whitelist enforced
- [x] XGB_h1, XGB_h3, XGB_h6 direct models (no recursion)
- [x] Exactly 3 horizon-shared parameter sets selected
- [x] Exact 16-candidate Cartesian grid evaluated
- [x] Time-aware pre-validation tuning on calendar year 2016 targets
- [x] 2017–2024 validation data strictly isolated from tuning
- [x] Critical label-availability embargo enforced: `training_period + h <= forecast_origin`
- [x] Exact frozen 281 validation folds reused
- [x] 3,934 point predictions produced (0 fit failures)
- [x] 42 metric groups produced (sMAPE, MAE, RMSE)
- [x] Prequential residual conformal uncertainty calibrated (80% and 95% intervals)
- [x] Interval invariants verified (0 violations)
- [x] Fold-local TreeSHAP computed with verified additive consistency
- [x] Global feature importance aggregated across all 42 groups
- [x] Non-causal SHAP wording strictly enforced
- [x] Methodological manifest versioned (`configs/xgboost_spec_v1_1.yaml`)
- [x] `xgboost_spec_manifest` artifact generated (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/xgboost_spec_v1_1.yaml`)
- [x] `tuning_results` artifact generated (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_tuning_results.parquet`)
- [x] `predictions` artifact generated (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_predictions.parquet`)
- [x] `metrics` artifact generated (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_metrics.parquet`)
- [x] `interval_audit` artifact generated (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_interval_audit.parquet`)
- [x] `shap_values` artifact generated (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_shap_values.parquet`)
- [x] `shap_global` artifact generated (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_shap_global.parquet`)
- [x] `fit_log` artifact generated (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_fit_log.parquet`)
- [x] Reproducibility run metadata generated (`/Users/kagansmtdms/Downloads/Проекты/tr-ru_forecast/data/processed/m9_acceptance_2026-10-14_r2/m9_xgboost_run.json`)
- [x] M10 champion selection NOT started
- [x] This run scores only the frozen development-period artifact ending 2024-12

---

## 13. Acceptance Status

This file reports model outputs. Local software acceptance and milestone readiness are tracked separately in `M9_ACCEPTANCE_REPORT.md`.

