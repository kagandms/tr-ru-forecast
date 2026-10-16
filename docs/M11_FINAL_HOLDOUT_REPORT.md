# M11 Final Holdout Evaluation Report

**Generated:** 2026-10-14T17:25:26.594619+00:00
**Run ID:** `m11_40c224da2433`
**Status:** Final retrospective holdout evaluation; assignments unchanged.

## 1. Executive Summary & Acceptance Scope

**Retrospective latest-vintage holdout evaluation under deterministic lag rules.** All 42 tasks cover twelve 2025 targets with origins target-month minus h1/h3/h6, giving 504 task/origin cells and 1,512 role rows. Frozen champion/fallback identities, orders, parameters, feature families and interval levels are reused without tuning. Trade values are assumed available at reference-month end (zero added lag), consistent with the authorized retrospective convention. Latest-vintage macro values are joined as-of each historical feature-row month-end using the frozen approximate release dates. Historical publication timestamps, original vintages and real-time actionable performance are not established.

Early h3/h6 origins precede the 2024-12 model-selection freeze and are retrospective reconstructions using subsequently frozen assignments; they are not ex-ante deployments of those assignments. Rolling holdout refits and chronological residual updates were amended explicitly before scoring.

XGBoost calibration admits only same-series/horizon/model out-of-sample residuals with target <= origin; later CV and holdout residuals are admitted as their dates become eligible. Minimum count 19 is checked. Native classical intervals and prequential conformal intervals are assessed separately; Seasonal Naive intervals are unavailable.

Invalid/negative/non-converged forecasts are retained raw, with failure logs and valid-origin denominators. Operational fallback uses only the pre-registered fallback; failed fallbacks leave the served prediction unavailable. Primary metrics exclude invalid forecasts but raw values/errors remain in the prediction artifact. Coverage is conditional on valid available intervals, with interval-origin counts. No retrospective champion changes or interval tuning occurs.

## 2. Global Champion / Fallback / Baseline Performance

| horizon | role | n_expected_origins | n_valid_origins | n_failed_origins | mean_smape | std_smape | median_smape | max_fold_smape | mean_mae | mean_rmse | n_finite_raw_origins | raw_mean_smape | n_valid_operational_origins | operational_mean_smape | operational_mean_mae | operational_mean_rmse | n_interval_origins | empirical_coverage_80 | empirical_coverage_95 | mean_interval_width_80 | mean_interval_width_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | baseline | 168 | 168 | 0 | 37.4965 | 28.2114 | 34.4981 | 128.1668 | 91096153.6012 | 190508878.6810 | 168 | 37.4965 | 168 | 37.4965 | 91096153.6012 | 190508878.6810 | 0 | unavailable | unavailable | unavailable | unavailable |
| 1 | champion | 168 | 168 | 0 | 21.1753 | 20.4332 | 16.0678 | 112.6639 | 57973906.2087 | 133687303.1605 | 168 | 21.1753 | 168 | 21.1753 | 57973906.2087 | 133687303.1605 | 168 | 0.8810 | 0.9940 | 296006795.9442 | 465681780.4717 |
| 1 | fallback | 168 | 168 | 0 | 19.5664 | 19.7420 | 13.4307 | 106.1218 | 62147958.3938 | 142398435.7078 | 168 | 19.5664 | 168 | 19.5664 | 62147958.3938 | 142398435.7078 | 156 | 0.8718 | 0.9872 | 266493609.4241 | 519916836.7201 |
| 3 | baseline | 168 | 168 | 0 | 37.4965 | 28.2114 | 34.4981 | 128.1668 | 91096153.6012 | 190508878.6810 | 168 | 37.4965 | 168 | 37.4965 | 91096153.6012 | 190508878.6810 | 0 | unavailable | unavailable | unavailable | unavailable |
| 3 | champion | 168 | 167 | 1 | 25.3519 | 22.7975 | 17.6329 | 107.5176 | 74844472.1652 | 176297734.4673 | 168 | 26.3915 | 168 | 25.2014 | 74399252.2118 | 175772255.7376 | 167 | 0.8862 | 0.9880 | 398414805.6379 | 612758611.4483 |
| 3 | fallback | 168 | 168 | 0 | 27.1295 | 24.0143 | 19.6845 | 113.6958 | 80596499.8004 | 183908581.6704 | 168 | 27.1295 | 168 | 27.1295 | 80596499.8004 | 183908581.6704 | 156 | 0.8462 | 0.9936 | 384949353.1311 | 729470580.4999 |
| 6 | baseline | 168 | 168 | 0 | 37.4965 | 28.2114 | 34.4981 | 128.1668 | 91096153.6012 | 190508878.6810 | 168 | 37.4965 | 168 | 37.4965 | 91096153.6012 | 190508878.6810 | 0 | unavailable | unavailable | unavailable | unavailable |
| 6 | champion | 168 | 168 | 0 | 32.1346 | 26.3850 | 25.7108 | 112.6860 | 86402863.3312 | 175310216.5544 | 168 | 32.1346 | 168 | 32.1346 | 86402863.3312 | 175310216.5544 | 168 | 0.8393 | 0.9821 | 489433899.1680 | 826217234.3505 |
| 6 | fallback | 168 | 167 | 1 | 30.4896 | 29.9940 | 20.2594 | 182.9186 | 85572187.2687 | 177170959.2407 | 168 | 31.4986 | 167 | 30.4896 | 85572187.2687 | 177170959.2407 | 155 | 0.9097 | 0.9677 | 516751148.7399 | 924711394.4270 |
| overall | baseline | 504 | 504 | 0 | 37.4965 | 28.2114 | 34.4981 | 128.1668 | 91096153.6012 | 190508878.6810 | 504 | 37.4965 | 504 | 37.4965 | 91096153.6012 | 190508878.6810 | 0 | unavailable | unavailable | unavailable | unavailable |
| overall | champion | 504 | 503 | 1 | 26.2223 | 23.7688 | 18.1275 | 112.6860 | 73070226.9071 | 162951841.8654 | 504 | 26.5671 | 504 | 26.1704 | 72925340.5839 | 162790103.0384 | 503 | 0.8688 | 0.9881 | 394610952.9232 | 634929866.0080 |
| overall | fallback | 504 | 503 | 1 | 25.7190 | 25.3443 | 17.9499 | 182.9186 | 76086728.1322 | 168791759.6732 | 504 | 26.0648 | 503 | 25.7190 | 76086728.1322 | 168791759.6732 | 467 | 0.8758 | 0.9829 | 389125332.3625 | 724271313.1103 |

## 3. Empirical Coverage by Role and Family

| role | model_family | n_expected_origins | n_valid_origins | n_failed_origins | mean_smape | std_smape | median_smape | max_fold_smape | mean_mae | mean_rmse | n_finite_raw_origins | raw_mean_smape | n_valid_operational_origins | operational_mean_smape | operational_mean_mae | operational_mean_rmse | n_interval_origins | empirical_coverage_80 | empirical_coverage_95 | mean_interval_width_80 | mean_interval_width_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | seasonal_naive | 504.0000 | 504.0000 | 0.0000 | 37.4965 | 28.2114 | 34.4981 | 128.1668 | 91096153.6012 | 190508878.6810 | 504.0000 | 37.4965 | 504.0000 | 37.4965 | 91096153.6012 | 190508878.6810 | 0.0000 | unavailable | unavailable | unavailable | unavailable |
| champion | sarima | 396.0000 | 395.0000 | 1.0000 | 22.9590 | 20.8543 | 17.3833 | 112.6639 | 80672818.5678 | 180353139.1655 | 396.0000 | 23.4061 | 396.0000 | 22.9012 | 80469219.3290 | 180125276.6277 | 395.0000 | 0.9215 | 0.9924 | 467826323.3883 | 715478619.4430 |
| champion | xgboost | 108.0000 | 108.0000 | 0.0000 | 38.1576 | 29.3124 | 32.1651 | 112.6860 | 45264451.8519 | 68586633.7306 | 108.0000 | 38.1576 | 108.0000 | 38.1576 | 45264451.8519 | 68586633.7306 | 108.0000 | 0.6759 | 0.9722 | 126832514.6481 | 340330258.5370 |
| fallback | sarima | 108.0000 | 107.0000 | 1.0000 | 32.2345 | 33.5320 | 17.7348 | 182.9186 | 37086732.6480 | 59937168.7172 | 108.0000 | 33.7879 | 107.0000 | 32.2345 | 37086732.6480 | 59937168.7172 | 107.0000 | 0.9065 | 0.9907 | 174752169.1432 | 267260379.4881 |
| fallback | sarimax | 36.0000 | 36.0000 | 0.0000 | 27.4764 | 17.2124 | 24.1328 | 73.9434 | 17242602.2824 | 28586145.2562 | 36.0000 | 27.4764 | 36.0000 | 27.4764 | 17242602.2824 | 28586145.2562 | 36.0000 | 0.8611 | 0.9722 | 58076424.0266 | 88820225.8133 |
| fallback | seasonal_naive | 36.0000 | 36.0000 | 0.0000 | 29.6412 | 24.5643 | 27.0592 | 102.7673 | 22156697.1111 | 27356981.7235 | 36.0000 | 29.6412 | 36.0000 | 29.6412 | 22156697.1111 | 27356981.7235 | 0.0000 | unavailable | unavailable | unavailable | unavailable |
| fallback | xgboost | 324.0000 | 324.0000 | 0.0000 | 22.9362 | 22.4076 | 16.6095 | 113.6958 | 101496818.1451 | 207051961.4441 | 324.0000 | 22.9362 | 324.0000 | 22.9362 | 101496818.1451 | 207051961.4441 | 324.0000 | 0.8673 | 0.9815 | 496704619.9074 | 945803439.7778 |

## 4. Complete 42-Task CV-to-Holdout Matrix

| sector_id | horizon | champion_model | fallback_model | mean_smape | cv_mean_smape | cv_vs_holdout_smape_drift | baseline_mean_smape | skill_vs_naive_pct | n_valid_origins | n_operational_fallbacks | empirical_coverage_80 | empirical_coverage_95 | n_interval_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RU_TR:10 | 1 | sarima | xgboost | 40.5933 | 33.4924 | 7.1009 | 75.5455 | 46.2664 | 12 | 0 | 1.0000 | 1.0000 | 12 |
| RU_TR:10 | 3 | sarima | xgboost | 53.9176 | 41.8938 | 12.0237 | 75.5455 | 33.7771 | 11 | 1 | 0.9091 | 1.0000 | 11 |
| RU_TR:10 | 6 | xgboost | sarima | 65.8765 | 42.5731 | 23.3034 | 75.5455 | 12.7989 | 12 | 0 | 0.8333 | 1.0000 | 12 |
| RU_TR:23 | 1 | sarima | xgboost | 12.9620 | 22.5869 | -9.6248 | 18.1776 | 28.6922 | 12 | 0 | 0.9167 | 1.0000 | 12 |
| RU_TR:23 | 3 | sarima | xgboost | 13.5669 | 25.2322 | -11.6653 | 18.1776 | 25.3647 | 12 | 0 | 0.9167 | 1.0000 | 12 |
| RU_TR:23 | 6 | sarima | xgboost | 15.2596 | 25.5492 | -10.2896 | 18.1776 | 16.0522 | 12 | 0 | 0.9167 | 1.0000 | 12 |
| RU_TR:27 | 1 | sarima | xgboost | 10.6985 | 18.9680 | -8.2696 | 14.6357 | 26.9017 | 12 | 0 | 0.9167 | 1.0000 | 12 |
| RU_TR:27 | 3 | sarima | xgboost | 13.9399 | 27.4649 | -13.5250 | 14.6357 | 4.7547 | 12 | 0 | 1.0000 | 1.0000 | 12 |
| RU_TR:27 | 6 | sarima | xgboost | 15.3455 | 35.9870 | -20.6415 | 14.6357 | -4.8498 | 12 | 0 | 1.0000 | 1.0000 | 12 |
| RU_TR:72 | 1 | sarima | xgboost | 14.3860 | 23.3096 | -8.9236 | 30.4493 | 52.7544 | 12 | 0 | 1.0000 | 1.0000 | 12 |
| RU_TR:72 | 3 | sarima | xgboost | 19.1357 | 26.8743 | -7.7385 | 30.4493 | 37.1555 | 12 | 0 | 0.9167 | 1.0000 | 12 |
| RU_TR:72 | 6 | sarima | xgboost | 22.5240 | 32.0466 | -9.5227 | 30.4493 | 26.0280 | 12 | 0 | 1.0000 | 1.0000 | 12 |
| RU_TR:74 | 1 | sarima | seasonal_naive | 25.5693 | 46.6870 | -21.1177 | 32.2002 | 20.5928 | 12 | 0 | 0.9167 | 1.0000 | 12 |
| RU_TR:74 | 3 | sarima | xgboost | 23.1128 | 50.1042 | -26.9913 | 32.2002 | 28.2215 | 12 | 0 | 0.9167 | 1.0000 | 12 |
| RU_TR:74 | 6 | xgboost | sarima | 45.1320 | 53.2161 | -8.0842 | 32.2002 | -40.1605 | 12 | 0 | 0.6667 | 1.0000 | 12 |
| RU_TR:76 | 1 | sarima | xgboost | 45.4531 | 27.5196 | 17.9335 | 47.7056 | 4.7217 | 12 | 0 | 0.7500 | 0.9167 | 12 |
| RU_TR:76 | 3 | xgboost | sarima | 53.4102 | 29.8757 | 23.5345 | 47.7056 | -11.9579 | 12 | 0 | 0.4167 | 0.9167 | 12 |
| RU_TR:76 | 6 | sarima | xgboost | 38.9083 | 38.1443 | 0.7640 | 47.7056 | 18.4409 | 12 | 0 | 0.9167 | 0.9167 | 12 |
| RU_TR:TOTAL | 1 | sarima | xgboost | 7.9459 | 14.0738 | -6.1279 | 10.3079 | 22.9149 | 12 | 0 | 1.0000 | 1.0000 | 12 |
| RU_TR:TOTAL | 3 | sarima | xgboost | 9.1780 | 19.1725 | -9.9946 | 10.3079 | 10.9619 | 12 | 0 | 1.0000 | 1.0000 | 12 |
| RU_TR:TOTAL | 6 | sarima | xgboost | 10.4518 | 25.7697 | -15.3179 | 10.3079 | -1.3959 | 12 | 0 | 1.0000 | 1.0000 | 12 |
| TR_RU:08 | 1 | sarima | xgboost | 31.3628 | 22.1981 | 9.1647 | 28.3617 | -10.5815 | 12 | 0 | 0.8333 | 1.0000 | 12 |
| TR_RU:08 | 3 | sarima | seasonal_naive | 36.0090 | 25.3345 | 10.6745 | 28.3617 | -26.9633 | 12 | 0 | 0.6667 | 1.0000 | 12 |
| TR_RU:08 | 6 | sarima | seasonal_naive | 22.8424 | 27.0907 | -4.2483 | 28.3617 | 19.4606 | 12 | 0 | 0.9167 | 1.0000 | 12 |
| TR_RU:39 | 1 | sarima | xgboost | 12.7047 | 15.9420 | -3.2373 | 21.4303 | 40.7161 | 12 | 0 | 0.6667 | 1.0000 | 12 |
| TR_RU:39 | 3 | sarima | sarimax | 15.8140 | 21.5827 | -5.7687 | 21.4303 | 26.2074 | 12 | 0 | 0.8333 | 1.0000 | 12 |
| TR_RU:39 | 6 | sarima | xgboost | 19.8459 | 27.2760 | -7.4301 | 21.4303 | 7.3932 | 12 | 0 | 0.8333 | 1.0000 | 12 |
| TR_RU:73 | 1 | xgboost | sarima | 21.3624 | 17.4940 | 3.8684 | 63.2431 | 66.2218 | 12 | 0 | 0.8333 | 1.0000 | 12 |
| TR_RU:73 | 3 | sarima | sarimax | 31.7406 | 22.2317 | 9.5090 | 63.2431 | 49.8117 | 12 | 0 | 1.0000 | 1.0000 | 12 |
| TR_RU:73 | 6 | xgboost | sarima | 62.8265 | 23.2315 | 39.5949 | 63.2431 | 0.6588 | 12 | 0 | 0.1667 | 0.8333 | 12 |
| TR_RU:84 | 1 | xgboost | sarima | 19.7118 | 16.9960 | 2.7158 | 43.0908 | 54.2553 | 12 | 0 | 0.5833 | 1.0000 | 12 |
| TR_RU:84 | 3 | sarima | xgboost | 17.9414 | 20.4252 | -2.4838 | 43.0908 | 58.3637 | 12 | 0 | 0.9167 | 1.0000 | 12 |
| TR_RU:84 | 6 | sarima | sarimax | 27.4546 | 25.4771 | 1.9775 | 43.0908 | 36.2866 | 12 | 0 | 0.9167 | 1.0000 | 12 |
| TR_RU:85 | 1 | sarima | xgboost | 21.4457 | 22.1856 | -0.7399 | 46.4010 | 53.7819 | 12 | 0 | 1.0000 | 1.0000 | 12 |
| TR_RU:85 | 3 | sarima | xgboost | 26.9007 | 26.9408 | -0.0401 | 46.4010 | 42.0257 | 12 | 0 | 1.0000 | 1.0000 | 12 |
| TR_RU:85 | 6 | xgboost | sarima | 44.0980 | 31.9375 | 12.1605 | 46.4010 | 4.9634 | 12 | 0 | 0.8333 | 1.0000 | 12 |
| TR_RU:87 | 1 | sarima | xgboost | 20.4605 | 21.0499 | -0.5894 | 68.1596 | 69.9815 | 12 | 0 | 1.0000 | 1.0000 | 12 |
| TR_RU:87 | 3 | sarima | xgboost | 25.0994 | 25.4578 | -0.3584 | 68.1596 | 63.1755 | 12 | 0 | 1.0000 | 1.0000 | 12 |
| TR_RU:87 | 6 | sarima | xgboost | 40.1162 | 30.0064 | 10.1098 | 68.1596 | 41.1436 | 12 | 0 | 0.9167 | 1.0000 | 12 |
| TR_RU:TOTAL | 1 | xgboost | sarima | 11.7979 | 14.8326 | -3.0348 | 25.2420 | 53.2609 | 12 | 0 | 0.9167 | 1.0000 | 12 |
| TR_RU:TOTAL | 3 | sarima | xgboost | 17.5410 | 21.7855 | -4.2445 | 25.2420 | 30.5088 | 12 | 0 | 0.9167 | 0.9167 | 12 |
| TR_RU:TOTAL | 6 | xgboost | sarima | 19.2031 | 25.4387 | -6.2356 | 25.2420 | 23.9240 | 12 | 0 | 0.8333 | 1.0000 | 12 |

## 5. Generalization, Drift and Economic Interpretation

Champions have lower mean valid-origin sMAPE than the full Seasonal Naive benchmark in 36/42 tasks. Mean CV-to-holdout drift is -0.9955 percentage points; median is -3.1360. The matrix retains deteriorations and high-error cases. Different valid-origin counts must be considered before comparing failed models. Stored skill uses paired valid primary origins; raw complete-origin errors and operational fallback forecasts remain separately accessible.

Differences across fruit, energy, metals, machinery and TOTAL reflect predictive heterogeneity; they do not identify causal macro effects. XGBoost uses trade history/calendar alongside macro features. SHAP is a forecast-driver decomposition, not economic causality. Native and conformal empirical coverage can differ from nominal 80%/95%; no calibration parameters were adjusted after these outcomes.

## 6. Reproducibility and Provenance

Git commit: `c7c8edc9e750d49d6de8aad676f1553f51c22387`. Dirty source tree: `True`. Source fingerprint: `f66981621d8dc854a5ba7321524ada30341a0c445617893f8b22ee41239faf4d`. Input and output SHA-256 values are in `m11_holdout_run.json`. Commands: `make m11-holdout`; independent reproduction: `make m11-holdout M11_OUTPUT_DIR=/tmp/m11-repro`. Outputs never overwrite existing files. Data hashes and forecasts must reproduce; timestamps/Git state may differ. Exact software-test results are recorded separately in `M11_ACCEPTANCE_REPORT.md`, not asserted by this generator.
