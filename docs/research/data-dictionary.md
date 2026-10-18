# Data Dictionary

This dictionary describes the versioned v1.1 contracts and the local M1–M11 Parquet artifacts used by the research package. Raw and processed datasets under `data/` are gitignored and are not included in the public source repository. Column types below use logical meanings; physical Parquet types can be inspected from an authorized local bundle.

## Data layers

| Layer / artifact | Grain | Main purpose |
| --- | --- | --- |
| Raw trade records | Source record | Preserve reporter, partner, flow, HS classification, value, quantity and retrieval provenance from UN Comtrade |
| `trade_monthly_canonical.parquet` | Month × direction × HS2/total × classification | Standardized Türkiye-reported forecast-target history |
| `trade_monthly_mirror_RU_TR.parquet` | Month × mirror direction × HS2/total × classification | Counter-reporter observations; only used for diagnostics |
| `mirror_monthly.parquet` | Month × direction × HS2 | Paired canonical/mirror values, gaps, pair state and classification metadata |
| `macro_monthly.parquet` | Month × indicator | Processed macro observations with origin/distribution source, lag and linkage metadata |
| `model_features_development.parquet` | Forecast origin × direction × HS2 | Development-only target labels and backward-looking model features |
| `m11_holdout_predictions.parquet` | Task × forecast target × model role | Raw champion, fallback and baseline predictions plus validity and interval audit fields |
| `m11_holdout_leaderboard.parquet` | Direction × HS2 × horizon | CV/holdout task-level model, error, fallback and interval summaries |
| `m11_shap_values.parquet` | Task × origin × explained feature | Local XGBoost forecast attributions |
| `m11_shap_summary.parquet` | Task × feature | Mean signed and absolute local SHAP summaries |

The local verified trade span is 2012-01–2025-12. The development period is 2012-01–2024-12; the holdout is 2025-01–2025-12. The live calendar's later months are not part of the verified modeling dataset.

## Shared identifiers and source fields

| Field | Meaning | Type / unit | Rules |
| --- | --- | --- | --- |
| `period` | Reference month | `YYYY-MM` string | Monthly calendar period; does not itself encode publication time |
| `reporter_code`, `reporter_name` | Country submitting the observation | UN M49-like numeric identifier / label | `792` is Türkiye; `643` is Russian Federation |
| `partner_code`, `partner_name` | Counterparty named by reporter | Numeric identifier / label | Partner attribution may differ between imports and exports |
| `flow`, `flow_code`, `flow_name` | Trade flow | `X` export, `M` import | Direction is separately standardized as `TR_RU` or `RU_TR` |
| `hs_code`, `hs2` | Product classification code | String | `hs2` is a two-digit chapter; preserve leading zeros and HS revision |
| `classification`, `classification_code` | HS nomenclature/revision | String | H4, H5 and H6 metadata are retained; no silent cross-revision relabeling |
| `trade_value_usd`, `canonical_value_usd`, `mirror_value_usd` | Reported trade value | USD, floating-point source representation | Current-dollar source measure; no deflation in v1.1 |
| `net_weight_kg` | Net weight | kilograms | Nullable when not reported |
| `quantity`, `quantity_unit` | Supplementary quantity and unit | Source-defined | Interpret only together with unit; no cross-unit summation |
| `source`, `source_record_id_or_key` | Source identity and source record key | String | Preserve source-level provenance; no source is silently substituted |
| `retrieved_at` | Local retrieval timestamp | ISO timestamp string | Records retrieval time, not original publication time or historical vintage |
| `data_status` | Ingestion/quality state | Categorical string | Must distinguish unavailable, reported zero and observed non-zero |
| `is_total` | Whether row is bilateral aggregate | Boolean | `true` for total, `false` for HS2 detail |

## Standardized trade and mirror fields

| Field | Definition |
| --- | --- |
| `direction` | `TR_RU` for Türkiye exports to Russia; `RU_TR` for Türkiye imports from Russia |
| `hs_description` | Product label; not a stable key across classification versions |
| `mirror_gap_usd` | `canonical_value_usd - mirror_value_usd` |
| `mirror_gap_pct` | `(canonical - mirror) / canonical × 100`; null when canonical is zero or the pair is unavailable |
| `abs_mirror_gap_pct` | Absolute relative difference under the stored mirror contract |
| `canonical_missing`, `mirror_missing` | Side-specific availability flags; missing never means zero |
| `pair_status` | Pairing/availability state for the canonical and mirror observations |
| `classification_mismatch` | Indicates reporter-side classification metadata disagreement |
| `canonical_retrieved_at`, `mirror_retrieved_at` | Retrieval timestamps for each paired record |
| `gap_flag` | Diagnostic threshold/category flag; not a data-cleaning instruction |

Mirror values do not fill a missing canonical value, and mirror gaps are not estimates of freight cost or causal effects.

## Macroeconomic fields

| Field | Meaning |
| --- | --- |
| `indicator_id`, `indicator_name` | Stable series key and human-readable indicator |
| `value`, `unit` | Numeric observation and canonical unit |
| `origin_provider` | Institution that originates/compiles the source series |
| `distribution_provider` | Platform through which the series was retrieved |
| `underlying_national_source` | National source for a composite/redistributed series, if present |
| `source_provider`, `source_series_id` | Configured citation/source label and series identifier |
| `source_frequency`, `aggregation_method` | Native frequency and monthly normalization rule |
| `reference_period` | Period measured by the observation |
| `availability_date_or_approximation` | Configured deterministic date used by the as-of feature accessor |
| `release_lag_rule`, `availability_is_exact` | Release-lag description and whether the date is exact or approximate |
| `linkage_method`, `linkage_factor`, `linkage_version` | Provenance for the linked Russia industrial-production composite; null for ordinary series |
| `retrieved_at`, `data_status` | Local retrieval time and processing state |

The five indicators are `usd_try` (TRY per USD), `rub_usd` (RUB per USD), `brent` (USD/barrel), `tr_ip` (seasonally adjusted index, 2021=100), and `ru_ip` (seasonally adjusted linked composite on the 2015=100 scale). Configured availability is approximate-conservative: one month for USD/TRY, RUB/USD and Brent; two months for industrial production. It is not a record of the exact historical publication timestamp.

## Development feature fields

`model_features_development.parquet` stores 2,184 rows (14 series × 156 development months). `period` is the forecast origin month; `feature_snapshot_date` is that month's calendar end. `target` is observed monthly trade value at the origin. `target_h1`, `target_h3`, and `target_h6` are future supervised labels at t+1, t+3 and t+6; they are never predictors.

| Feature | Definition / unit |
| --- | --- |
| `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12` | Within-series target at the named prior month offset; USD |
| `rolling_mean_3`, `_6`, `_12` | Mean of the complete prior 3/6/12 months, computed after shifting target by one month; USD |
| `rolling_std_3`, `_6`, `_12` | Sample standard deviation (`ddof=1`) of the same backward-only windows; USD |
| `mom_growth`, `yoy_growth` | `(y_t / y_(t-1)) - 1` and `(y_t / y_(t-12)) - 1`; null when denominator unavailable or zero |
| `month`, `quarter`, `year` | Calendar values derived from `period` |
| `usd_try_lagged`, `rub_usd_lagged`, `brent_lagged`, `tr_ip_lagged`, `ru_ip_lagged` | Macro values selected only when configured approximate availability is on or before `feature_snapshot_date` |

No forward fill, backward fill, interpolation, mean/median imputation, or missing-to-zero conversion is permitted by the feature contract. The 2025 target holdout is excluded before development feature and label construction.

## Holdout prediction and leaderboard fields

The full physical schemas are maintained by the Parquet artifacts and M11 reports. The primary prediction columns are:

| Field | Meaning |
| --- | --- |
| `sector_id`, `direction`, `hs2`, `horizon` | Frozen task key |
| `forecast_origin`, `origin_date`, `forecast_period` | Origin month/date and target month |
| `role`, `model_family` | `champion`, `fallback` or `baseline` role and fitted model family |
| `actual`, `prediction`, `operational_prediction` | Target, raw role forecast and governed operational forecast; USD |
| `fit_status`, `converged`, `is_valid_prediction`, `failure_message` | Fitting and forecast validity audit |
| `optimizer_used`, `retry_count`, `warning_type` | Classical fitting audit fields |
| `train_start`, `train_end` | Fitting sample bounds |
| `calibration_count`, `max_calibration_target_period` | Prequential interval calibration pool metadata |
| `has_operational_fallback`, `operational_model_family` | Whether fallback supplied the governed value and which family was used |
| `pre_freeze_origin` | Identifies reconstructed early multi-step origins before the selection freeze date |
| `feature_count` | Number of features supplied to the fitted task model |
| `lower_80`, `upper_80`, `lower_95`, `upper_95` | Raw model interval bounds; USD; may be unavailable |
| `operational_lower_80`, `operational_upper_80`, `operational_lower_95`, `operational_upper_95` | Bounds associated with the operational forecast where available |
| `residual`, `absolute_error`, `squared_error`, `smape_component` | Derived error fields under the M11 scoring convention |
| `interval_available`, `covered_80`, `covered_95` | Interval availability and target inclusion flags |

The task-level generated CSV keeps raw champion and governed operational sMAPE separate and includes the denominator for each task. `operational_mean_smape` and coverage are percentages when displayed in prose; the CSV stores coverage as a fraction from 0 to 1 and error metrics in percentage points where appropriate. `operational_mean_mae` is in USD. `skill_vs_naive_pct` is a percent, not a fraction.

## SHAP fields

| Field | Meaning |
| --- | --- |
| `feature` | Model input receiving an attribution |
| `feature_value` | Feature value presented to the model for the explained row; feature-specific units |
| `shap_value` | Signed local attribution for a specific fitted forecast; model-output units |
| `base_value` | SHAP reference/base output for the fitted explanation |
| `point_prediction` | Forecast output being decomposed by the local attribution row; USD |
| `mean_abs_shap` | Mean absolute attribution within the reported group |
| `mean_signed_shap` | Mean signed attribution within the reported group |
| `n_origins` | Number of forecast origins represented in the summary |

SHAP is a predictive decomposition of model outputs. It does not identify causal or intervention effects.

## Status and null semantics

`null`/NaN means unavailable, not zero. A finite negative point forecast is preserved as raw model evidence and may be marked invalid for operational selection. Missing intervals stay unavailable. Failure and fallback columns are part of the result rather than hidden during aggregation.
