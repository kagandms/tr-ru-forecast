# Monthly Türkiye–Russia Bilateral Trade Forecasting with Mirror-Data Diagnostics

**Technical working paper — Specification v1.1**

**Evaluation window:** development 2012-01–2024-12; holdout 2025-01–2025-12

**Evidence cut:** M10 champion registry and M11 accepted holdout package, verified 2026-10-16
**Status:** Technical report; not peer reviewed

## 1. Abstract

This study evaluates monthly forecasts for selected Türkiye–Russia merchandise-trade flows and separately examines reporter mirror discrepancies. Four model families—Seasonal Naive, SARIMA, SARIMAX and direct-horizon XGBoost—were compared on 14 Türkiye-reported series, comprising two bilateral totals and six HS2 chapters in each direction. Champion assignments were selected using development-only expanding-window cross-validation and frozen before the 2025 holdout was scored. The frozen registry contains 33 SARIMA and 9 XGBoost assignments. Across 504 task-origin cells, the governed operational champion/fallback forecasts achieve a mean sMAPE of 26.1704%, compared with 37.4965% for Seasonal Naive. Thirty-six of 42 tasks have lower task-level mean valid-origin sMAPE than the full Seasonal Naive benchmark. Aggregate valid-interval coverage is 86.8787% for nominal 80% intervals and 98.8072% for nominal 95% intervals; XGBoost champions' nominal 80% coverage is lower, at 67.5926%. These are descriptive results from one 12-month holdout. Evaluation uses retrospective latest-vintage data and deterministic availability-lag rules, not historical publication-time vintages. The findings support heterogeneous predictive performance, not causal claims about macroeconomic drivers or real-time availability.

## 2. Introduction

Monthly bilateral trade values are difficult forecasting targets: they combine sector-specific seasonality, large shocks, changing trade composition and measurement differences between reporters. A useful system must compare model families against a simple seasonal benchmark, preserve missing and invalid outcomes, and keep model selection separate from final evaluation.

This project asks three questions. First, how do frozen statistical and machine-learning champions perform against Seasonal Naive across directions, HS2 chapters and 1-, 3- and 6-month horizons? Second, how often do the selected champions beat the baseline at the task level? Third, what do paired reporter records reveal about data coverage and mirror gaps, without treating one reporter as a silent replacement for the other?

The contribution is an auditable monthly workflow joining source and feature contracts, rolling-origin model selection, a frozen 2025 holdout, uncertainty diagnostics, local XGBoost attributions, and a read-only artifact-serving interface. It is not a structural trade model and does not estimate the causal effect of exchange rates, industrial production, oil prices, sanctions or policy changes.

## 3. Bilateral Trade Background

The two forecasting targets are reported by Türkiye: exports to Russia (`TR_RU`, reporter 792, partner 643, flow X) and imports from Russia (`RU_TR`, reporter 792, partner 643, flow M). UN Comtrade distributes reporter-submitted trade records. The forecast target is therefore a clearly defined reporting series, not a reconciled “true” bilateral value.

Mirror records are the corresponding partner-reported flows where present. Export and import reports can differ because of partner attribution, timing, valuation, trade-system coverage, transshipment and classification practices. CIF/FOB valuation can contribute to a difference, but a measured mirror gap is not a freight-cost estimate. The project's mirror layer is diagnostic: it does not impute, average or overwrite the frozen Türkiye-reported target.

The verified bundle has paired Russian-reporter mirror observations for the relevant RU→TR comparison through 2021-12, with no paired observations after that date. This is reported as a coverage boundary. The project does not infer a political, administrative or economic cause for the missing pairs.

## 4. Literature Review

International merchandise-trade standards emphasize reporter definitions, partner attribution, valuation, trade systems and data-quality metadata. The UN's IMTS 2010 recommendations and UN Comtrade's guidance on bilateral asymmetries explain why partner-reported exports and imports need not match. These sources support treating mirror comparisons as data diagnostics rather than assuming one series is the truth for both reporters.

Forecast evaluation must respect temporal order. Rolling-origin evaluation repeatedly fits or evaluates a model using only observations available before each forecast origin; this directly reflects a forecasting task and avoids random shuffling across time. Bergmeir, Hyndman and Koo discuss conditions under which ordinary K-fold CV can be valid for certain autoregressive prediction settings, but this project follows its frozen expanding-origin protocol rather than relying on those conditions. The M4 competition provides broad empirical context for comparing statistical and machine-learning forecasting methods across many series; its aggregate rankings do not predict results for this narrower bilateral-trade sample.

SHAP provides additive feature attributions for individual model predictions. Those values describe the fitted model's allocation of prediction contributions under the chosen feature representation. They do not identify causal effects, and correlated or jointly constructed time-series features can make economic interpretation especially delicate. Accordingly, this study labels SHAP as predictive explanation only.

## 5. Data

The modeling dataset contains 14 monthly series from 2012-01 through 2025-12: a bilateral total plus six HS2 chapters for each direction. The selected TR→RU chapters are 08, 84, 39, 87, 73 and 85; RU→TR chapters are 27, 72, 10, 76, 23 and 74. Monetary values are current USD as supplied in the source; they are not deflated in the frozen v1.1 target contract.

Development covers 156 months, 2012-01–2024-12. The final holdout is the twelve months 2025-01–2025-12. Feature construction uses target lags, backward-only rolling summaries, growth rates, calendar fields and five configured macro indicators: USD/TRY, RUB/USD, Brent, Türkiye industrial production and a linked Russia industrial production index. HS2 classifications span H4, H5 and H6 regimes; the project retains the classification metadata and models at chapter level.

Macro observations are latest-vintage local processed data. The feature contract uses configured approximate conservative lag rules and a forecast-origin cutoff. These rules provide a deterministic retrospective information screen; they do not recreate the exact value published on every historical decision date. The source and rights details are recorded separately in the [sources register](sources-and-licensing.md).

## 6. Mirror Statistics and Data Quality

The data contract distinguishes reported zero, missing/unavailable and non-zero observation. Missing values are not converted to zero, interpolated or replaced by mirror data. Duplicate resolution is deterministic and classification identifiers are retained. Extreme values are flagged for investigation and preserved absent evidence of an error.

The canonical target is Türkiye-reported in both directions. Mirror records are paired by month, direction and HS2 where both reporters provide observations. The signed gap is canonical minus mirror; percentage gap is undefined when the canonical value is zero. A gap can reflect multiple compilation and valuation differences, so it is not interpreted as a causal or freight-only adjustment.

The project's verified UN Comtrade bundle has no Russian-reporter RU→TR records after 2021-12. Independently acquired TurkStat and Rosstat bilateral trade datasets are not present in the bundle. The word “mirror” therefore refers to paired reporter records distributed by UN Comtrade, not a separate national-source replication exercise.

## 7. Methodology

The frozen baseline is Seasonal Naive, forecasting month `t+h` with the value at `t+h−12` when available. Statistical families are SARIMA and SARIMAX. XGBoost uses direct models for horizons 1, 3 and 6 months with lag, calendar and publication-safe macro features. Future realized macro observations are forbidden as forecast inputs.

Development validation uses expanding rolling origins with a minimum 60-month training history and a frozen calendar of 281 origins across horizons: 96 for h1, 94 for h3 and 91 for h6. Primary metric is mean sMAPE, MAE breaks ties, with diagnostic RMSE, dispersion, worst-origin behavior, fit validity and complexity recorded. M10 chooses the simplest eligible model whose relative sMAPE loss against the raw winner is strictly below 2%, using the simpler model's score as denominator. This tolerance is a deterministic parsimony rule, not a statistical-equivalence test.

## 8. Statistical Models

Seasonal Naive is the common reference model. SARIMA is a univariate seasonal autoregressive integrated moving-average model with orders frozen by the development configuration. SARIMAX uses the frozen SARIMA order and origin-known macro paths under the project's as-of accessor. It does not receive future realized macro values. The configuration and earlier milestone reports specify fitting and interval details.

M10 compared all four candidate families using the permitted development artifacts and froze one champion and fallback for each direction × HS2 × horizon task. The accepted matrix contains 33 SARIMA and 9 XGBoost champions; SARIMAX has no champion assignments. Failed or negative forecasts remain visible as eligibility or outcome evidence. They are not silently clipped or removed to improve the reported metric.

## 9. Machine Learning

XGBoost is evaluated as a direct multi-horizon family, not as an autoregressive recursive forecast. Its inputs include within-series lags, backward-only rolling statistics, growth and calendar variables, plus macro variables admitted under the deterministic lag/origin cutoff. Since XGBoost also uses trade history and calendar inputs, a win by XGBoost does not isolate the incremental contribution of macro data.

M11 stores 9,072 local SHAP attributions across 432 unique XGBoost fits and a 756-row task/feature summary. These describe how the fitted trees allocated individual forecast outputs. No macro ablation was conducted, so the report makes no claim that macro covariates improve performance or have causal effects.

## 10. Experimental Design

Champion selection is performed on development CV records only. After the M10 assignment registry was frozen, M11 evaluated twelve 2025 targets for each of 42 tasks, yielding 504 task-origin cells and 1,512 champion/fallback/baseline role rows. Model assignment, feature selection, family, parameters and interval levels were not retuned on holdout outcomes.

The evaluation scope is **retrospective latest-vintage holdout evaluation under deterministic lag rules**. Trade target observations are treated as available at reference-month end with zero additional trade lag; macro data follow the frozen approximate availability rules. No historical `published_at`/`available_at` archive or macro vintage snapshots were used. Early h3/h6 origins before the 2024-12 selection freeze are retrospective reconstructions using the subsequently frozen assignment. This design is useful for a governed historical comparison but does not prove decision-time feasibility.

Operational scores preserve the registered fallback behavior. A finite negative SARIMA champion output remains in raw champion scoring but is invalid operationally; the pre-registered XGBoost fallback supplies the governed output for that task-origin. Raw champion, fallback benchmark and operational metrics remain separate in the persisted artifacts.

## 11. Results

Across all 504 governed operational cells, mean sMAPE is 26.1704%. Seasonal Naive has 504 valid cells and mean sMAPE 37.4965%. The relative aggregate reduction is approximately 30.2%: `(37.4965 − 26.1704) / 37.4965`. This descriptive aggregate weights forecast cells equally and is not a guarantee for a future period or any individual sector. The raw champion-only score is 26.5671% over finite raw champion values, while the fallback-only benchmark is 25.7190%; the fallback benchmark does not retroactively alter M10 assignments.

Thirty-six of 42 champions have lower task-level mean valid-origin sMAPE than the full Seasonal Naive benchmark. One primary champion origin failed due to an invalid negative forecast, and governed fallback behavior restored a valid operational forecast for the cell. Sector and horizon results vary materially; see the complete [42-task CSV](results/holdout-task-results.csv) and the source M11 report for denominators and diagnostic fields.

For valid intervals, aggregate empirical coverage is 86.8787% at nominal 80% and 98.8072% at nominal 95%, using 503 interval origins. This aggregate masks model-family differences: XGBoost champions achieve 67.5926% coverage (73/108) at nominal 80%, below target, and 97.2222% (105/108) at nominal 95%. Interval levels were not adjusted after observing holdout outcomes. These finite-sample coverage rates are descriptive, not a guarantee of future calibration.

## 12. Explainability

The SHAP artifacts provide local additive attributions for XGBoost outputs and summaries grouped by task and feature. Their interpretation is conditional on the fitted model, selected training data, feature engineering and background/reference setup. A positive attribution means a feature moved a model output upward relative to its baseline for that explanation; it does not mean an economic intervention would raise trade by that amount. The attributions are not causal estimates and do not replace controlled ablation or identification strategies.

## 13. Discussion

The 2025 results show that the frozen champion-and-fallback system outperformed the full Seasonal Naive benchmark on the aggregate operational sMAPE reported here, and the majority of the 42 task-level champions beat their task baseline on mean valid-origin sMAPE. The registry's 33 SARIMA and 9 XGBoost assignments also show that a single model family is not the preferred answer for every series and horizon under this protocol.

The aggregate should be read with the task table: task errors are heterogeneous, holdout length is one year, and a lower mean sMAPE does not establish statistical significance or economic value. The XGBoost 80% interval coverage result calls for calibration monitoring and explicit uncertainty display. It does not justify changing the frozen models after the holdout. Separately, the current source terms require rights review for the macro inputs before making a broad public or commercial reuse claim.

## 14. Limitations

1. **Vintage and timing:** Latest-vintage data with deterministic approximate lags are not original historical vintages or exact decision-time availability. Trade target release timing was not archived.
2. **Holdout size:** The final test contains twelve target months per task. It is a single recent period and does not establish stability across future regimes.
3. **Source coverage:** Russian-reporter mirror observations end at 2021-12 in the verified bundle. No independent national trade datasets are included.
4. **Measurement:** Bilateral mirror gaps can reflect partner attribution, timing, trade systems, classifications and valuation. They are not freight-only estimates.
5. **Model scope:** The result table covers 14 monthly HS2/total series and horizons 1, 3 and 6. It does not cover other products, frequencies or countries.
6. **Inference:** No significance test, causal model or macro-feature ablation supports causal interpretation or incremental macro-value claims.
7. **Intervals:** Aggregate coverage hides heterogeneity; XGBoost nominal 80% coverage is 67.59%. Seasonal Naive intervals are unavailable.
8. **Economic scale:** Targets are nominal USD values; inflation-adjusted or quantity forecasts are outside this specification.
9. **Licensing:** Source terms are provider-specific. The current FRED terms include software/ML-use restrictions, and the source owners' permission status for this workflow is unresolved. See the sources register.
10. **Serving:** The published dashboard is an artifact-backed decision-support interface. It does not retrain models or provide a live inference and data-refresh pipeline.

## 15. Conclusion

Under the frozen v1.1 protocol and retrospective deterministic-lag scope, the operational system's 2025 holdout mean sMAPE is lower than the Seasonal Naive aggregate, and 36 of 42 tasks show lower champion mean sMAPE than their full benchmark. The evidence supports a descriptive, sector- and horizon-dependent forecasting result. It does not establish real-time availability, macro causality, durable future performance, or universally calibrated intervals. The reproducible artifacts and current limitations are published alongside this report; source-rights clearance is still required before the macro-assisted workflow can be declared cleared for broader reuse.

## References

1. United Nations Statistics Division. *International Merchandise Trade Statistics: Concepts and Definitions 2010 (IMTS 2010).* [Official UNSD page](https://unstats.un.org/unsd/trade/EG-IMTS/EG-IMTS%20IMTS2010.htm); [document PDF](https://unstats.un.org/unsd/trade/eg-imts/imts2010-final-22march2011.pdf).
2. United Nations Statistics Division. *Guidelines on Analyzing and Reducing Bilateral Asymmetry in International Merchandise Trade Statistics* (2019). Linked from [UN Comtrade's bilateral asymmetry references](https://uncomtrade.org/docs/bilateral-asymmetries/).
3. Eurostat. *Asymmetries in foreign trade statistics in Mediterranean partner countries.* [Working paper PDF](https://ec.europa.eu/eurostat/documents/7828051/8076585/Asymmetries__trade_goods.pdf).
4. Bergmeir, C., Hyndman, R. J., and Koo, B. (2018). “A note on the validity of cross-validation for evaluating autoregressive time series prediction.” *Computational Statistics & Data Analysis*, 120, 70–83. [doi:10.1016/j.csda.2017.11.003](https://doi.org/10.1016/j.csda.2017.11.003).
5. Hyndman, R. J. and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed. [Open textbook, time-series cross-validation chapter](https://otexts.com/fpp3/tscv.html).
6. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2020). “The M4 Competition: 100,000 time series and 61 forecasting methods.” *International Journal of Forecasting*, 36(1), 54–74. [doi:10.1016/j.ijforecast.2019.04.014](https://doi.org/10.1016/j.ijforecast.2019.04.014).
7. Lundberg, S. M. and Lee, S.-I. (2017). “A Unified Approach to Interpreting Model Predictions.” *Advances in Neural Information Processing Systems 30*. [Proceedings paper](https://papers.nips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf).

## Project evidence

The paper is a synthesis of the versioned project evidence, not an independent rerun. Primary records: [data contract](../../specification/DATA_SPEC.md), [experiment protocol](../../specification/EXPERIMENT_PROTOCOL.md), [M10 champion report](../M10_CHAMPION_SELECTION_REPORT.md), [M11 holdout report](../M11_FINAL_HOLDOUT_REPORT.md), [M11 acceptance record](../M11_ACCEPTANCE_REPORT.md), and [M4 macro foundation report](../M4_MACRO_DATA_FOUNDATION_REPORT.md).
