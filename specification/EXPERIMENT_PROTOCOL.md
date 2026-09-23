# EXPERIMENT_PROTOCOL.md
# Forecasting & Evaluation Protocol

**Version:** 1.1  
**Status:** AMENDED (v1.1)  
**Previous Version:** 1.0 (Frozen initial release)  
**Changelog:** [`specification/CHANGELOG.md`](file:///Users/kagansmtdms/Downloads/tr-ru_forecast/specification/CHANGELOG.md)

## 1. Objective
Compare model families fairly under the same data and evaluation protocol.

Mandatory:
- Seasonal Naive
- SARIMA
- SARIMAX
- XGBoost

## 2. Forecast Targets (Specification v1.1)
Her iki yön için de official Türkiye-reported seriler modelleme hedefi (forecast target) olarak alınır:
- **TR → RU Direction:** monthly `trade_value_usd` of Türkiye exports to Russia (`reporter=792`, `partner=643`, `flow=X`).
- **RU → TR Direction:** monthly `trade_value_usd` of Türkiye imports from Russia (`reporter=792`, `partner=643`, `flow=M`).

Her seçilen aylık seri için (TOTAL + max 6 HS2 per direction = max 14 series):
- `h=1`
- `h=3`
- `h=6`

Target: monthly `trade_value_usd`.

## 3. Forecast strategy
### Seasonal Naive
Primary baseline:
```text
Forecast(t+h) = Actual(t+h-12)
```

### SARIMA
Univariate monthly model; native multi-step forecast.

### SARIMAX
SARIMA + publication-safe exogenous variables. Future actual exogenous values are forbidden.

### XGBoost
**Direct forecasting**:
```text
XGB_h1
XGB_h3
XGB_h6
```
No recursive XGBoost forecast.

## 4. Final holdout (Specification v1.1)
**FINAL HOLDOUT LENGTH = 12 MONTHS** (Kesin kural olarak korunmuştur).

Mevcut doğrulanmış Comtrade veri evreni (`2012-01 → 2025-12`) için tarihler:
- **Development Period:** `2012-01 → 2024-12` (156 ay)
- **Final Untouched Holdout:** `2025-01 → 2025-12` (12 ay)

> [!IMPORTANT]
> 24 aylık holdout KULLANILMAYACAKTIR. Pre-M3 doğrulama raporundaki 24 aylık holdout önerisi authoritative değildir; bu protokol frozen 12 aylık holdout süresini kilitli tutar.

Forbidden uses of holdout:
- sector selection
- feature selection
- hyperparameter tuning
- champion-rule design

Opened only for final evaluation after development freeze.

## 5. Development backtest
Validation: **expanding walk-forward / rolling-origin**.

Default:
- monthly origins
- aim for at least 36 pre-holdout origins where feasible
- minimum initial training history: **60 months**

Each origin uses only information available at that origin.

## 6. Tuning
No holdout tuning.

XGBoost:
- controlled/small search space
- time-aware inner validation
- deterministic seed

SARIMA/SARIMAX:
- development-period order selection
- no excessive brute-force search

## 7. Metrics
Primary: **sMAPE**  
Secondary: **MAE**  
Diagnostic: **RMSE**

Compute by:
- direction
- sector
- horizon
- model

Aggregates may use medians; one global metric cannot replace per-series reporting.

## 8. Champion selection
For each `(direction, sector, horizon)`:

1. lowest validation sMAPE
2. MAE as tie-breaker
3. if complex model gives **<2% relative sMAPE improvement** over simpler model, choose simpler model

Relative improvement:
```text
(simple_sMAPE - complex_sMAPE) / simple_sMAPE
```

Complexity order:
```text
Seasonal Naive < SARIMA < SARIMAX < XGBoost
```

Rule is applied before holdout is opened.

## 9. Final evaluation
After champion freeze:
- refit using development data (`2012-01 → 2024-12`)
- evaluate untouched 12-month holdout (`2025-01 → 2025-12`)
- report validation and holdout separately

If holdout result disappoints, champion is not retroactively changed.

## 10. Leakage rules
Forbidden:
- future targets
- future rolling windows
- holdout-informed sector selection
- scaling before fold split
- future macro actuals
- test/holdout-driven feature editing

Every feature must pass: **"Was this knowable at forecast origin?"**

## 11. Feature rules
Allowed:
- lags
- backward rolling stats
- growth rates
- calendar variables
- publication-safe macro variables

Direct h1/h3/h6 target shifts require explicit unit tests.

## 12. Prediction intervals
UI/API:
- 80%
- 95%

SARIMA/SARIMAX: native intervals acceptable.

XGBoost: horizon-specific development backtest residual/conformal calibration. Holdout cannot calibrate intervals.

## 13. Explainability
XGBoost MUST provide:
- global SHAP
- local SHAP for selected/latest forecast

Wording must be "forecast drivers/model contribution", never causal effect.

## 14. Statistical diagnostics
SARIMA/SARIMAX minimum artifacts:
- order
- seasonal order
- residual summary
- convergence status
- failure/fallback logs

Fit failures must not crash full pipeline.

## 15. Reproducibility metadata
Every run:
```text
run_id
code_version
data_snapshot_id
direction
sector
horizon
model_type
features
hyperparameters
train_start
train_end
validation_origins
metrics
random_seed
created_at
```

## 16. Research reporting
Final report separates:
- validation
- holdout
- model family
- horizon
- direction
- sector heterogeneity
- mirror/data quality
- limitations

Do not report generic "accuracy %" for regression.

## 17. Interpretation
Allowed:
> XGBoost achieved lower holdout sMAPE than the seasonal naive baseline for HSxx at h=1.

Not allowed:
> Brent caused exports to rise.

## 18. Experiment freeze before holdout
Must already be frozen:
- sectors
- feature families
- model families
- metrics
- champion rule
- interval method
- holdout months
