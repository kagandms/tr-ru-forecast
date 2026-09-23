# PROJECT_SPEC.md
# TR–RU Bilateral Trade Forecasting & Intelligence Platform

**Version:** 1.0  
**Status:** FROZEN FOR V1

## 1. Product statement
Türkiye ile Rusya arasındaki iki yönlü aylık ticaret akımlarını analiz eden; 1/3/6 aylık tahminler, model karşılaştırmaları, uncertainty, SHAP ve mirror-statistics veri kalite analizi sunan research-oriented economic intelligence platform.

## 2. Primary research question
> Can statistical and machine-learning models improve forecasts of bilateral trade between Türkiye and Russia compared with simple historical benchmarks?

Secondary:
- Makro/emtia değişkenleri yalnızca geçmiş ticarete göre tahmini iyileştiriyor mu?
- TR→RU ve RU→TR farklı tahmin edilebilirliğe sahip mi?
- En başarılı model sektör/horizon bazında değişiyor mu?
- Reporter/mirror gaps veri kalitesi ve forecast error ile nasıl ilişkilidir?

## 3. Canonical directions
### TR → RU
Canonical: Türkiye reporter, Russia partner, Export.  
Mirror: Russia reporter, Türkiye partner, Import.

### RU → TR
Canonical: Russia reporter, Türkiye partner, Export.  
Mirror: Türkiye reporter, Russia partner, Import.

Mirror value canonical target'ı otomatik overwrite edemez.

## 4. Forecast scope
Her seri için:
- `h=1`
- `h=3`
- `h=6`

Seriler:
- TOTAL
- direction başına kalite kriterlerini geçen **max 6 HS2**

Maximum: **14 series**.

## 5. Mandatory model families
1. Seasonal Naive
2. SARIMA
3. SARIMAX
4. XGBoost

LightGBM optional; deep learning V1 dışı.

## 6. Product modules
### Overview
TR→RU, RU→TR, bilateral total, Türkiye trade balance, YoY, last update.

### Trade Explorer
Direction/sector seçimi, historical series, rolling stats, YoY, canonical-vs-mirror.

### Forecasts
1/3/6 month point forecast, 80/95 interval, champion model, latest evaluation summary.

### Sector Analysis
Selected HS2, volume, continuity, recent relevance, selection score, trade contribution.

### Model Lab
Seasonal Naive / SARIMA / SARIMAX / XGBoost için sMAPE, MAE, RMSE.

### Mirror Statistics
Canonical, mirror, absolute gap, percentage gap, trend, missing/mismatch flags.

### Economic Indicators
Modelde kullanılan exogenous variables ve publication-safe lags.

### Methodology
Target, sources, sector selection, validation, holdout, metrics, intervals, limitations.

## 7. Languages
UI MUST support:
- Türkçe
- English
- Русский

Model layer language-independent kalır.

## 8. Architecture
```text
External Sources
      ↓
Ingestion
      ↓
Raw / Validated Trade & Macro
      ↓
Feature Engineering
      ├──→ Mirror/Data Quality Analytics
      ↓
Forecasting
Seasonal Naive / SARIMA / SARIMAX / XGBoost
      ↓
Evaluation + Champion Selection
      ↓
Model Registry + Forecast Storage
      ↓
FastAPI
      ↓
Next.js Dashboard
TR / EN / RU
```

## 9. Technology constraints
### MUST
- Python data/model layer
- PostgreSQL-compatible DB; Supabase acceptable
- FastAPI
- Next.js
- reproducible dependencies
- automated tests for critical transformations/evaluation
- secrets via environment variables
- no secrets in Git

### SHOULD
- Docker/local reproducibility
- structured logging
- deterministic seeds
- source/data snapshot metadata

## 10. V1 non-goals
- LSTM / Transformer / TFT
- other countries
- HS6 forecasting
- mobile app
- real-time streaming
- LLM-generated forecasts
- auto-trading
- custom user-trained models
- dozens of macro variables
- continuous/daily retraining
- causal inference claims

## 11. Principles
- Baselines mandatory.
- Complexity is not success.
- No future information leakage.
- Point forecast alone is insufficient; uncertainty visible.
- Mirror data is a quality layer first.
- Reproducibility is mandatory.
- Negative/null research results are reported, not hidden.

## 12. Project-level Definition of Done
V1 complete only when:
- both canonical directions work,
- TOTAL + selected HS2 work,
- 1/3/6 forecasts work,
- all four mandatory model families backtested,
- expanding walk-forward applied,
- final 12-month holdout remained untouched during development,
- deterministic champion rule applied,
- 80/95 intervals available,
- SHAP available for XGBoost,
- mirror analytics works,
- API feeds frontend,
- TR/EN/RU UI works,
- Methodology page exists,
- README + research report exist,
- critical tests pass,
- limitations documented.

## 13. Success criterion
XGBoost'un her seride baseline'ı yenmesi şart değildir. Başarı; protocol correctness, reproducibility, honest comparison ve working product ile ölçülür.

## 14. Change control
Holdout görüldükten sonra metric, champion rule, holdout definition veya feature policy geriye dönük değiştirilemez.
