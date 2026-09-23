# TR–RU Bilateral Trade Forecasting & Intelligence Platform
## Specification Pack v1.0

**Status:** SPECIFICATION FREEZE  
**Version:** 1.0

Bu paket coding başlamadan önce proje kapsamını, veri sözleşmesini, deney protokolünü ve milestone kabul kriterlerini kilitler.

## Belgeler
1. `PROJECT_SPEC.md` — ürün kapsamı, V1 sınırları, mimari ve Definition of Done.
2. `DATA_SPEC.md` — canonical/mirror veri kuralları, şemalar, sektör seçimi ve veri kalitesi.
3. `EXPERIMENT_PROTOCOL.md` — modeller, validation, holdout, leakage, metrikler ve champion seçimi.
4. `MILESTONES.md` — coding sırası, çıktılar, acceptance criteria ve stop/go kapıları.

## Freeze edilen ana kararlar
- Primary trade source: **UN Comtrade**
- National sources: **cross-check**
- Period: **2012-01 → latest complete month**
- Canonical flows: **TR→RU exporter-reported** ve **RU→TR exporter-reported**
- Mirror data: quality/research layer; canonical target replacement değil
- Product granularity: **HS2**
- Sectors: **max 6 per direction + TOTAL**
- Selection: **quality filter + weighted ranking**
- Horizons: **1 / 3 / 6 months**
- Models: **Seasonal Naive, SARIMA, SARIMAX, XGBoost**
- XGBoost: **direct forecasting**
- Validation: **expanding walk-forward**
- Final holdout: **last 12 complete months, untouched**
- Primary metric: **sMAPE**
- Secondary: **MAE**
- Diagnostic: **RMSE**
- Exogenous variables: **publication-aware**
- Explainability: **SHAP**
- Uncertainty: **80% and 95% intervals**
- Stack: **Python + PostgreSQL/Supabase + FastAPI + Next.js**
- Languages: **TR / EN / RU**

## Change control
`MUST` kuralları ancak somut teknik/veri gerekçesi, yazılı etki analizi ve spec version bump (`v1.1+`) ile değiştirilebilir. Holdout sonucuna bakıp metodoloji değiştirmek yasaktır.

## Coding workflow
`Spec Freeze → Milestone Prompt → Implementation → Review → Acceptance Gate → Next Prompt`

Coding agent bütün projeyi tek seferde uygulamayacaktır.
