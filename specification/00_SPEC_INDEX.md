# TR–RU Bilateral Trade Forecasting & Intelligence Platform
## Specification Pack v1.1

**Status:** FROZEN SPECIFICATION v1.1; implementation accepted through M13
**Version:** 1.1  
**Previous Version:** 1.0 (Frozen initial release)  
**Changelog:** [`CHANGELOG.md`](CHANGELOG.md)

Bu paket proje kapsamını, veri sözleşmesini, deney protokolünü ve milestone kabul kriterlerini kilitler. v1.1 sürümü, Pre-M3 authenticated full-history verification bulguları doğrultusunda forecast target source politikasını güncellemiştir.

## Belgeler
1. `00_SPEC_INDEX.md` — spec paketi indeksi, temel kararlar ve change control.
2. `PROJECT_SPEC.md` — ürün kapsamı, V1 sınırları, mimari ve Definition of Done.
3. `DATA_SPEC.md` — forecast target / reference veri kuralları, şemalar, sektör seçimi ve veri kalitesi.
4. `EXPERIMENT_PROTOCOL.md` — modeller, validation, holdout, leakage, metrikler ve champion seçimi.
5. `MILESTONES.md` — coding sırası, çıktılar, acceptance criteria ve stop/go kapıları.
6. `CHANGELOG.md` — specification versiyon geçmişi ve değişiklik gerekçeleri.

## Freeze edilen ana kararlar (v1.1)
- Primary trade source: **UN Comtrade** (authenticated subscription access)
- National sources: **cross-check & verification**
- Historical period: **2012-01 → latest complete available reporting month** (verified dataset: `2012-01` → `2025-12`)
- Forecast targets:
  - **TR→RU Forecast Target:** Türkiye reporter, Russia partner, Export (Türkiye'nin Rusya'ya ihracatı)
  - **RU→TR Forecast Target:** Türkiye reporter, Russia partner, Import (Türkiye'nin Rusya'dan ithalatı)
  - *Her iki forecast target da Türkiye-reported resmi dış ticaret serisidir.*
- Reference / mirror series:
  - TR→RU reference: Russia imports from Türkiye
  - RU→TR reference: Russia exports to Türkiye
  - Reference seriler forecast target değildir; imputation, automatic replacement veya averaging için kullanılamaz. Tarihsel veri kalitesi ve karşılaştırmalı araştırma katmanı olarak korunur.
- Terminology disambiguation: **SOURCE SERIES**, **FORECAST TARGET**, **MIRROR / REFERENCE SERIES**
- Product granularity: **HS2**
- Sectors: **max 6 per direction + TOTAL**
- Selection: **quality filter + weighted ranking** (development period üzerinde hesaplanır; holdout hariç)
- Horizons: **1 / 3 / 6 months**
- Models: **Seasonal Naive, SARIMA, SARIMAX, XGBoost**
- XGBoost: **direct forecasting**
- Validation: **expanding walk-forward**
- Final holdout: **last 12 complete available months, untouched** (`2025-01` → `2025-12`)
- Development period: **2012-01 → 2024-12** (156 months)
- Primary metric: **sMAPE**
- Secondary: **MAE**
- Diagnostic: **RMSE**
- Exogenous variables: **publication-aware**
- Explainability: **SHAP**
- Uncertainty: **80% and 95% intervals**
- Stack: **Python + PostgreSQL/Supabase + FastAPI + Next.js**
- Languages: **TR / EN / RU**

## Change control
`MUST` kuralları ancak somut teknik/veri gerekçesi, yazılı etki analizi ve spec version bump (`v1.1+`) ile değiştirilebilir. Holdout sonucuna bakıp metodoloji değiştirmek yasaktır. v1.1 güncellemesi, Pre-M3 full-history verification sırasında Russia-reported RU→TR serisinin 2021-12 sonrası Comtrade'de yer almadığının doğrulanması ve Türkiye-reported serilerin 2012-2025 arasında kesintisiz mevcudiyeti nedeniyle yapılmıştır.

## Coding workflow
`Spec Freeze → Milestone Prompt → Implementation → Review → Acceptance Gate → Next Prompt`

Coding agent bütün projeyi tek seferde uygulamayacaktır.
