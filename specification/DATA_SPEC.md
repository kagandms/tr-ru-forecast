# DATA_SPEC.md
# Data Contract & Data Quality Specification

**Version:** 1.1  
**Status:** AMENDED (v1.1)  
**Previous Version:** 1.0 (Frozen initial release)  
**Changelog:** [`specification/CHANGELOG.md`](file:///Users/kagansmtdms/Downloads/tr-ru_forecast/specification/CHANGELOG.md)

## 1. Source policy
Primary trade source: **UN Comtrade** (authenticated subscription data access).  
National sources: aggregate sanity-check, anomaly investigation, selected-period verification. They cannot silently overwrite primary data.

## 2. Time scope & Data cutoff
- **Historical Horizon:** `2012-01 → latest complete available reporting month`.
- **Latest Complete Available Reporting Month:** Defined as the most recent month for which official trade returns have been released and verified (`2025-12` in verified dataset).
- **Calendar vs Source Cutoff Distinction:** The calendar latest complete month (`2026-08`) is distinct from source availability (`2025-12`). Periods `2026-01` through `2026-08` have not yet been released by the reporting source and are **NOT** observations in the modeling dataset.
- **Modeling Dataset Span:** `2012-01 → 2025-12` (168 complete monthly observations).
- **Development Period:** `2012-01 → 2024-12` (156 months).
- **Final Untouched Holdout:** `2025-01 → 2025-12` (12 months).

## 3. Frequency
Monthly only. Standard period key: `YYYY-MM`.

## 4. Product classification
V1: **HS2**. HS4/HS6 modeling is out of scope.

HS revision/classification metadata must be retained. Silent cross-revision category merging is forbidden. Verified nomenclature regimes: H4 (2012-2016), H5 (2017-2021), H6 (2022-2025).

## 5. Forecast target & Reference mapping (Specification v1.1)

### Terminology Disambiguation
- **SOURCE SERIES:** Raw bilateral trade data reported by an official customs agency to UN Comtrade.
- **FORECAST TARGET:** The dependent series to be forecasted and evaluated in modeling milestones.
- **MIRROR / REFERENCE SERIES:** Counter-reporting bilateral series preserved strictly as a data-quality and research layer.

### Bilateral Mappings
- **TR → RU Direction:**
  - **Forecast Target:** Türkiye reporter (792), Russia partner (643), Export ('X').  
    *Semantic:* Türkiye'nin Rusya'ya ihracatı. (Continuous 2012-01 → 2025-12).
  - **Reference / Mirror Series:** Russia reporter (643), Türkiye partner (792), Import ('M').  
    *Semantic:* Russia's imports from Türkiye. (Available 2013-01 → 2021-12).
- **RU → TR Direction:**
  - **Forecast Target:** Türkiye reporter (792), Russia partner (643), Import ('M').  
    *Semantic:* Türkiye'nin Rusya'dan ithalatı. (Continuous 2012-01 → 2025-12).
  - **Reference / Mirror Series:** Russia reporter (643), Türkiye partner (792), Export ('X').  
    *Semantic:* Russia's exports to Türkiye. (Available 2013-01 → 2021-12).

### Target and Reference Policy Constraints:
1. **Official Türkiye-reported series:** Both forecasting targets (`TR_RU` export, `RU_TR` import) are official Türkiye-reported series. This resolves the lack of post-2021 Russian reporting while maintaining unbroken 2012-2025 series for both trade directions.
2. **Reference / Mirror Status:** Reference series are:
   - **NOT** forecast targets,
   - **NOT** imputations,
   - **NOT** automatic replacements,
   - **NOT** averaging sources.
3. **Neutral Research Interpretation:** "No Russia-reported RU→TR bilateral export observations are available in the verified UN Comtrade dataset after 2021-12." Speculative or causal assertions regarding reporting absence must not be stated.
4. **Storage Provenance:** Historical M1/M2 storage naming (`trade_monthly_TR_RU`, `trade_monthly_RU_TR`, `mirror_monthly`) remains intact for raw data provenance. Modeling layers query the designated forecast target series under this v1.1 policy.

## 6. Required raw trade fields
```text
period
reporter_code
reporter_name
partner_code
partner_name
flow_code
flow_name
hs_code
hs_description
trade_value_usd
net_weight_kg
quantity
quantity_unit
classification
source
source_record_id_or_key
retrieved_at
```

Unavailable source fields may be nullable but never silently fabricated.

## 7. Standardized trade contract
```text
trade_monthly
-------------
period
direction
reporter_code
reporter_name
partner_code
partner_name
flow
hs2
hs_description
trade_value_usd
net_weight_kg
quantity
classification
is_total
source
retrieved_at
data_status
```

`direction ∈ {TR_RU, RU_TR}`.

## 8. Mirror analytical contract
```text
mirror_monthly
--------------
period
direction
hs2
canonical_value_usd
mirror_value_usd
mirror_gap_usd
mirror_gap_pct
canonical_missing
mirror_missing
gap_flag
retrieved_at
```

Definitions:
```text
mirror_gap_usd = canonical - mirror
mirror_gap_pct = (canonical - mirror) / canonical * 100
```
Canonical zero ise percentage gap NULL.

## 9. Missing policy
Ayrı durumlar:
1. reported zero
2. missing/unavailable
3. observed non-zero

Rules:
- missing → 0 yasak
- automatic interpolation yasak
- mirror → canonical replacement yasak
- modeling sırasında drop/impute decisions raporlanır

## 10. Outlier policy
Default: **flag + investigate + preserve if economically genuine**.

Possible flags:
- robust z-score / MAD
- extreme MoM
- extreme YoY
- mirror-gap spike

Data error kanıtı yoksa observation korunur. Real shocks cleaned away edilmez.

## 11. Duplicate policy
Minimum canonical key:
```text
period + reporter + partner + flow + hs2 + classification
```
Duplicate resolution deterministic ve logged olmalı. Blind `drop_duplicates(keep="first")` yasak.

## 12. Sector eligibility filters (Specification v1.1)
Her trade direction için HS2 adayları doğrudan ilgili **Forecast Target** serisi üzerinden (TR_RU için Türkiye ihracatı, RU_TR için Türkiye ithalatı) ve **development period** (`2012-01 → 2024-12`, 156 ay) sınırları içinde değerlendirilir. Final 12 aylık holdout (`2025-01 → 2025-12`) kesinlikle bu filtrelemeye dahil edilmez.

Filtreler:
1. minimum potential history: **96 months**
2. development-period coverage: **>=90%** (156 aylık development periyodunda >= 141 gözlem)
3. zero-month ratio: **<=10%**
4. recent activity: last 24 development months (`2023-01 → 2024-12`) içinde **>=21 observed months**
5. non-trivial economic size: direction-specific development distribution üzerinden; holdout kullanılmadan

## 13. Sector ranking (Specification v1.1)
Eligibility kriterlerini geçen sektörler 0–100 arasında ağırlıklı puanlanır:

```text
Trade Volume        40%
Data Continuity     30%
Recent Relevance    20%
Forecastability     10%
```

### Volume
Development forecast target value (`2012-01 → 2024-12` toplam değeri), log-transformed then normalized/ranked.

### Continuity
Coverage, longest missing streak penalty, zero-month penalty.

### Recent relevance
Last 24 development months (`2023-01 → 2024-12`): trade share + activity continuity + recent central volume.

### Forecastability
Final 2025 holdout kullanılmadan Seasonal Naive development backtest performance. Lower sMAPE → higher score.

### Final selection
Each direction: highest scoring **max 6 HS2**. Failed quality series cannot fill quota. Sektör seçimi M3 aşamasında dondurulur.

## 14. Initial exogenous candidates
- USD/TRY
- RUB/USD
- Brent / energy indicator
- Türkiye industrial production
- Russia industrial production

Additional macro variables only if:
- reliable source
- enough history
- known/defensible release lag
- economic justification

## 15. Publication-awareness
Each indicator metadata:
```text
indicator_id
indicator_name
source
frequency
reference_period
release_lag_rule
availability_date_or_approximation
transformation
```

Forecast origin'de bilinmeyen actual macro values forbidden.

Fallback if exact release calendar impractical: **lag macro feature by at least 1 month**, document it.

## 16. Allowed macro transformations
- lagged level
- MoM change
- YoY change
- log transform
- rolling mean

Transformers/scalers fit only within training folds.

## 17. Modeling feature contract
```text
period
direction
hs2
target
target_h1
target_h3
target_h6
lag_1
lag_2
lag_3
lag_6
lag_12
rolling_mean_3
rolling_mean_6
rolling_mean_12
rolling_std_3
rolling_std_6
rolling_std_12
mom_growth
yoy_growth
month
quarter
year
usd_try_lagged
rub_usd_lagged
brent_lagged
tr_ip_lagged
ru_ip_lagged
feature_snapshot_date
```

All rolling/lag features must be strictly backward-looking.

## 18. Data quality report
Every refresh must report:
- row count
- min/max period
- duplicates
- missing ratio
- zero ratio
- coverage by sector
- longest missing streak
- canonical/mirror availability
- extreme-change flags
- selected sectors
- retrieval timestamp

## 19. Experiment data freeze
Final holdout months saved explicitly. Sector selection and feature design cannot use them.

## 20. Data acceptance gate
Data foundation passes only if:
- both canonical directions exist,
- TOTAL and HS2 distinguishable,
- missing != zero guaranteed,
- duplicates deterministic,
- mirror pairing works,
- quality report generated,
- sector selection reproducible,
- holdout excluded from selection.
