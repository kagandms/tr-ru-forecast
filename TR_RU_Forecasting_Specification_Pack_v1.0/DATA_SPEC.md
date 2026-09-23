# DATA_SPEC.md
# Data Contract & Data Quality Specification

**Version:** 1.0  
**Status:** FROZEN FOR V1

## 1. Source policy
Primary canonical source: **UN Comtrade**.  
National sources: aggregate sanity-check, anomaly investigation, selected-period verification. They cannot silently overwrite canonical data.

## 2. Time scope
Default: `2012-01 → latest complete month`.

If data audit proves an early period unusable for a series, start may move forward with documented reason. Minimum history rules still apply.

## 3. Frequency
Monthly only. Standard period key: `YYYY-MM`.

## 4. Product classification
V1: **HS2**. HS4/HS6 modeling is out of scope.

HS revision/classification metadata must be retained. Silent cross-revision category merging is forbidden.

## 5. Canonical/mirror mapping
TR→RU canonical = TR exports to RU. Mirror = RU imports from TR.  
RU→TR canonical = RU exports to TR. Mirror = TR imports from RU.

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

## 12. Sector eligibility filters
Her direction için HS2 candidates:

1. minimum potential history: **96 months**
2. development-period coverage: **>=90%**
3. zero-month ratio: **<=10%**
4. recent activity: last 24 development months içinde **>=21 observed months**
5. non-trivial economic size: direction-specific development distribution üzerinden; holdout kullanılmadan

## 13. Sector ranking
Eligibility geçenler 0–100 score:

```text
Trade Volume        40%
Data Continuity     30%
Recent Relevance    20%
Forecastability     10%
```

### Volume
Development canonical value, log-transformed then normalized/ranked.

### Continuity
Coverage, longest missing streak penalty, zero-month penalty.

### Recent relevance
Last 24/36 development months: trade share + activity continuity + recent central volume.

### Forecastability
Final holdout kullanılmadan Seasonal Naive development backtest performance. Lower sMAPE → higher score.

### Final selection
Each direction: highest scoring **max 6 HS2**. Failed quality series cannot fill quota.

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
