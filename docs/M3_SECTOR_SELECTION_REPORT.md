# M3 — DATA AUDIT & SECTOR SELECTION REPORT

**Document Version:** 1.1.0  
**Execution Date:** 2026-09-21T17:52:12.100234+00:00  
**Authoritative Specification:** Specification Pack v1.1 (`specification/`)  
**Author:** Senior Data Analyst & Forecasting Research Engineer  

---

## 1. EXECUTIVE SUMMARY

Milestone M3 establishes the empirical data foundation and frozen commodity scope for the TR–RU Bilateral Trade Forecasting & Intelligence Platform under Specification v1.1. Following the pre-M3 discovery of reporting divergence, both forecast targets are materialized strictly as official Türkiye-reported series. Sector eligibility and multi-factor ranking are executed exclusively over the 156-month development period (`2012-01` through `2024-12`). The 12-month final holdout (`2025-01` through `2025-12`) is strictly isolated and untouched.

A total of **12 commodity sectors** (6 per direction) plus **2 bilateral total series** are selected for the subsequent forecasting milestones, establishing an empirical scope of **14 distinct monthly target series** across 3 forecast horizons ($h=1, 3, 6$).

---

## 2. TARGET POLICY & SOURCE PROVENANCE VERIFICATION

Under Specification v1.1, forecasting targets must be official, continuous series. Both directions utilize Türkiye customs returns (`reporter_code=792`, `partner_code=643`):

| Metric / Property | TR → RU Forecast Target | RU → TR Forecast Target |
| :--- | :--- | :--- |
| **Semantic Concept** | Türkiye exports to Russia | Türkiye imports from Russia |
| **Reporter Code & Country** | `792` (Türkiye) | `792` (Türkiye) |
| **Partner Code & Country** | `643` (Russian Federation) | `643` (Russian Federation) |
| **Trade Flow** | Export (`flow_code='X'`) | Import (`flow_code='M'`) |
| **Source Dataset** | `trade_monthly_TR_RU.parquet` | `trade_monthly_mirror_RU_TR.parquet` |
| **Full Modeling Span** | `2012-01` → `2025-12` (168 months) | `2012-01` → `2025-12` (168 months) |
| **Development Period** | `2012-01` → `2024-12` (156 months) | `2012-01` → `2024-12` (156 months) |
| **Final Holdout (Untouched)** | `2025-01` → `2025-12` (12 months) | `2025-01` → `2025-12` (12 months) |
| **Russia Exporter Data Used?** | **NO** (0 records) | **NO** (0 records) |

> [!IMPORTANT]
> **Russia Exporter Data Quarantine:** The historical dataset `trade_monthly_RU_TR.parquet` contains Russian customs returns ending at `2021-12`. Under Specification v1.1 and runtime guards, this dataset is strictly prohibited from serving as the RU_TR forecast target. The official Turkish import dataset (`trade_monthly_mirror_RU_TR.parquet`) provides the unbroken 2012–2025 observational series.

---

## 3. DEVELOPMENT PERIOD & HOLDOUT LEAKAGE GUARD

- **Development Window:** `2012-01` through `2024-12` (156 calendar months).
- **Holdout Period:** `2025-01` through `2025-12` (12 calendar months).
- **Programmatic Holdout Guard:** Implemented in `tr_ru_forecast.data.sectors.panel_builder.enforce_holdout_guard`. Any record beyond `2024-12` raises an unhandled `ValueError` at runtime.
- **Leakage Invariant:** Unit tests confirm that mutating 2025 holdout trade values has zero effect on sector eligibility, metric values, or selection rankings.

---

## 4. CANDIDATE HS2 UNIVERSE & NOMENCLATURE AUDIT

- **TR_RU Observed HS2 Chapters:** 97 chapters.
- **RU_TR Observed HS2 Chapters:** 97 chapters.
- **TOTAL Scope:** Bilateral total flows are automatically preserved for modeling but strictly excluded from sector candidate ranking.

### HS Revision Regimes (2012–2024 Development Horizon)
- **Nomenclatures Observed:** H4, H5, H6
- **H4 (HS 2012):** 2012-01 to 2016-12
- **H5 (HS 2017):** 2017-01 to 2021-12
- **H6 (HS 2022):** 2022-01 to 2024-12 (in development span)
- **Chapter Stability:** HS 2-digit chapter headings (01–97) demonstrate complete structural continuity across all 3 revision transitions.

### Special & Residual Chapter Exclusions
- **Chapter 77:** Reserved by WCO for future potential allocation; 0 records observed.
- **Chapter 98:** Special classification provisions / postal packages; excluded from ranking.
- **Chapter 99:** Commodities not specified according to kind / special provisions / confidential trade; audited and excluded from ranking as a residual non-production category.

---

## 5. ELIGIBILITY FILTER EVALUATION

All candidate sectors were subjected to five deterministic criteria (A through E):
- **Filter A (History Potential):** $\ge 96$ months potential span.
- **Filter B (Coverage):** Observed months $\ge 141$ / 156 ($\ge 90.0\%$).
- **Filter C (Zero Ratio):** Zero trade months $\le 10.0\%$ of observed months.
- **Filter D (Recent Activity):** Observed months in `2023-01` → `2024-12` $\ge 21$ / 24.
- **Filter E (Economic Size Floor):** Development total trade value $\ge 10\text{th}$ percentile ($P_{10}$) among candidates passing A–D.

### Direction Summary:

| Direction | Total Candidates | Passing A–D | P10 Size Floor (USD) | Passing Filter E (Final Eligible) | Disqualified |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TR → RU** | 97 | 69 | $50,943,545.20 | **62** | 35 |
| **RU → TR** | 97 | 47 | $29,569,974.40 | **42** | 55 |

---

## 6. MULTI-FACTOR SCORING & WEIGHTED RANKING

Eligible candidates were scored on a 0–100 scale using four orthogonal components:
$$\text{Final Score} = 0.40 \times \text{Volume} + 0.30 \times \text{Continuity} + 0.20 \times \text{Recent} + 0.10 \times \text{Forecastability}$$

1. **Volume Score (40%):** Fractional percentile rank of $\log(1 + \text{Development Total Trade Value})$.
2. **Continuity Score (30%):** $100 \times [0.60 \times \text{Coverage} + 0.25 \times (1 - \text{Zero Ratio}) + 0.15 \times (1 - \text{Streak} / 156)]$.
3. **Recent Relevance (20%):** $0.80 \times \text{Recent Volume Percentile} + 0.20 \times (\text{Recent Coverage} \times 100)$.
4. **Forecastability (10%):** Development Seasonal Naive ($h=1, t-12$) sMAPE fractional percentile rank (lower sMAPE yields higher score).

Deterministic tie-break order: `final_sector_score DESC` $\rightarrow$ `volume_score DESC` $\rightarrow$ `development_total_trade_value DESC` $\rightarrow$ `hs2 ASC`.

---

## 7. FINAL RANKINGS — TR → RU (TÜRKİYE EXPORTS TO RUSSIA)

### Top 15 Eligible Candidates (Top 6 Frozen for Modeling):

| Rank | HS2 | Description | Volume | Continuity | Recent | Forecastability | sMAPE (%) | Total Dev Value (USD) | Final Score | Selection Status |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **08** | Edible fruit and nuts; peel of citrus fr... | 98.4 | 100.0 | 98.7 | 93.4 | 34.0% | $9,426,138,591 | **98.43** | **SELECTED** |
| 2 | **84** | Nuclear reactors, boilers, machinery and... | 100.0 | 100.0 | 100.0 | 80.3 | 38.6% | $11,331,507,857 | **98.03** | **SELECTED** |
| 3 | **39** | Plastics and articles thereof... | 95.1 | 100.0 | 96.1 | 91.8 | 34.5% | $4,058,065,836 | **96.43** | **SELECTED** |
| 4 | **87** | Vehicles other than railway or tramway r... | 96.7 | 100.0 | 97.4 | 78.7 | 39.0% | $6,478,746,694 | **96.03** | **SELECTED** |
| 5 | **73** | Articles of iron or steel... | 88.5 | 100.0 | 90.8 | 96.7 | 33.3% | $2,078,967,808 | **93.25** | **SELECTED** |
| 6 | **85** | Electrical machinery and equipment and p... | 93.4 | 100.0 | 94.8 | 68.8 | 41.7% | $4,023,873,429 | **93.21** | **SELECTED** |
| 7 | **62** | Articles of apparel and clothing accesso... | 86.9 | 100.0 | 92.1 | 85.2 | 37.9% | $1,935,485,595 | **91.70** | Eligible (Unselected) |
| 8 | **61** | Articles of apparel and clothing accesso... | 85.2 | 100.0 | 88.2 | 70.5 | 41.6% | $1,821,388,186 | **88.79** | Eligible (Unselected) |
| 9 | **03** | Fish and crustaceans, molluscs and other... | 82.0 | 100.0 | 93.4 | 67.2 | 43.0% | $1,806,042,836 | **88.20** | Eligible (Unselected) |
| 10 | **83** | Miscellaneous articles of base metal... | 78.7 | 100.0 | 81.6 | 100.0 | 25.0% | $1,140,892,035 | **87.80** | Eligible (Unselected) |
| 11 | **32** | Tanning or dyeing extracts; tannins and ... | 77.0 | 100.0 | 89.5 | 75.4 | 40.7% | $1,104,902,175 | **86.26** | Eligible (Unselected) |
| 12 | **34** | Soap, organic surface-active agents, was... | 72.1 | 100.0 | 84.3 | 98.4 | 32.3% | $1,026,874,088 | **85.54** | Eligible (Unselected) |
| 13 | **60** | Knitted or crocheted fabrics... | 90.2 | 100.0 | 83.0 | 26.2 | 65.2% | $2,244,846,068 | **85.28** | Eligible (Unselected) |
| 14 | **07** | Edible vegetables and certain roots and ... | 91.8 | 100.0 | 76.4 | 19.7 | 69.4% | $2,299,153,009 | **83.97** | Eligible (Unselected) |
| 15 | **27** | Mineral fuels, mineral oils and products... | 83.6 | 100.0 | 85.6 | 29.5 | 63.0% | $1,816,402,449 | **83.51** | Eligible (Unselected) |

---

## 8. FINAL RANKINGS — RU → TR (TÜRKİYE IMPORTS FROM RUSSIA)

### Top 15 Eligible Candidates (Top 6 Frozen for Modeling):

| Rank | HS2 | Description | Volume | Continuity | Recent | Forecastability | sMAPE (%) | Total Dev Value (USD) | Final Score | Selection Status |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **27** | Mineral fuels, mineral oils and products... | 100.0 | 100.0 | 100.0 | 90.2 | 36.7% | $244,275,005,985 | **99.02** | **SELECTED** |
| 2 | **72** | Iron and steel... | 97.6 | 100.0 | 98.0 | 95.1 | 32.0% | $43,113,837,633 | **98.15** | **SELECTED** |
| 3 | **10** | Cereals... | 95.1 | 100.0 | 96.1 | 75.6 | 51.8% | $21,182,640,603 | **94.83** | **SELECTED** |
| 4 | **76** | Aluminium and articles thereof... | 92.7 | 100.0 | 92.2 | 87.8 | 38.2% | $12,821,599,197 | **94.29** | **SELECTED** |
| 5 | **23** | Residues and waste from the food industr... | 85.4 | 100.0 | 90.2 | 92.7 | 33.2% | $5,245,108,836 | **91.46** | **SELECTED** |
| 6 | **74** | Copper and articles thereof... | 90.2 | 100.0 | 94.2 | 51.2 | 67.1% | $8,740,595,885 | **90.05** | **SELECTED** |
| 7 | **15** | Animal, vegetable or microbial fats and ... | 87.8 | 99.5 | 88.3 | 63.4 | 61.8% | $8,427,289,144 | **88.98** | Eligible (Unselected) |
| 8 | **29** | Organic chemicals... | 82.9 | 100.0 | 82.4 | 82.9 | 42.4% | $3,750,605,062 | **87.95** | Eligible (Unselected) |
| 9 | **39** | Plastics and articles thereof... | 80.5 | 100.0 | 86.3 | 68.3 | 56.5% | $3,413,247,318 | **86.29** | Eligible (Unselected) |
| 10 | **28** | Inorganic chemicals; compounds of precio... | 78.0 | 100.0 | 76.6 | 85.4 | 40.1% | $3,343,574,717 | **85.07** | Eligible (Unselected) |
| 11 | **48** | Paper and paperboard; articles of paper ... | 73.2 | 100.0 | 68.8 | 100.0 | 28.6% | $2,045,118,481 | **83.02** | Eligible (Unselected) |
| 12 | **84** | Nuclear reactors, boilers, machinery and... | 68.3 | 100.0 | 84.4 | 65.8 | 60.4% | $1,957,198,749 | **80.78** | Eligible (Unselected) |
| 13 | **44** | Wood and articles of wood; wood charcoal... | 70.7 | 100.0 | 72.7 | 78.0 | 49.3% | $2,013,319,378 | **80.63** | Eligible (Unselected) |
| 14 | **31** | Fertilizers... | 75.6 | 100.0 | 78.5 | 39.0 | 74.8% | $3,049,100,227 | **79.85** | Eligible (Unselected) |
| 15 | **40** | Rubber and articles thereof... | 58.5 | 100.0 | 70.7 | 97.6 | 31.8% | $1,413,658,099 | **77.32** | Eligible (Unselected) |

---

## 9. FROZEN SELECTED SECTORS (MODELING SCOPE)

The following 12 sectors are formally frozen for the modeling milestones (M6–M10):

### TR → RU Direction (Türkiye Exports to Russia)
1. **HS 08** — Edible fruit and nuts; peel of citrus fruit or melons  
   - Final Score: `98.43` | Volume: `$9,426,138,591` | Coverage: `100.0%` | sMAPE: `34.0%`
2. **HS 84** — Nuclear reactors, boilers, machinery and mechanical appliances; parts thereof  
   - Final Score: `98.03` | Volume: `$11,331,507,857` | Coverage: `100.0%` | sMAPE: `38.6%`
3. **HS 39** — Plastics and articles thereof  
   - Final Score: `96.43` | Volume: `$4,058,065,836` | Coverage: `100.0%` | sMAPE: `34.5%`
4. **HS 87** — Vehicles other than railway or tramway rolling-stock, and parts and accessories thereof  
   - Final Score: `96.03` | Volume: `$6,478,746,694` | Coverage: `100.0%` | sMAPE: `39.0%`
5. **HS 73** — Articles of iron or steel  
   - Final Score: `93.25` | Volume: `$2,078,967,808` | Coverage: `100.0%` | sMAPE: `33.3%`
6. **HS 85** — Electrical machinery and equipment and parts thereof; sound/television recorders and reproducers  
   - Final Score: `93.21` | Volume: `$4,023,873,429` | Coverage: `100.0%` | sMAPE: `41.7%`

### RU → TR Direction (Türkiye Imports from Russia)
1. **HS 27** — Mineral fuels, mineral oils and products of their distillation; bituminous substances; mineral waxes  
   - Final Score: `99.02` | Volume: `$244,275,005,985` | Coverage: `100.0%` | sMAPE: `36.7%`
2. **HS 72** — Iron and steel  
   - Final Score: `98.15` | Volume: `$43,113,837,633` | Coverage: `100.0%` | sMAPE: `32.0%`
3. **HS 10** — Cereals  
   - Final Score: `94.83` | Volume: `$21,182,640,603` | Coverage: `100.0%` | sMAPE: `51.8%`
4. **HS 76** — Aluminium and articles thereof  
   - Final Score: `94.29` | Volume: `$12,821,599,197` | Coverage: `100.0%` | sMAPE: `38.2%`
5. **HS 23** — Residues and waste from the food industries; prepared animal fodder  
   - Final Score: `91.46` | Volume: `$5,245,108,836` | Coverage: `100.0%` | sMAPE: `33.2%`
6. **HS 74** — Copper and articles thereof  
   - Final Score: `90.05` | Volume: `$8,740,595,885` | Coverage: `100.0%` | sMAPE: `67.1%`

### Complete Modeling Scope
- **Bilateral Total Series:** 2 (`TR_RU_TOTAL`, `RU_TR_TOTAL`)
- **TR_RU Commodity Sectors:** 6
- **RU_TR Commodity Sectors:** 6
- **Total Modeled Series Count:** **14**
- **Forecast Horizons per Series:** 3 ($h=1, 3, 6$)
- **Total Forecast Targets Evaluated:** **42**

---

## 10. NOTABLE EXCLUSIONS & DISQUALIFICATION AUDIT

### Top Excluded Candidates by Volume:

#### TR → RU Exclusions:
- **HS 89** (Ships, boats and floating structures): $749,024,309 — *Disqualified*: Filter B (Coverage): Observed 110 months < minimum 141 (90%)
- **HS 99** (Commodities not specified according to kind / Special classification provisions / Confidential): $171,806,579 — *Disqualified*: Excluded: Commodities not specified according to kind, special classification provisions, or confidential trade.
- **HS 23** (Residues and waste from the food industries; prepared animal fodder): $118,291,126 — *Disqualified*: Filter B (Coverage): Observed 104 months < minimum 141 (90%)
- **HS 26** (Ores, slag and ash): $74,149,446 — *Disqualified*: Filter B (Coverage): Observed 112 months < minimum 141 (90%)
- **HS 47** (Pulp of wood or other fibrous cellulosic material; recovered paper or paperboard): $60,367,474 — *Disqualified*: Filter A (History): Observed 63 months < minimum 96; Filter B (Coverage): Observed 63 months < minimum 141 (90%)

#### RU → TR Exclusions:

- **HS 99** (Commodities not specified according to kind / Special classification provisions / Confidential): $14,836,293,518 — *Disqualified*: Excluded: Commodities not specified according to kind, special classification provisions, or confidential trade.; Filter A (History): Observed 35 months < minimum 96; Filter B (Coverage): Observed 35 months < minimum 141 (90%); Filter D (Recent activity): Recent observed 0/24 < minimum 21
- **HS 71** (Natural/cultured pearls, precious stones, precious metals, imitation jewellery; coin): $2,065,482,727 — *Disqualified*: Filter B (Coverage): Observed 107 months < minimum 141 (90%)
- **HS 79** (Zinc and articles thereof): $189,924,574 — *Disqualified*: Filter A (History): Observed 65 months < minimum 96; Filter B (Coverage): Observed 65 months < minimum 141 (90%); Filter D (Recent activity): Recent observed 8/24 < minimum 21
- **HS 25** (Salt; sulphur; earths and stone; plastering materials, lime and cement): $139,548,383 — *Disqualified*: Filter B (Coverage): Observed 119 months < minimum 141 (90%)
- **HS 89** (Ships, boats and floating structures): $73,822,241 — *Disqualified*: Filter A (History): Observed 80 months < minimum 96; Filter B (Coverage): Observed 80 months < minimum 141 (90%); Filter D (Recent activity): Recent observed 15/24 < minimum 21

---

## 11. ACCEPTANCE CRITERIA VERIFICATION

- [x] **v1.1 Target Mappings Verified:** TR_RU = Turkey export (792->643), RU_TR = Turkey import (792<-643).
- [x] **Russian Exporter Data Quarantine:** Russia export records (`trade_monthly_RU_TR.parquet`) rejected for RU_TR target.
- [x] **Strict Development Window:** Exclusively `2012-01` to `2024-12` utilized for sector selection.
- [x] **Holdout Leakage Guard Active:** 2025 data strictly isolated; zero influence on metrics or rankings.
- [x] **156-Month Panel Integrity:** Denominator strictly 156 months; missing values kept as `NaN` (never converted to 0).
- [x] **Deterministic Eligibility Filters:** Filters A through E applied programmatically.
- [x] **P10 Size Filter:** Computed direction-specifically on positive-volume development data only.
- [x] **Deterministic Multi-Factor Scoring:** Volume (40%), Continuity (30%), Recent (20%), Forecastability (10%).
- [x] **Maximum 6 Sectors per Direction:** Top 6 selected; no quota-filling with disqualified sectors.
- [x] **Bilateral Total Preserved:** Included in modeling scope outside sector selection.
- [x] **Frozen Manifest Version-Controlled:** Persisted to `configs/selected_sectors_v1_1.yaml`.
- [x] **Audit Outputs Generated:** Parquet metrics and JSON results persisted to `data/processed/`.

---

## 12. GATE VERDICT

```
================================================================================
MILESTONE M3: DATA AUDIT & SECTOR SELECTION
FINAL VERDICT: PASS
STATUS: FROZEN — READY FOR MILESTONE M4 (MACRO DATA FOUNDATION)
================================================================================
```
