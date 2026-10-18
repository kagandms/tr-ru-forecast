# Sources, Provenance and Licensing Register

**Terms review date:** 2026-10-18. Provider terms can change; recheck before each public or commercial release. This register records source lineage and current findings; it is not legal advice or a license grant.

## Trade data

| Use | Source and identifiers | Retrieval/provenance | Rights and publication status |
| --- | --- | --- | --- |
| Forecast targets and paired reporter mirrors | UN Comtrade; authenticated subscription access; monthly goods data; Türkiye reporter 792, Russian partner 643, HS chapters and total flows. | `docs/PRE_M3_FULL_HISTORY_VERIFICATION_REPORT.md` records API v1, authenticated access, requested dates, HTTP outcomes, record counts and retrieval audit. `DATA_SPEC.md` defines flow semantics. Every stored observation carries source and retrieval metadata. Credentials are local only and are never included in documents. | UN Comtrade's current data-use policy says the data are for internal use unless a redistribution exception/permission applies. Its current FAQ describes a limited-volume threshold and distinguishes transformed outputs from format-only changes; public visualization is treated separately from bulk data/API redistribution. Do not place the raw subscribed bundle in Git or a public downloadable data package. The dashboard and report use derived analysis; confirm project-specific treatment if reproducing extensive source values. Cite “UN Comtrade.” See [data-use policy](https://uncomtrade.org/docs/policy-on-comtrade-data-use/), [redistribution policy](https://uncomtrade.org/docs/re-dissemination-of-data/) and [license agreement](https://comtradeplus.un.org/LicenseAgreement). |

The official data source is UN Comtrade's distribution of national reporter records. The repository does not contain separately acquired, independently verified TurkStat and Rosstat bilateral trade datasets. The mirror records must not be described as such.

## Macroeconomic sources

| Indicator | Origin / distributor / configured series | Frequency, unit and configured availability | Rights review |
| --- | --- | --- | --- |
| `usd_try` | OECD / FRED; `CCUSMA02TRM618N` | Monthly average; TRY per USD; approximate availability first day of following month | FRED terms state series may carry third-party restrictions and the current service terms restrict use of FRED content in software or machine-learning development/training. Obtain written permission or replace this input via an authorized source and rerun affected development/holdout evidence before declaring the macro-assisted model rights-cleared. |
| `rub_usd` | OECD / FRED; `CCUSMA02RUM618N` | Monthly average; RUB per USD; approximate availability first day of following month | Same FRED restriction/permission gate as above; verify series-level owner terms and intended publication use. |
| `brent` | IMF / FRED; `POILBREUSDM` | Monthly average; USD per barrel; approximate availability first day of following month | Same FRED restriction/permission gate as above; FRED redistribution does not override IMF/source-owner rights. |
| `tr_ip` | TÜİK / Eurostat STS; dataset `sts_inpr_m`, filters `geo=TR`, `unit=I21`, `nace_r2=B-D`, `s_adj=SCA` | Monthly seasonally adjusted index, 2021=100; configured availability on the 15th of M+2 (approximate conservative) | Eurostat authorizes reuse of many statistical products with attribution and disclosure of changes, subject to item-specific and third-party exceptions. Verify the individual dataset's applicable conditions and cite Eurostat/TÜİK lineage. See [Eurostat reuse notice](https://ec.europa.eu/eurostat/help/copyright-notice). |
| `ru_ip` | OECD/FRED series `RUSPROINDMISMEI` through 2021-12, linked to a Rosstat JDemetra+ seasonal-adjusted spreadsheet series from 2022-01; configured composite id `RUSPROINDMISMEI_LINKED_ROSSTAT` | Monthly index linked to 2015=100 using the frozen 24-month overlap factor `1.2128086297589498`; approximate availability on the 15th of M+2 | Historical FRED segment has the FRED/source-owner restriction above. The recent Rosstat file's exact web URL and file-specific reuse notice are not present in the frozen registry; verify and archive those before redistributing the source or claiming clearance. Rosstat's open-data page describes reuse conditions for datasets published in its open-data catalog; that does not by itself prove this exact workbook is covered. |

### FRED use gate

The configured project downloads several series through FRED's CSV graph endpoints and uses them in macro features. FRED's currently published terms say FRED content is owned by FRED or third parties, that series-specific owner conditions apply, and that use of FRED Services/Content in connection with software or machine-learning development/training is restricted. This is a material unresolved rights question for the existing macro-assisted model and any derived public research package. The data being accessible without a login does not establish permission for this use. Before representing the workflow as cleared, either obtain written authorization that covers the exact series and use, or replace the affected inputs with an authorized source and rerun every affected feature, model-selection, holdout, SHAP and snapshot artifact under a newly frozen protocol. FRED terms reviewed: [legal notices and terms](https://fred.stlouisfed.org/legal/fraud-awareness) and [API terms](https://fred.stlouisfed.org/docs/api/terms_of_use.html).

### Attribution and redistribution controls

- Cite source institution, distribution channel, series/dataset ID, transformation and retrieval/access date in every publication.
- Keep the access-controlled raw bundle and subscription credential out of Git, the static frontend and downloadable release artifacts.
- Do not treat a file-format conversion, filtering, rounding or column rename as a license-clearing transformation.
- Mark the derived holdout table as analytical output; do not include source monthly trade or macro observations in it.
- Recheck source terms and individual series copyright/status before commercial use, public bulk downloads or API redistribution.
- Preserve the exact source URL/endpoint, source identifier, access date, local file hash, transformation, license URL and permission evidence in the next source snapshot manifest. For the Rosstat workbook, first restore the missing exact retrieval URL and file-specific terms record.

## Research references and methods sources

References supporting trade-statistics interpretation and forecasting methods are listed in the [working paper](working-paper-en.md#references). Current provider rights notices are separately linked above because their terms may change independently of academic references.
