# M14 Research Package

This package documents the empirical design and results of the TR–RU monthly trade forecasting project. It is a technical working paper and reproducibility record, not a peer-reviewed publication or a claim of real-time decision-time validation.

| Document | Purpose |
| --- | --- |
| [English working paper](working-paper-en.md) | Full research narrative, methods, results and limitations |
| [Turkish project summary](project-summary-tr.md) | Executive summary of the research and product |
| [Russian abstract](abstract-ru.md) | Russian-language research abstract |
| [Data dictionary](data-dictionary.md) | Dataset and field semantics, units, and missing-value rules |
| [Sources and licensing register](sources-and-licensing.md) | Dataset provenance, retrieval rules, attribution and rights status |
| [Reproduction guide](reproducibility.md) | Environment, local data requirements, commands and expected evidence |
| [Portfolio card](portfolio-card.md) | Concise, evidence-bounded project description for a portfolio |
| [Holdout task results](results/holdout-task-results.csv) | Generated 42-task derived M11 metrics; no raw monthly trade values |

The research numbers are generated from the local protected artifact `data/processed/m11_holdout_leaderboard.parquet` using `scripts/export_m14_evidence.py`. Raw research datasets remain outside Git. A clean source clone therefore needs an authorized local data bundle for full reproduction.

## Release boundary

The included sources register records the source-specific terms reviewed on 2026-10-18. It does not constitute legal advice or establish a permission grant. In particular, FRED's current terms describe restrictions on using FRED content in software or machine-learning development/training, and third-party series may have additional owner restrictions. Those rights must be resolved before claiming the macro-assisted modeling workflow is cleared for public or commercial reuse. UN Comtrade raw datasets are not included in this repository; their terms also govern redistribution and public use.

The holdout is **retrospective latest-vintage evaluation under deterministic lag rules**. Historical publication-time archives and macro vintage snapshots were not used. No causal macro effect, real-time forecast availability, or live model service is claimed.
