# M14 Research and Portfolio Packaging Acceptance Record

**Updated:** 2026-10-18
**Status:** Research-documentation deliverables prepared; full M14 acceptance remains open pending source-rights review and portfolio-media items.

## Delivered in this update

- [English technical working paper](research/working-paper-en.md) with the 15 frozen sections, source-grounded references, M10/M11 results and explicit epistemic limits.
- [Turkish project summary](research/project-summary-tr.md) and [Russian abstract](research/abstract-ru.md), consistent with the English report's metrics and scope.
- [Data dictionary](research/data-dictionary.md) documenting source, mirror, macro, feature, holdout and SHAP semantics, units and null behavior.
- [Sources and licensing register](research/sources-and-licensing.md) distinguishing data origin from distributor, documenting retrieval provenance and current public terms reviewed on 2026-10-18.
- [Reproduction guide](research/reproducibility.md) explaining the protected local-data requirement, M10/M11 commands, artifact checks and evaluation limits.
- [Portfolio card](research/portfolio-card.md) with verified claims, live demo, stack and explicit limits.
- `scripts/export_m14_evidence.py` and [42-task holdout CSV](research/results/holdout-task-results.csv), derived from the local M11 task leaderboard without copying monthly source observations.
- README, milestone and completion-plan links/status updated to point at the delivered package.

## Source-rights gate

The sources register identifies a material unresolved rights question: FRED's currently published terms restrict use of FRED content in connection with software or machine-learning development/training, while three configured macro series and the historical segment of the linked Russian industrial-production series are retrieved through FRED. No permission covering this use is present in the repository. Before claiming broad public or commercial clearance, obtain written permission for the exact series/use or replace affected inputs through an authorized source and freeze/re-run the affected research pipeline. Exact reuse conditions for the Rosstat workbook also need a source URL and file-specific terms record.

Raw UN Comtrade and macro datasets remain gitignored and were not added to this documentation package. The source register distinguishes public analytical outputs from bulk source-data redistribution and records the applicable provider terms; it does not claim to grant permission.

## Evidence and checks

- `.venv/bin/python scripts/export_m14_evidence.py` generated 42 task rows plus a header from `data/processed/m11_holdout_leaderboard.parquet`.
- `.venv/bin/pytest tests/unit/test_export_m14_evidence.py -q -W error -o addopts=''` — **5 passed**.
- `.venv/bin/pytest -q -W error -o addopts=''` — **516 passed, 0 warnings/errors** in 840.05 seconds.
- `.venv/bin/ruff check .` and `git diff --check` — passed.
- Repeated evidence export produced the same CSV bytes; SHA-256: `949bdc0febb034c4dd0105722bef1b2561917543eda905105ed0db32b4d0b186`.
- The reported operational sMAPE, Seasonal Naive sMAPE, champion counts and interval coverage match `docs/M11_ACCEPTANCE_REPORT.md` and `docs/M11_FINAL_HOLDOUT_REPORT.md`.
- Research text labels the holdout as **retrospective latest-vintage evaluation under deterministic lag rules**. It does not claim original-vintage replay, real-time availability, causal macro effects, or statistical significance.
- No full research experiment or source refresh was run as part of this documentation change; generated reporting rows use the already accepted local M11 artifact.

## Remaining to close M14

- Resolve FRED permission for software/ML use or replace the restricted inputs and rerun the affected experiments; retain exact Rosstat workbook provenance and rights evidence.
- Capture and accept demo screenshots/video for the portfolio package.

M15 user-acceptance and remaining final-release criteria remain a separate milestone; the frontend production deployment is already documented in the release history.
