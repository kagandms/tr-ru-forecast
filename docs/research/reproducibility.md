# Reproduction Guide

## What can be reproduced from a clean clone

The source repository contains code, configurations, contracts, test fixtures and frozen documentation, but large research datasets under `data/raw/`, `data/interim/` and `data/processed/` are gitignored. A clean clone alone cannot reproduce M3–M11 results. Full reproduction requires an authorized local data bundle and the exact protected artifacts listed by the M10/M11 run manifests. This package does not redistribute those data.

The dashboard's static snapshot can be built without the local Python API when its verified snapshot is present. That is a product-build path, not a full research reproduction.

## Environment and integrity checks

Use the project's locked Python environment and frontend lockfile. Do not place credentials in command arguments or reports. Use the project root as the working directory.

```bash
uv sync --locked --all-extras
uv lock --check
```

Mount the authorized local dataset into the expected `data/` paths, then verify protected serving inputs and provenance before running experiments:

```bash
make m12-verify
```

The M12 manifest checks byte consistency against the operator-supplied trusted manifest; it is not an external authenticity proof or historical source-vintage proof.

## Reproduce selection, holdout and evidence table

M10 reads allowlisted development CV artifacts and writes to a fresh output path because experiment outputs are immutable by default:

```bash
make m10-select M10_OUTPUT_DIR=/tmp/m10-reproduction
```

M11 evaluates the frozen registry and writes to a separate fresh directory:

```bash
make m11-holdout M11_OUTPUT_DIR=/tmp/m11-reproduction
```

The M14 task table is derived from the accepted local M11 leaderboard, keeping task-level raw champion and operational scores separate:

```bash
./.venv/bin/python scripts/export_m14_evidence.py
```

This writes `docs/research/results/holdout-task-results.csv`. It does not copy monthly target observations. Run the focused exporter contract tests with:

```bash
./.venv/bin/pytest tests/unit/test_export_m14_evidence.py -q -W error -o addopts=''
```

For independent-run comparison, compare decision and numerical artifacts separately from run timestamps, report timestamps and Git dirty-state metadata. M11's accepted replay showed byte-identical numerical data artifact hashes; the generated Markdown report includes a run timestamp and may differ byte-for-byte.

## Validate the published claims

The main text's aggregate numbers come from `docs/M11_ACCEPTANCE_REPORT.md` and `docs/M11_FINAL_HOLDOUT_REPORT.md`. The per-task appendix is generated from `data/processed/m11_holdout_leaderboard.parquet`. M10 champion assignments and the 33 SARIMA / 9 XGBoost counts come from the frozen registry and `docs/M10_CHAMPION_SELECTION_REPORT.md`.

The evaluation description must remain “retrospective latest-vintage holdout evaluation under deterministic lag rules.” Do not call it real-time or historical-vintage replay. Do not infer macro causality from SHAP. Do not rewrite a champion assignment because the final holdout looks better for a fallback.

## Rebuild product artifacts

With the protected local data bundle mounted, use the documented project targets for API verification and static snapshot export:

```bash
make m12-verify
make m13-export-snapshot
```

Then run the repository's required Python checks and, from `frontend/`, the locked frontend type check and production build. Report these as local build evidence; they do not establish logged-in user acceptance.

## Rights and reproducibility boundary

The source/licensing register records a current unresolved FRED terms issue for the macro-assisted ML pipeline. Reproducing the existing artifacts does not resolve source permissions. Any source replacement or authorized alternative changes the research inputs and requires a newly documented, frozen experiment and new evaluation evidence; do not silently substitute a series.
