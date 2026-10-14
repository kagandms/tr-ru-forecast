# M9 XGBoost Acceptance Record

**Status:** Model run and reproducibility evidence verified; overall M9 closure remains open on A2 code-structure findings. This is a development-period validation run, not an M10 champion decision or final holdout result.

## Accepted run evidence

- Run ID: `run_m9_xgboost_v1_1_72c6bf10bc56`
- Output directory: `data/processed/m9_acceptance_2026-10-14_r2/`
- Git commit: `9d539b8124cff22bc56b32d34b211ce087b8ec9b` (working tree dirty; exact changed paths recorded in run metadata)
- M9 source fingerprint: `586d27f8cb7fe680a9a9617061d3a6119c73d01cd2bb51aadc7008a78f548eb3`
- Run metadata SHA-256: `d289d0ca658032a94bd97931251161619fcc88ba6030863f68da8e8549d7fb93`
- Artifact dimensions: 48 tuning rows; 3,934 predictions; 42 metric groups; 3,934 interval audit rows with 3,038 enabled; 82,614 local SHAP rows; 882 global SHAP rows; 0 outer-fit failures.
- Recomputed all eight recorded output artifact checksums; all matched. Interval nesting had zero violations.
- Empirical coverage remains below nominal: h1 65.5%/91.0%, h3 62.0%/89.3%, h6 62.9%/87.5% for nominal 80%/95% intervals.

## Scope and review findings

The uncommitted M9 implementation adds direct XGBoost models for horizons 1, 3, and 6 months, pre-validation tuning, prequential residual intervals, fold-local SHAP attribution, a specification manifest, and a report generator. The review found the following acceptance risks:

1. Run metadata used a hardcoded M8 commit SHA, so it did not identify the M9 source tree.
2. Missing files could produce an empty checksum in an earlier implementation path, and a missing frozen fold artifact could fall back to regenerated folds.
3. Default run IDs were reused, and output files lacked recorded checksums.
4. The model report claimed stronger 2025 quarantine and mutation-test evidence than the executed tests established. The full-suite checkbox was also generated as if it were universally true.
5. Tests could write generated artifacts into the local research output directory.
6. The `httpx` package is imported by the production Comtrade client but had been declared only in the development extra.
7. M4, M8, M9, and README status text did not describe the same current milestone.

The implementation now fails on missing required inputs, requires the frozen validation fold artifact, assigns unique run IDs, records the actual Git commit and dirty paths, fingerprints the M9 source files, records input/output SHA-256 values, and isolates test output under pytest temporary storage. The M9 report now identifies the run as development validation and narrows its 2025 statement to the development artifact actually used. Documentation corrections and their validation are tracked below.

The code-style audit remains open. An AST count found ten functions over the repository's 30-line limit: `calibrate` (87), `explain_fold_prediction` (64), `compute_global_shap_importance` (31), `generate_m9_report` (389), `get_training_subset` (47), `generate_xgboost_manifest` (92), `aggregate_m9_group_metrics` (58), `run_m9_xgboost_pipeline` (372), `evaluate_tuning_candidate` (92), and `run_pre_validation_tuning` (62). ML records also use loosely typed dictionaries and `Any`. These are maintainability findings in A2; they must be refactored and revalidated before the checklist can say M9 acceptance is fully closed.

## Data vintage and decision-time limits

The M4/M5 records show latest-vintage macro series with approximate conservative release dates, and the M5 feature snapshot is a calendar month-end cutoff. The available files do not establish historical vintage fidelity, exact publication timestamps for every observation, or the date on which a trade target would have been actionable. Accordingly, M9 is interpreted as a retrospective development-period validation over the frozen feature artifact. It makes no real-time forecasting claim. The 2025 feature/target data are not scored by this M9 run; this record does not claim that no earlier workflow ever accessed 2025.

## Verification evidence

- [x] M9 targeted tests pass: `pytest tests/unit/test_m9_provenance.py tests/unit/test_xgboost.py -q -o addopts=''` — 40 passed in 192.52s; one Starlette/httpx deprecation warning.
- [x] Full Python suite passes with processed artifacts isolated: `PYTHONPATH=src:. /tmp/tr-ru-m9-prod-venv/bin/pytest -q -o addopts=''` — 364 passed in 820.29s; one Starlette/httpx deprecation warning.
- [x] `uv sync --locked --no-dev` installed 42 packages and the locked dev extra installed 5 more in a fresh Python 3.13 environment; `ComtradeClient` and runtime `httpx 0.28.1` imported successfully.
- [x] Ruff and `uv lock --check` pass after final edits.
- [x] A fresh M9 run produced 14 series, 281 folds, 3,934 predictions, 42 metric groups, 3,038 interval-enabled rows, 82,614 local SHAP rows, and 882 global SHAP rows with zero fit failures.
- [x] Run metadata input and output checksums recompute exactly; interval nesting has zero violations; fold-local SHAP execution completed without additivity failures.
- [x] Generated M9 report points to the fresh run and reports empirical interval coverage below nominal honestly.

## Remaining acceptance boundary

This record does not approve champion selection, open the 2025 holdout, claim external review, establish point-in-time macro vintages, or establish a real-time trade decision workflow. A2 remains the only unchecked M9 task; M10 is still gated.
