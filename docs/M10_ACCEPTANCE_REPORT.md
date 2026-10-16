# M10 Verification and Conditional Acceptance Record

**Date:** 2026-10-14
**Status:** M10 implementation, CV artifact reproduction and all 415 tests verified. Experiment acceptance remains conditional; the literal zero-warning criterion is not met.

## Implemented scope

The selector implements the authoritative v1.1 monthly-origin protocol across 14 series and horizons 1/3/6: 168 candidate entries, 15,736 observed fold rows, and 42 champion/fallback entries. It preserves sMAPE primary, MAE tie-break and the strictly <2% relative-improvement parsimony rule. The conflicting pasted five-fold/RMSE-first assumptions were not adopted.

| Model family | h1 wins | h3 wins | h6 wins | Total |
| --- | ---: | ---: | ---: | ---: |
| Seasonal Naive | 0 | 0 | 0 | 0 |
| SARIMA | 11 | 13 | 9 | 33 |
| SARIMAX | 0 | 0 | 0 | 0 |
| XGBoost | 3 | 1 | 5 | 9 |

Mean task-level CV sMAPE reduction against Seasonal Naive is 30.51%; median is 28.30%. These are development-selection results, not holdout results. Three tasks prefer the simpler model under the relative parsimony threshold: TR_RU:85 at h1/h3 and TR_RU:TOTAL at h3. The RU_TR:72 h1 SARIMA champion has a descriptive worst-origin stability flag; the flag does not override the frozen ranking rule. SARIMAX has 27 negative forecasts across 15 excluded tasks; XGBoost has one negative forecast in one excluded task. Scores remain visible; no predictions were clipped.

## Executed verification

- `make m10-select` generated the default registry, report, leaderboard, fold details, champion matrix and run manifest.
- `make m10-select M10_OUTPUT_DIR=/tmp/tr-ru-m10-final-reproduction` independently reproduced all 42 champion/fallback records, all consumed input hashes and all four data-output hashes exactly. Run timestamps and Git dirty-path metadata may differ.
- Recomputed every default run-manifest output checksum, every registry input checksum, and the current evaluation/source-provenance fingerprint; all matched.
- Independently compared all 168 recomputed sMAPE/MAE/RMSE groups to the stored M6–M9 metric artifacts at `rtol=1e-12, atol=1e-8`; all matched, including descriptive metrics for ineligible candidates.
- Latest locked-environment targeted run: `/tmp/tr-ru-m10-locked-venv/bin/pytest tests/unit/test_champion_selector.py -q -o addopts=''` — **51 passed** in 3.13s, one existing Starlette/httpx deprecation warning.
- `.venv/bin/ruff check .` and `/tmp/tr-ru-m10-locked-venv/bin/ruff check .` — passed.
- `uv lock --check` — passed.
- AST inspection of the four new production modules found no function over 30 lines; signatures are typed and no `Any` or type-ignore directives were introduced. No mypy target/configuration exists in this repository, so no mypy pass is claimed.
- Full-suite command: `PYTHONPATH=src:. /tmp/tr-ru-m10-locked-venv/bin/pytest -q -o addopts=''`, with output in `/tmp/tr-ru-m10-verified-pytest.log` and processed artifacts isolated by the existing pytest session fixture. The environment was installed with `UV_PROJECT_ENVIRONMENT=/tmp/tr-ru-m10-locked-venv uv sync --locked --extra dev --python 3.13`. **415 passed, 0 failed, 1 existing Starlette/httpx deprecation warning in 795.54s (13m15s).**

The M10 tests cover aggregation and pooled RMSE, relative skill and zero baseline, exact 2% boundaries, MAE/RMSE/stability/complexity ordering, parsimony, diagnostic worst-fold flags, failed/missing/negative/NaN/Inf predictions, deterministic ordering, fallback selection, finite registry schema and exact coverage, output immutability, checksum mismatch rejection, CV path allowlisting, and calendar rejection before prediction values are loaded. An independent pipeline replay guards parquet reads against any raw target, macro or feature dataset access.

## Open acceptance boundaries

1. `docs/M9_ACCEPTANCE_REPORT.md` still records open A2 code-structure findings. The user's direct M10 implementation request authorized this scope; it did not establish that A2 was resolved.
2. M8 run metadata records a validation YAML hash that differs from the current frozen manifest. M10 independently verifies the exact persisted calendar against the canonical 281 monthly folds, and records the discrepancy in registry provenance. The historical hash discrepancy remains unexplained; no M8 metadata was rewritten.
3. The existing Starlette/httpx warning prevents a literal zero-warning verdict. Dependencies and warning filters were not changed to conceal it.
4. No raw holdout targets were opened or scored by M10. This does not prove that earlier project workflows never accessed 2025 data. Historical macro vintage fidelity and real-world trade decision-time availability remain unestablished.
5. M11 remains gated. Holdout-origin/refit geometry, decision-time rules and interval calibration availability must be frozen separately before its evaluation. Serving/API/frontend work is outside this milestone.

## Earlier milestone boundary-test maintenance

The first full-suite run failed three project-wide absence assertions after M10 outputs were generated (412 passed, 3 failed). The fresh locked run additionally exposed the same stale boundary in M5; M7 had the same assertion and was found by a complete search. These five M5–M9 tests now snapshot later milestone reports before earlier pipelines run and assert their content hashes remain unchanged, whether the reports originally exist or are absent. This preserves the no-cross-milestone-write contract without requiring M10 to be absent forever. No numerical, leakage, convergence or model-quality assertion was weakened. The final full-suite rerun passed all 415 tests. The complete original 364-test set remains present alongside the 51 new M10 tests.

## Stage A / M11 follow-up — 2026-10-14

M9 A2 is resolved and the Starlette/httpx warning is fixed at its dependency source, without suppression. M8's historical YAML hash is unchanged; the gate accepts an explicit non-champion equivalence rule only after exact 281-origin calendar alignment, 3,934 unique series/origin keys and equality of scoring actuals to M6 are verified. SARIMAX has zero champion assignments; its independently refitted fallback slots retain frozen configurations.

The complete original registry bytes are preserved at `data/processed/m11_original_champion_registry_v1_1.yaml` and verified against the original M10 run-manifest registry checksum. The active registry status is now `CV_SELECTION_FROZEN_M11_READY`; all 42 champion/fallback records and CV metrics are unchanged. `configs/m11_gate_v1_1.yaml` stores both original and active content identities and the champion-decision hash. Original M10 run hashes remain historical evidence; they are not rewritten to pretend that the status amendment existed in the initial run.

Before scoring, the user authorized retrospective latest-vintage holdout evaluation under deterministic lag rules, monthly expanding refits and CV-seeded chronological residual updates. Experiment Protocol §19 and `configs/holdout_spec_v1_1.yaml` freeze that amendment. No original publication/vintage archive is claimed. M11 artifacts and validation are recorded in `M11_FINAL_HOLDOUT_REPORT.md` and `M11_ACCEPTANCE_REPORT.md`.
