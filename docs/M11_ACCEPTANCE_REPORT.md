# M11 Final Holdout Acceptance Record

**Evaluation date:** 2026-10-14
**Delivery verification:** 2026-10-16
**Status:** ACCEPTED within the explicitly authorized retrospective scope. M9 A2 and pre-M11 engineering gates are closed; M12 remains a separate milestone.

## Scope and pre-score freeze

This is **retrospective latest-vintage holdout evaluation under deterministic lag rules**. The user explicitly chose the existing local processed datasets instead of external publication-time/vintage archives. Experiment Protocol §19 and `configs/holdout_spec_v1_1.yaml` were recorded before the first holdout scoring.

The experiment uses twelve 2025 targets for each of the 42 sector/horizon tasks, with monthly origin = target minus h1/h3/h6. Refit data are cut off at the origin; trade is assumed available at reference-month end with zero added lag, and macro observations use the frozen approximate release-date/as-of rules. Original-vintage fidelity and real-time actionable forecasting are not established. Early h3/h6 origins before the 2024-12 selection freeze are explicitly retrospective reconstructions using subsequently frozen model assignments.

The frozen registry still assigns 33 SARIMA and 9 XGBoost champions. No sector, feature, family, order, tuned parameter, nominal interval level or model assignment was reselected using holdout outcomes. CV-seeded prequential residual updates admit only same-task/model residuals with target month <= origin. XGBoost's observed minimum calibration pool was 86, above the frozen minimum of 19. Seasonal Naive intervals remain explicitly unavailable.

## Stage A closure

- All M9 ML functions are <=30 lines, signatures are annotated, and the ML package contains no `Any` or `type: ignore`. Typed forecast/task dataclasses and residual records carry model state. Existing public call contracts remain compatible.
- M9 targeted validation: 40 passed, zero warnings in 194.65 seconds. A complete frozen-parameter outer-CV replay exactly matched the accepted rerun's 3,934 prediction rows, 3,934 interval-audit rows, 82,614 local SHAP rows and 3,934 fit-log rows, using `assert_frame_equal(check_exact=True)`. Original accepted artifacts were preserved.
- The Starlette warning was fixed by adding the supported `httpx2` test backend to the development extra, while retaining runtime Comtrade `httpx`. The pre-existing blanket DeprecationWarning filter was removed. API and subsequent checks run with `-W error`. See [Starlette TestClient documentation](https://starlette.dev/testclient/).
- M8's original YAML hash discrepancy is retained, not rewritten. The accepted non-champion gate rule verifies the exact 281-origin calendar, 3,934 unique task/origin keys and exact scoring-target equality against M6. SARIMAX has zero champion assignments; its fallback slots are independently refitted from frozen configurations.
- Original M10 registry bytes are archived at `data/processed/m11_original_champion_registry_v1_1.yaml` and verified against the original M10 run-manifest checksum. Only readiness status changed in the active registry, now `CV_SELECTION_FROZEN_M11_READY`. All champion/fallback/CV records are unchanged. The gate records both content identities and the champion-decision hash.
- Earlier milestone boundary tests preserve later reports/data by content hashes instead of requiring those artifacts to remain absent forever.

## Final outputs and observed results

The default `data/processed/` package contains:

| Artifact | Rows / purpose |
| --- | --- |
| `m11_holdout_predictions.parquet` | 1,512 role rows = 42 tasks × 12 targets × 3 roles |
| `m11_holdout_leaderboard.parquet` and `.csv` | Complete 42-task CV/holdout matrix |
| `m11_holdout_fit_log.parquet` | 1,512 role audit rows, including optimizer, retry and warning category |
| `m11_shap_values.parquet` | 9,072 local attributions for 432 unique XGBoost fits |
| `m11_shap_summary.parquet` | 756 per-task/feature signed and absolute summaries |
| `m11_holdout_run.json` | Exact input/output SHA-256 values, source fingerprint, frozen scope and run metadata |
| `M11_FINAL_HOLDOUT_REPORT.md` | Generated report; identical copy under `docs/` |

| Evaluation view | Valid origins | Mean sMAPE |
| --- | ---: | ---: |
| Champion, valid primary forecasts | 503 / 504 | 26.2223% |
| Registered fallback benchmark | 503 / 504 | 25.7190% |
| Seasonal Naive | 504 / 504 | 37.4965% |
| Governed operational champion/fallback | 504 / 504 | 26.1704% |

These views have explicit denominators. Raw champion sMAPE including its finite negative forecast is 26.5671%, stored separately. The fallback benchmark evaluates every registered fallback; it is not a retroactive replacement champion. No assignment was changed because its fallback performed better in this holdout.

One SARIMA champion forecast was negative: `RU_TR:10`, h3, origin 2025-04, target 2025-07. Its raw value is retained, and the pre-registered XGBoost fallback supplied the operational forecast. One SARIMA fallback forecast was also negative (`RU_TR:10`, h6, origin 2025-01); it remained invalid and did not replace the valid XGBoost champion. All 540 classical fits used the frozen primary lbfgs optimizer with zero retries; categorical estimation warnings, where recorded, remain in the fit audit.

Primary valid-interval empirical coverage is 86.8787% at nominal 80% and 98.8072% at nominal 95%, with 503 interval origins. This aggregate conceals heterogeneity: XGBoost champions' 80% coverage is 67.5926% (73/108), below nominal, while their 95% coverage is 97.2222% (105/108). No interval tuning followed these outcomes. Native/conformal methods, unavailable baseline intervals, widths and denominators are reported separately in the full report.

Thirty-six of 42 champions have lower task-level mean valid-origin sMAPE than the full Seasonal Naive benchmark. Mean task-level CV-to-holdout drift is -0.9955 percentage points; primary skill uses paired valid origins when failures occur. These are descriptive findings, not statistical significance or causal claims.

## Executed verification

- `make m11-holdout` generated the complete default package. Independent fresh-directory make-target runs reproduced all six numerical data-artifact hashes, all frozen input hashes and the source fingerprint; timestamps/report hashes may differ because run time is reported.
- Delivery replay: `make m11-holdout M11_OUTPUT_DIR=data/processed/archive_m11_repro_2026-10-16` completed on 2026-10-16. Persistent evidence: `data/processed/m11_delivery_reproduction_2026-10-16.log` and `data/processed/m11_delivery_verification_2026-10-16.json`. Six numerical artifacts matched byte-for-byte; the seventh output is the Markdown report, whose timestamp changes between runs.
- `pytest -q -W error -o addopts=''` — **441 passed, zero warnings/errors, 784.64 seconds**, executed on 2026-10-14.
- The only subsequent production changes were documentation, a private formatter name and a documented public compatibility forwarder. The underlying M9 orchestrator AST was asserted unchanged before/after the rename. API/provenance follow-up: **5 passed**, zero warnings, 1.62 seconds. The final holdout package was then regenerated from frozen inputs; all numerical values, intervals and fallback decisions remained unchanged.
- Delivery-date follow-up: `.venv/bin/pytest tests/unit/test_holdout_runner.py tests/unit/test_m9_provenance.py tests/api -q -W error -o addopts=''` — **31 passed, zero warnings, 6.90 seconds** on 2026-10-16. Persistent log: `data/processed/m11_delivery_targeted_2026-10-16.log`.
- `.venv/bin/ruff check .`, `uv lock --check`, `git diff --check` and AST length/annotation checks passed on delivery verification.
- Recomputed every current default input/output checksum, the docs report checksum and the current source fingerprint; all matched.
- The first package was preserved in `archive_m11_initial_package/`; audit enrichment added fields only. All 1,512 original predictions, actuals, residuals, bounds and fallback decisions were asserted exactly equal to the enriched package. The validated package before the documentation-only update is retained separately in `archive_m11_validated_package_before_doc_only_update/`.

The full-suite temporary log from October 14 is no longer retained after the interrupted session; its executed tool result and durable Stage A verification record preserve the result. The delivery follow-up and independent replay logs are stored under processed data. No external deployment, production API integration or serving-model serialization is claimed by this milestone.
