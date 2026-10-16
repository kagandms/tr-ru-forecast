# M12 Acceptance Record

**Date:** 2026-10-16
**Status:** ACCEPTED for the requested read-only artifact-serving scope.

## Accepted scope

The user's M12 prompt explicitly selects a deterministic, read-only FastAPI layer over existing M10/M11, history, macro and mirror disk artifacts. This API is ready for the M13 dashboard contract; the earlier roadmap's future-model serialization and PostgreSQL layer remain separate unfinished work. No training, retuning, ingestion, artifact rewriting, database migration or production deployment was performed.

The API preserves **retrospective latest-vintage holdout evaluation under deterministic lag rules**. It exposes the frozen 33 SARIMA / 9 XGBoost assignments, all 42 task summaries and 168 CV candidates, 1,512 role predictions, and all 9,072 local SHAP records. It retains the raw negative SARIMA champion and governed XGBoost replacement, the negative unused fallback, and XGBoost nominal 80% coverage of 67.59%. SHAP availability and source vintage limits remain explicit.

## Evidence

- `make m12-verify`: Ruff passed; AST/type audit passed; **70 API tests passed, 0 warnings, 6.21 s**, all under `-W error`. Durable log: `docs/M12_TARGETED_VERIFICATION.log`.
- Tests compare every task's actual/raw/operational forecast and served interval values against persisted M11 rows; all 9,072 signed/local SHAP values and 756 global feature summaries are covered by artifact-backed assertions.
- Failure injection covers missing/corrupt bytes, omitted required files, incorrect scope, original-run checksum disagreement even after repinning the serving manifest, unknown catalog keys, invalid literals/dates/bounds, and restart after disk damage.
- Per-process immutable model records and frozen tuples retain deterministic cached responses. The request path does not reread/retrain or publish partial snapshots. Checksum status explicitly refers to load-time verification; restart is required to observe disk changes.
- Configurable artifact/config roots and frontend origins are tested; CORS does not allow credentials or wildcard origins. API data access is GET-only; invalid write attempts return structured 405.
- `docs/M12_OPENAPI.json` records the strongly typed response and discriminated explanation schemas. Contract, endpoint filters and run commands: `docs/M12_SERVING_API_REPORT.md`.
- Local ASGI benchmark: 30 warm serial requests per measured endpoint; startup 1.770 s, measured p95 0.653–2.311 ms. Details and response sizes: `docs/M12_LATENCY_BENCHMARK.json`. This is not a concurrent or network capacity guarantee.
- Local Uvicorn process: TCP health, sectors, leaderboard, governed forecast explorer, mirror and allowed-origin CORS smoke passed. Temporary process stopped after checks. Evidence: `docs/M12_HTTP_SMOKE.json`.

## Integrity and final verification

A pre-M12 recursive snapshot protects all 102 existing processed-data files, including M6–M11 artifacts and archived runs. A pytest session fixture independently hashes the real research tree before and after the suite; API tests use copied artifact roots. The final checksum comparison records that all 102 files match their pre-M12 hashes; the full suite passed 509 tests in 806.75 s with zero warnings or errors.

The serving manifest pins 20 input files, including the original run manifests, active and archived registry identities, history/macro and M2 mirror/QA bytes. Its checksum verification is consistency against a trusted operator-supplied manifest, not external authenticity or original-vintage evidence. A clean source clone needs an authorized mounted data bundle; no public redistribution was performed.

## Remaining product work

M13 frontend/browser UAT, a hosted API/deployment, production concurrency/load validation, PostgreSQL migrations, forward inference model serialization and full V1 release acceptance remain separate work. These limits do not leave the requested artifact-backed M12 API gate closed.
