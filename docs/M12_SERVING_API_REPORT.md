# M12 Frozen Artifact Serving API

## Scope and decisions

The explicitly requested M12 delivers a read-only artifact API for the M13 dashboard. It serves the frozen retrospective latest-vintage deterministic-lag M10/M11 evaluation, not new forward forecasts. The user's M12 prompt selects disk-backed serving for this milestone; model serialization/refitting, database migrations and M13 UI remain separate work.

- Repository → service → router separates verified disk access, domain projection and HTTP validation. Per-process immutable Pydantic snapshots cache responses; a lock coordinates first load. Restart publishes a new verified snapshot. No request writes artifacts or runs training/ingestion.
- Assumptions: one small frozen snapshot, public read-only data, bounded catalog (14 series/42 tasks), no secrets in responses, no authentication requirement in this milestone. Use exact-origin configurable CORS without credentials. Benchmark local warm responses; no production throughput claim.
- A serving manifest pins all served files, including M10/M11 manifests, registry, history, macro, mirror and QA. Original M10/M11 checksums are cross-checked without rewriting them. Historical M10 registry hash refers to its archived pre-readiness bytes. Mirror files have no original M2 checksum manifest; M12 explicitly pins their current bytes.
- Missing/corrupt inputs fail closed: degraded health and structured 503 data responses. Unknown catalog keys return 404; invalid horizons/months return 422. Non-finite optional diagnostics become null.
- Frozen forecasts retain raw negative champions and registered fallback operational values/intervals. All nominal and empirical interval information remains distinct. No synthetic SHAP for tasks without an XGBoost role.
- Mirror canonical columns retain M2 exporter-minus-importer semantics. RU_TR's forecast target is the importer column, identified explicitly. National reporters arrive through UN Comtrade; independent TURKSTAT/Rosstat datasets are not fabricated. Post-2022 paired gaps and causal/CIF-FOB adjustments are unavailable when the source cannot support them.

Alternatives: request-time pandas/disk reads waste I/O and risk inconsistent snapshots; a database adds migrations and publication complexity outside the explicit disk-artifact scope. Immutable per-process projections fit this bounded dataset.

Implementation references: [FastAPI router composition](https://fastapi.tiangolo.com/tutorial/bigger-applications/), [lifespan](https://fastapi.tiangolo.com/advanced/events/), [Pydantic serialization](https://docs.pydantic.dev/latest/concepts/serialization/).

Contract, verification and measured latency sections will be completed from executed checks.

## Endpoint contract

All data routes are GET-only under `/api/v1`. `/` retains the existing service identity. `/health` preserves the previous fields (`status`, `timestamp`, `version`, `environment`) and adds the same readiness/provenance contract as `/api/v1/health`.

| Endpoint | Filters | Response |
| --- | --- | --- |
| `/health`, `/api/v1/health` | none | `HealthResponse`: spec/scope, verified 20-file snapshot, M10/M11 commit/source/manifest digests, readiness, 42/1512/9072 counts |
| `/api/v1/sectors` | none | `SectorCatalog`: 14 source-labelled sectors, three assignments each, historical range/latest observation |
| `/api/v1/leaderboard`, `/metrics` | direction, hs2, horizon, champion_model | `LeaderboardResponse`: global equal-task KPIs plus filtered 42 task comparisons; 168 CV candidate records nested under their task |
| `/api/v1/forecasts/{sector_id}` | optional horizon | `ForecastResponse`: 168 historical observations and twelve 2025 points per selected task, raw/operational/fallback/baseline values and served intervals |
| `/api/v1/forecasts` | sector_id, optional horizon | Query-form equivalent of the forecast explorer |
| `/api/v1/explainability/{sector_id}` | required horizon, optional target_month | Discriminated available/unavailable response; 21 global features and twelve local waterfalls for each of 36 XGBoost-role tasks |
| `/api/v1/mirror-analytics`, `/mirror` | sector_id, direction, hs2, page/date bounds | `MirrorResponse`: persisted M2 gaps, paired/unpaired counts, pre/post-2022 diagnostics and explicit source/valuation limits |
| `/api/v1/trade` | required sector_id, page/date bounds | `TradeResponse`: official target history with reporter/flow/retrieval metadata |
| `/api/v1/indicators` | optional indicator_id, page/date bounds | `IndicatorsResponse`: macro series with source and approximate publication-lag metadata |
| `/api/v1/overview` | none | `OverviewResponse`: aligned latest TOTAL exports/imports, bilateral total and Türkiye balance; no sector double-counting |
| `/api/v1/methodology` | none | `MethodologyResponse`: target policy, evaluation limits, units, serialization and manifest identities |

Pagination uses `offset >= 0`, `1 <= limit <= 10000`, default 1000. Date bounds are inclusive `YYYY-MM` values; reversed bounds return 422. `sector_id` accepts `RU_TR:10` or `RU_TR__10`. Valid horizons are 1/3/6; SHAP target months must be in 2025. Global SHAP importance remains across all twelve origins when filtering a local waterfall. Leaderboard KPIs remain global when filtering task rows. No rows found for a valid catalog intersection return an empty selection; an unknown sector/HS2/indicator returns 404. Undeclared query keys return 422.

`ErrorResponse` is `{ "error": { "code": "...", "message": "..." } }`. Invalid parameters produce `invalid_request` (422), unknown keys `catalog_not_found` (404), missing/corrupt snapshots `artifact_snapshot_unavailable` (503). Unhandled failures are logged server-side and return a redacted `internal_error` (500). Framework 404/405 responses use the same envelope. Health failure returns its typed degraded health payload with HTTP 503.

## Numerical and availability semantics

- Official trade observations and M2 USD gaps are decimal strings serialized from the persisted artifact value. This preserves its stored precision; it cannot recover precision already absent from the upstream dataset. Overview arithmetic uses `Decimal`. Model fitting outputs, model errors, SHAP contributions and percentages remain numeric. Optional NaN/Inf diagnostics serialize as JSON null.
- Forecast points have raw `champion_forecast`, `fallback_forecast`, `champion_is_valid`, `fallback_is_valid`, and governed `operational_forecast`/`operational_model`/`used_fallback`. Served bounds belong to the actual governed model. Raw negative forecasts/lower bounds are preserved. `interval_policy` is `native`, `prequential_conformal` or `unavailable`.
- Task `metrics` are operational; `primary_metrics` and task `intervals` describe valid champion origins. `calibration_pool_size` is the minimum served-model calibration count over task points (native intervals have count zero); each point reports its actual count. Coverage rates are fractions, not percentages. M11's primary coverage uses 503 valid origins; it is not silently recomputed with the fallback as a different primary model.
- A task without an XGBoost role returns `{status: "unavailable", reason: "no_xgboost_role", ...}`. An explanation's role is champion or fallback, including when that fallback was not operationally used. SHAP explains model predictions, not causal economic effects.
- Snapshot `observation_cutoff` is 2025-12; retrieval dates are separate. These endpoints serve historical evaluation, not current-market inference. No wall-clock freshness claim is made. Source data are not refreshed in an API request.
- M2 `canonical` means exporter-reported and `mirror` importer-reported. For RU_TR the v1.1 forecast target is `mirror_value_usd`; for TR_RU it is `canonical_value_usd`. `forecast_target_column` makes this distinction explicit without rewriting the persisted gaps.
- Paired mirror observations end at 2021-12. Post-2022 `paired_count=0`, gaps and the structural-shift statistic remain null. A CIF/FOB-related observed discrepancy is not a freight-only effect estimate. The files represent national reporting distributed through UN Comtrade, not three independently acquired national/source feeds.

## Loading, configuration and deployment

```sh
uv sync --locked --extra dev --python 3.13
make run-api
# Swagger: http://127.0.0.1:8000/docs
# OpenAPI: http://127.0.0.1:8000/openapi.json

# Example deployment overrides; no credentials are required by the artifact API.
export SERVING_ARTIFACT_ROOT=/absolute/path/to/processed
export SERVING_CONFIG_ROOT=/absolute/path/to/configs
export SERVING_CORS_ORIGINS='["http://localhost:3000","https://dashboard.example"]'
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The application factory `api.main.create_app(settings, repository)` also accepts an injected `SnapshotRepository`. `SERVING_*` settings are separate from pipeline settings. No `.env` secrets are read or returned by the repository. Browser credentials and wildcard origins are disabled; CORS permits GET from exact configured frontend origins. CORS is a browser policy, not server authentication.

A deployment must supply the 20 relative files pinned in `configs/serving_artifacts_v1_1.yaml` plus that manifest itself. Its config/artifact roots may differ from the original repository paths. The research data remain gitignored; a clean source clone alone therefore reports degraded readiness until a verified authorized bundle is mounted. This milestone does not publish or redistribute the data bundle. No database credentials, network source access or trained-model deserialization are used.

Checksum verification is load-time: every pinned byte hash must match, then overlapping original M10/M11 input/output digests are cross-checked. The archived original M10 registry satisfies its historical hash; the active M11-ready registry satisfies M11's input hash. Original manifests are not rewritten. Integrity pins establish consistency against the operator-supplied manifest, not external cryptographic authenticity. Filesystem paths are constrained to configured roots.

Snapshot loading runs in the lifespan threadpool; the fallback first-request path has the same lock. Only one fully validated immutable snapshot is published per process. Missing/invalid inputs are logged and cache a degraded state. Repairing/changing files requires restart. After successful load, readiness refers to the loaded snapshot; disk changes are observed on restart, not continuously. Use read-only mounts and restart atomically with the new bundle. Each Uvicorn worker loads its own copy; memory/throughput at production scale has not been benchmarked.

## Measured local latency

2026-10-16, Python 3.13.9, Darwin arm64; local ASGI TestClient, 30 serial warm requests per route. Snapshot startup: **1.770 s**. This includes full mirror projection and checksum verification.

| Route | Median ms | p95 ms |
| --- | ---: | ---: |
| Health | 0.498 | 0.653 |
| Sectors | 0.567 | 0.670 |
| Leaderboard | 0.915 | 1.106 |
| Forecast RU_TR:10 h3 | 0.724 | 0.943 |
| SHAP RU_TR:10 h3 | 0.855 | 1.063 |
| Mirror RU_TR:10 | 1.830 | 2.311 |
| Overview | 0.583 | 0.662 |
| Brent indicator page | 0.699 | 0.842 |

Exact sample method, byte sizes and timings: `docs/M12_LATENCY_BENCHMARK.json`. These measurements include local response serialization but exclude real network latency and concurrent load. `docs/M12_HTTP_SMOKE.json` separately records a temporary local Uvicorn TCP health/data/CORS smoke test; the process was stopped afterward. Neither is hosted/deployed or M13 browser UAT evidence.

## Verification

`make m12-verify` runs repository Ruff, the M12 AST/type-escape audit and API tests under `-W error`. The final targeted run passed **70 tests, 0 warnings** (6.21 s). Checks cover all 42 tasks' frozen forecast values/bounds and all 9,072 signed/local SHAP values, not only exemplar payloads. Missing/corrupt files, upstream checksum disagreement, weakened manifests, immutable caches, dates/filters, JSON nulls, exact USD totals, CORS and read-only errors have explicit tests.

The final full repository pytest command passed 509 tests, zero warnings/errors, in 806.75 s. `docs/M12_VERIFICATION.json` records final checks; all 102 original processed-data hashes are unchanged. Machine-readable contract: `docs/M12_OPENAPI.json` (also live `/openapi.json`).
