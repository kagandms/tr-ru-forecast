# M14 Portfolio Documentation Acceptance Record

**Date:** 2026-10-16
**Status:** Documentation showcase and specification synchronization accepted; broader M14 research package remains in progress.

## Delivered

- Rebuilt the root README as a three-layer portfolio guide: verified results and dataflow, empirical/methodological interpretation and limits, then API/dashboard and reproduction instructions.
- Updated the milestone roadmap, specification index/changelog and completion-plan status with M10–M13 evidence and the retrospective latest-vintage deterministic-lag scope.
- Corrected source descriptions to match the artifacts: trade mirror series are paired reporter records distributed through UN Comtrade; the repository does not contain separately acquired, independently verified TurkStat and Rosstat trade feeds.
- Documented `frontend/` as the Vercel Root Directory. The verified snapshot is bundled locally and no live Python API is required at build time.

## Verification

- `.venv/bin/ruff check .` — passed.
- `.venv/bin/pytest -W error` — **511 passed, 0 warnings/errors** (822.47 seconds).
- `npm run build` from `frontend/` — passed; Next.js statically prerendered the dashboard without a Python backend.
- `./node_modules/.bin/tsc --noEmit` from `frontend/` — passed.
- Processed artifact integrity — all 102 SHA-256 values matched `docs/M12_ARTIFACT_INTEGRITY.json` after the full test run.
- Frontend snapshot — matched manifest and expected SHA-256 `867a3d8230ca62c23a2f7bc4e7021e0f9f8eba0fe8a727469f1c11b07bebf574`.
- `git diff --check` — passed.

## Remaining scope

This record does not claim completion of the full M14 research package (working paper/technical report, Turkish summary, Russian abstract, data dictionary and source/licensing package) or M15 release, hosted deployment and user acceptance. No commit, push or deployment was performed.
