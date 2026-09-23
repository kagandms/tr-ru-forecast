# TR–RU Bilateral Trade Forecasting & Intelligence Platform

Research-oriented economic intelligence platform analyzing bilateral trade flows between Türkiye and Russia, providing 1, 3, and 6-month horizon forecasts, model comparisons, prediction intervals, SHAP explainability, and mirror-statistics data quality analytics.

---

## 1. Project Status & Milestone

- **Current Milestone:** `M1 — Trade Data Ingestion`
- **Specification Status:** `SPECIFICATION FREEZE v1.0`
- **Phase:** UN Comtrade bilateral trade ingestion pipeline, parameter serialization, deterministic deduplication, raw snapshot storage, and standardized `trade_monthly` datasets implemented. No M2+ business logic.

---

## 2. Specification Reference

The project scope, data contracts, modeling rules, and acceptance gates are governed by the frozen documents in [`specification/`](specification/):

1. [`00_SPEC_INDEX.md`](specification/00_SPEC_INDEX.md) — Master index and change-control policy.
2. [`PROJECT_SPEC.md`](specification/PROJECT_SPEC.md) — Product statement, canonical directions, scope, and Definition of Done.
3. [`DATA_SPEC.md`](specification/DATA_SPEC.md) — Data contracts, canonical/mirror rules, and sector eligibility filters.
4. [`EXPERIMENT_PROTOCOL.md`](specification/EXPERIMENT_PROTOCOL.md) — Mandatory models, walk-forward validation, holdout rules, and champion criteria.
5. [`MILESTONES.md`](specification/MILESTONES.md) — Implementation roadmap, deliverables, and acceptance gates (M0 to M15).

---

## 3. Architecture Overview

```text
External Sources (UN Comtrade, Macro APIs)
      ↓
Data Ingestion (Monthly canonical & mirror trade)
      ↓
Raw / Validated Data Storage
      ↓
Feature Engineering & Mirror Analytics
      ↓
Forecasting Models (Seasonal Naive, SARIMA, SARIMAX, XGBoost Direct)
      ↓
Evaluation & Champion Selection (Expanding Walk-Forward, Untouched Holdout)
      ↓
Model Registry & Forecast Storage
      ↓
FastAPI Backend (/health, trade, forecasts, metrics)
      ↓
Next.js Multi-language Dashboard (TR / EN / RU)
```

---

## 4. Repository Structure

```text
tr-ru_forecast/
├── specification/             # Verbatim frozen specification documents (v1.0)
├── src/
│   └── tr_ru_forecast/        # Core Python package
│       ├── core/              # Frozen constants and central structured logging
│       ├── config/            # Pydantic Settings and path resolution
│       ├── data/              # Ingestion, quality, and storage modules (skeletons)
│       ├── features/          # Feature engineering modules (skeletons)
│       ├── models/            # Baselines, classical, and ML model modules (skeletons)
│       └── evaluation/        # Validation, metrics, and champion selection (skeletons)
├── api/                       # FastAPI application entrypoint (minimal /health skeleton)
├── frontend/                  # Next.js frontend skeleton (localized TR/EN/RU architecture)
├── tests/                     # Test suite (unit, API integration, smoke tests)
│   ├── unit/                  # Settings and logging unit tests
│   └── api/                   # FastAPI health and root endpoint tests
├── configs/                   # Base configuration YAML files
├── scripts/                   # Development onboarding and maintenance scripts
├── data/                      # Local data folders (raw, interim, processed - gitignored)
├── models/                    # Model artifacts storage (gitignored)
├── research/                  # Research notebooks and methodology documentation
├── docs/                      # Technical documentation
├── .env.example               # Environment variables template
├── .gitignore                 # Standard ignore file for Python, Node, data, and secrets
├── pyproject.toml             # Python package build configuration and tool settings
├── Makefile                   # Convenient developer commands
└── README.md                  # Project documentation
```

---

## 5. Prerequisites

- **Python:** 3.11+ (Tested on Python 3.13.9)
- **Node.js:** 20+ (Tested on Node v24.12.0 / npm 11.6.2)
- **Make:** Standard build utility (optional, but recommended)

---

## 6. Local Setup & Quickstart

### Option A: Using Makefile

```bash
# 1. Setup virtual environment and install dependencies (Python & Frontend)
make install

# 2. Run automated tests
make test

# 3. Run lint checks
make lint

# 4. Start the FastAPI backend server (http://localhost:8000)
make run-api

# 5. In a separate terminal, start the Next.js frontend (http://localhost:3000)
make run-frontend
```

### Option B: Manual Setup

```bash
# 1. Python environment setup
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# 2. Environment variables configuration
cp .env.example .env

# 3. Verify backend and tests
pytest tests/
ruff check .

# 4. Run backend
python api/main.py

# 5. Setup and build frontend
cd frontend
npm install
npm run build
npm run dev
```

---

## 7. Development Commands

| Command | Description |
| :--- | :--- |
| `make test` | Run test suite via `pytest` |
| `make lint` | Check Python code formatting and lint rules with `ruff` |
| `make format` | Automatically fix and format Python code with `ruff` |
| `make ingest` | Run UN Comtrade ingestion pipeline for both directions |
| `make ingest-smoke`| Run smoke test ingestion for sample monthly periods |
| `make run-api` | Launch FastAPI development server with hot-reload |
| `make run-frontend` | Launch Next.js development server |
| `make build-frontend`| Produce optimized Next.js production build |
| `make clean` | Remove temporary cache files, build artifacts, and pycache |
