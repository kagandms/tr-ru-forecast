# ==============================================================================
# Makefile - TR-RU Bilateral Trade Forecasting & Intelligence Platform
# ==============================================================================

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
PYTEST ?= $(VENV)/bin/pytest
RUFF ?= $(VENV)/bin/ruff
UVICORN ?= $(VENV)/bin/uvicorn

.PHONY: help install install-python install-frontend test lint format run-api run-frontend build-frontend m13-export-snapshot clean

help:
	@echo "TR-RU Forecasting Platform - Developer Commands"
	@echo "--------------------------------------------------"
	@echo "make install          : Setup Python venv, dependencies, and frontend packages"
	@echo "make install-python   : Install Python dependencies in virtual environment"
	@echo "make install-frontend : Install Next.js frontend dependencies"
	@echo "make test             : Run Python test suite (pytest)"
	@echo "make m10-select       : Freeze CV champions; use M10_OUTPUT_DIR for a fresh reproduction"
	@echo "make lint             : Run Python lint checks (ruff)"
	@echo "make format           : Auto-format Python code (ruff format & fix)"
	@echo "make ingest            : Ingest bilateral trade data from UN Comtrade"
	@echo "make ingest-smoke      : Run quick smoke ingestion for sample months"
	@echo "make mirror            : Run mirror statistics & quality layer for both directions"
	@echo "make mirror-smoke      : Run quick mirror pairing smoke test for sample months"
	@echo "make run-api          : Run FastAPI backend server locally"
	@echo "make run-frontend     : Run Next.js frontend development server"
	@echo "make build-frontend   : Build Next.js frontend for production"
	@echo "make m13-export-snapshot : Export deterministic verified M12 projections for the frontend"
	@echo "make clean            : Clean build artifacts and caches"

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

install-python: $(VENV)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install-frontend:
	npm --prefix frontend install

install: install-python install-frontend

test:
	$(PYTEST) tests/

lint:
	$(RUFF) check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

ingest-smoke:
	$(PYTHON) scripts/ingest_trade.py --direction TR_RU --start-period 2023-01 --end-period 2023-01

ingest:
	$(PYTHON) scripts/ingest_trade.py --direction both

mirror-smoke:
	$(PYTHON) scripts/mirror_analytics.py --direction both --periods 2021-01,2023-01 --use-cache

mirror:
	$(PYTHON) scripts/mirror_analytics.py --direction both

run-api:
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	npm --prefix frontend run dev

build-frontend:
	npm --prefix frontend run build

m13-export-snapshot:
	PYTHONPATH=. $(PYTHON) scripts/export_m13_snapshot.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .ruff_cache/ frontend/.next/
	find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: m10-select
m10-select:
	$(PYTHON) -m tr_ru_forecast.evaluation.cli $(if $(M10_OUTPUT_DIR),--output-dir "$(M10_OUTPUT_DIR)",)

.PHONY: m11-holdout
m11-holdout:
	$(PYTHON) -m tr_ru_forecast.evaluation.holdout_cli $(if $(M11_OUTPUT_DIR),--output-dir "$(M11_OUTPUT_DIR)",)

.PHONY: m12-verify
m12-verify:
	$(RUFF) check .
	$(PYTHON) scripts/verify_serving.py
	$(PYTEST) tests/api -q -W error -o addopts=''
