"""Pytest fixtures shared across test modules."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from tr_ru_forecast.config import Settings, get_settings


@pytest.fixture(scope="session", autouse=True)
def isolate_processed_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep pipeline tests from overwriting the developer's research artifacts."""
    settings = get_settings()
    original_data_dir = settings.processed_data_dir
    isolated_data_dir = tmp_path_factory.mktemp("processed-artifacts")
    shutil.copytree(original_data_dir, isolated_data_dir, dirs_exist_ok=True)
    settings.processed_data_dir = isolated_data_dir

    yield

    settings.processed_data_dir = original_data_dir


@pytest.fixture
def client() -> TestClient:
    """Test client for FastAPI application."""
    return TestClient(app)


@pytest.fixture
def settings() -> Settings:
    """Application settings fixture."""
    return get_settings()


@pytest.fixture(scope="module", autouse=True)
def later_milestone_report_hashes() -> dict[Path, str | None]:
    """Capture later reports before earlier-model pipelines run; they must stay unchanged."""
    root = get_settings().project_root
    paths = [
        root / "docs" / name
        for name in (
            "M10_CHAMPION_SELECTION_REPORT.md",
            "M11_HOLDOUT_REPORT.md",
            "M11_FINAL_HOLDOUT_REPORT.md",
            "M11_ACCEPTANCE_REPORT.md",
        )
    ]
    paths += list(get_settings().processed_data_dir.glob("m10_*")) + list(
        get_settings().processed_data_dir.glob("m11_*")
    )
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        for path in paths
    }
