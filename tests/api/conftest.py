"""API clients use isolated disk roots and verify real research artifacts remain untouched."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from tr_ru_forecast.config import get_settings
from tr_ru_forecast.serving.config import ServingSettings


def _hash_tree(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


@pytest.fixture(scope="session")
def serving_settings(isolate_processed_artifacts: None) -> ServingSettings:
    """Bind API to pytest's copied data root, retaining pinned configuration."""
    settings = get_settings()
    return ServingSettings(artifact_root=settings.processed_data_dir,
                           config_root=settings.project_root / "configs")


@pytest.fixture(scope="session", autouse=True)
def protect_research_artifacts() -> Iterator[None]:
    """Hash original data before API tests and again after the entire pytest session."""
    directory = get_settings().project_root / "data/processed"
    original = _hash_tree(directory)

    yield

    assert _hash_tree(directory) == original


@pytest.fixture(scope="session")
def artifact_client(serving_settings: ServingSettings) -> Iterator[TestClient]:
    """Cache one immutable API snapshot over isolated research copies."""
    with TestClient(create_app(serving_settings)) as client:
        yield client
