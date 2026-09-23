"""Pytest fixtures shared across test modules."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from tr_ru_forecast.config import Settings, get_settings


@pytest.fixture
def client() -> TestClient:
    """Test client for FastAPI application."""
    return TestClient(app)


@pytest.fixture
def settings() -> Settings:
    """Application settings fixture."""
    return get_settings()
