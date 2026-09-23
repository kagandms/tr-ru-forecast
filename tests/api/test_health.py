"""Integration tests for API health and root endpoints."""

from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient) -> None:
    """Verify root endpoint returns basic service information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "TR-RU Trade Forecasting Platform API"
    assert data["status"] == "online"
    assert "version" in data


def test_health_endpoint(client: TestClient) -> None:
    """Verify /health endpoint returns healthy status and metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data
    assert "environment" in data
