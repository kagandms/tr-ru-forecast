"""Unit tests for UN Comtrade API client using mock transports."""

import httpx
import pytest

from tr_ru_forecast.config import Settings
from tr_ru_forecast.data.ingestion.comtrade_client import (
    ComtradeClient,
    ComtradeConnectionError,
    ComtradeError,
)


def test_endpoint_resolution() -> None:
    """Verify endpoint selection based on API key presence."""
    settings_no_key = Settings(comtrade_api_key=None)
    client_no_key = ComtradeClient(settings=settings_no_key)
    assert "public/v1/preview" in client_no_key.endpoint_url
    assert client_no_key.is_authenticated is False

    settings_with_key = Settings(comtrade_api_key="mock-key-12345")
    client_with_key = ComtradeClient(settings=settings_with_key)
    assert "data/v1/get" in client_with_key.endpoint_url
    assert client_with_key.is_authenticated is True


def test_successful_fetch_mocked() -> None:
    """Verify successful response parsing with mock HTTP client."""
    mock_payload = {
        "elapsedTime": 0.05,
        "count": 1,
        "data": [
            {
                "period": "202301",
                "reporterCode": 792,
                "partnerCode": 643,
                "cmdCode": "TOTAL",
                "primaryValue": 500000.0,
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "reporterCode=792" in str(request.url)
        assert "partnerCode=643" in str(request.url)
        return httpx.Response(200, json=mock_payload)

    mock_transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock_transport)

    client = ComtradeClient(http_client=http_client)
    res = client.fetch_monthly_flow(
        period="202301",
        reporter_code=792,
        partner_code=643,
    )

    assert "data" in res
    assert len(res["data"]) == 1
    assert res["data"][0]["cmdCode"] == "TOTAL"


def test_retry_on_server_error() -> None:
    """Verify client retries on transient 500/503 errors."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            return httpx.Response(503, text="Service Unavailable")
        return httpx.Response(200, json={"data": [{"period": "202301"}]})

    mock_transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock_transport)

    settings = Settings(comtrade_max_retries=3)
    client = ComtradeClient(settings=settings, http_client=http_client)

    res = client.fetch_monthly_flow(period="202301", reporter_code=792, partner_code=643)
    assert attempts == 2
    assert len(res["data"]) == 1


def test_timeout_raises_connection_error() -> None:
    """Verify timeout after max retries raises ComtradeConnectionError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("Connection timed out")

    mock_transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock_transport)

    settings = Settings(comtrade_max_retries=1)
    client = ComtradeClient(settings=settings, http_client=http_client)

    with pytest.raises(ComtradeConnectionError):
        client.fetch_monthly_flow(period="202301", reporter_code=792, partner_code=643)


def test_non_retriable_error() -> None:
    """Verify HTTP 400 immediately raises ComtradeError without useless retries."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="Bad Request: invalid parameter")

    mock_transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock_transport)

    settings = Settings(comtrade_max_retries=3)
    client = ComtradeClient(settings=settings, http_client=http_client)

    with pytest.raises(ComtradeError):
        client.fetch_monthly_flow(period="202301", reporter_code=792, partner_code=643)

    assert attempts == 1
