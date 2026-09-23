"""UN Comtrade API Client for monthly bilateral trade data ingestion.

Supports both the public preview endpoint and authenticated subscription endpoint,
handling pagination, bounded retries, rate limiting, and parameter serialization.
"""

import time
from typing import Any

import httpx

from tr_ru_forecast.config import Settings, get_settings
from tr_ru_forecast.core.logging import get_logger

logger = get_logger("tr_ru_forecast.data.ingestion.comtrade_client")

# Standard HS2 chapters: 01 to 99, excluding 77 (reserved in HS classification)
ALL_HS2_CHAPTERS: list[str] = [f"{i:02d}" for i in range(1, 100) if i != 77]
DEFAULT_COMMODITY_CODES: list[str] = ["TOTAL"] + ALL_HS2_CHAPTERS


class ComtradeError(Exception):
    """Base exception for UN Comtrade API interactions."""


class ComtradeRateLimitError(ComtradeError):
    """Raised when UN Comtrade returns HTTP 429 Too Many Requests."""


class ComtradeConnectionError(ComtradeError):
    """Raised on connection timeout or network errors."""


class ComtradeClient:
    """Client for fetching monthly trade flows from UN Comtrade v1 API."""

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._http_client = http_client

    def _get_client(self) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        return httpx.Client(timeout=self.settings.comtrade_timeout_seconds)

    @property
    def endpoint_url(self) -> str:
        """Resolve active endpoint based on API key availability."""
        if self.settings.comtrade_api_key:
            return self.settings.comtrade_subscription_url
        return self.settings.comtrade_base_url

    @property
    def is_authenticated(self) -> bool:
        """True if client has a subscription key configured."""
        return bool(self.settings.comtrade_api_key)

    def fetch_monthly_flow(
        self,
        period: str,
        reporter_code: int,
        partner_code: int,
        flow_code: str = "X",
        commodity_codes: list[str] | None = None,
        partner2_code: str | None = None,
        customs_code: str | None = None,
        mot_code: str | None = None,
    ) -> dict[str, Any]:
        """Fetch monthly trade records for a single period and direction.

        Args:
            period: Month in YYYYMM format (e.g. '202301').
            reporter_code: UN M49 country code of reporting country (e.g. 792 for TR).
            partner_code: UN M49 country code of partner country (e.g. 643 for RU).
            flow_code: 'X' for Exports, 'M' for Imports.
            commodity_codes: List of commodity codes (default: TOTAL + 01..99).
            partner2_code: 2nd partner filter (default '0' = World/Total).
            customs_code: Customs procedure filter (default 'C00' = Total).
            mot_code: Mode of transport filter (default '0' = All modes).

        Returns:
            Dictionary containing source response data and metadata.
        """
        if commodity_codes is None:
            commodity_codes = DEFAULT_COMMODITY_CODES

        partner2 = partner2_code or self.settings.comtrade_partner2_code
        customs = customs_code or self.settings.comtrade_customs_code
        mot = mot_code or self.settings.comtrade_mot_code

        params: dict[str, str] = {
            "period": period,
            "reporterCode": str(reporter_code),
            "partnerCode": str(partner_code),
            "flowCode": flow_code,
            "cmdCode": ",".join(commodity_codes),
            "partner2Code": partner2,
            "customsCode": customs,
            "motCode": mot,
        }

        headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "TR-RU-Trade-Forecast-Platform/0.1.0",
        }
        if self.settings.comtrade_api_key:
            headers["Ocp-Apim-Subscription-Key"] = self.settings.comtrade_api_key

        url = self.endpoint_url
        max_retries = self.settings.comtrade_max_retries
        backoff_factor = 1.5

        logger.debug(
            "Fetching Comtrade data: url=%s, period=%s, reporter=%s, partner=%s, flow=%s",
            url,
            period,
            reporter_code,
            partner_code,
            flow_code,
        )

        client = self._get_client()
        for attempt in range(1, max_retries + 1):
            try:
                if self._http_client is None and self.settings.comtrade_rate_limit_delay > 0:
                    time.sleep(self.settings.comtrade_rate_limit_delay)
                response = client.get(url, params=params, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    records = data.get("data", []) if isinstance(data, dict) else []
                    logger.info(
                        "Comtrade fetch successful: period=%s, reporter=%s, partner=%s, records=%d",
                        period,
                        reporter_code,
                        partner_code,
                        len(records),
                    )
                    return data

                if response.status_code == 429:
                    wait_time = backoff_factor**attempt
                    logger.warning(
                        "Rate limit (429) hit for period %s. Backing off for %.2fs (attempt %d/%d)",
                        period,
                        wait_time,
                        attempt,
                        max_retries,
                    )
                    time.sleep(wait_time)
                    continue

                if response.status_code in (500, 502, 503, 504):
                    wait_time = backoff_factor**attempt
                    logger.warning(
                        "Server error (%d) for period %s. Retrying in %.2fs (attempt %d/%d)",
                        response.status_code,
                        period,
                        wait_time,
                        attempt,
                        max_retries,
                    )
                    time.sleep(wait_time)
                    continue

                # Non-retriable error
                error_msg = f"UN Comtrade HTTP {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                raise ComtradeError(error_msg)

            except (httpx.TimeoutException, httpx.NetworkError) as err:
                if attempt == max_retries:
                    logger.error(
                        "Connection error to UN Comtrade after %d attempts: %s", max_retries, err
                    )
                    raise ComtradeConnectionError(
                        f"Connection failed after {max_retries} attempts: {err}"
                    ) from err
                wait_time = backoff_factor**attempt
                logger.warning(
                    "Network glitch on attempt %d/%d: %s. Retrying in %.2fs",
                    attempt,
                    max_retries,
                    err,
                    wait_time,
                )
                time.sleep(wait_time)

        raise ComtradeError(f"Exceeded max retries ({max_retries}) for period {period}")
