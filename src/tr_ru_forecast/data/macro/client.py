"""Macroeconomic data retrieval client.

Fetches authoritative macroeconomic series snapshots from FRED and Eurostat,
persisting raw data with full provenance audit trails, strictly adhering to DATA_SPEC.md.
"""

import hashlib
import json
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tr_ru_forecast.config.settings import get_settings
from tr_ru_forecast.core.logging import get_logger

logger = get_logger("tr_ru_forecast.data.macro.client")


class FredMacroClient:
    """Robust client to retrieve raw macroeconomic time series snapshots from FRED and Eurostat."""

    def __init__(self, raw_dir: Path | None = None, cookies_path: Path | None = None) -> None:
        settings = get_settings()
        self.raw_dir = raw_dir or (settings.raw_data_dir / "macro")
        self.cookies_path = cookies_path or (settings.project_root / ".fred_cookies.txt")

    def _ensure_session_cookies(self) -> None:
        """Initialize session cookies from FRED base domain if cookie jar is empty."""
        if not self.cookies_path.exists() or self.cookies_path.stat().st_size < 100:
            logger.info("Initializing FRED session cookies from base domain...")
            cmd = [
                "curl",
                "--http1.1",
                "-s",
                "--max-time",
                "15",
                "-c",
                str(self.cookies_path),
                "https://fred.stlouisfed.org/",
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except Exception as e:
                logger.warning(f"Could not pre-initialize session cookies: {e}")

    def fetch_eurostat_series(
        self,
        indicator_id: str,
        series_id: str,
        url: str,
        max_retries: int = 3,
    ) -> tuple[str, dict[str, Any]]:
        """Download raw JSON-stat series from Eurostat REST API.

        Args:
            indicator_id: Canonical indicator ID (e.g. 'tr_ip').
            series_id: Eurostat dataset/series identifier (e.g. 'sts_inpr_m').
            url: Full Eurostat REST API URL.
            max_retries: Maximum number of attempts.

        Returns:
            Tuple of (raw_json_text, audit_metadata).
        """
        logger.info(f"Fetching Eurostat series '{series_id}' for '{indicator_id}' from {url}")
        retrieved_at = datetime.now(UTC).isoformat()

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "curl/8.7.1",
                "Accept": "application/json",
            },
        )

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw_bytes = resp.read()
                    raw_text = raw_bytes.decode("utf-8")
                    data = json.loads(raw_text)
                    if "dimension" in data and "value" in data:
                        break
                    logger.warning(f"Attempt {attempt}/{max_retries} received invalid Eurostat JSON")
            except Exception as e:
                last_err = e
                logger.warning(f"Attempt {attempt}/{max_retries} failed for Eurostat series: {e}")
        else:
            raise ConnectionError(
                f"Failed to fetch Eurostat series {series_id} for indicator {indicator_id}: {last_err}"
            )

        # Parse observation metadata
        time_idx = data["dimension"]["time"]["category"]["index"]
        vals = data["value"]
        valid_periods = [p for p, idx in time_idx.items() if str(idx) in vals]
        sorted_valid = sorted(valid_periods)

        row_count = len(sorted_valid)
        sha256_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        audit_meta = {
            "indicator_id": indicator_id,
            "source_series_id": series_id,
            "source_url": url,
            "retrieved_at": retrieved_at,
            "http_status": 200,
            "raw_row_count": row_count,
            "sha256_checksum": sha256_hash,
            "first_observation_date": sorted_valid[0] if sorted_valid else None,
            "last_observation_date": sorted_valid[-1] if sorted_valid else None,
            "data_format": "eurostat_json_stat_2_0",
        }

        logger.info(
            f"Successfully fetched {row_count} Eurostat observations for '{indicator_id}' "
            f"({audit_meta['first_observation_date']} to {audit_meta['last_observation_date']})"
        )
        return raw_text, audit_meta

    def fetch_series(
        self,
        indicator_id: str,
        series_id: str,
        url: str,
        retrieval_method: str = "fred_csv_download",
        max_retries: int = 3,
    ) -> tuple[str, dict[str, Any]]:
        """Download raw series from configured provider.

        Routes Eurostat endpoints to fetch_eurostat_series and FRED to curl.
        """
        if retrieval_method == "eurostat_json_download" or "ec.europa.eu" in url:
            return self.fetch_eurostat_series(
                indicator_id=indicator_id,
                series_id=series_id,
                url=url,
                max_retries=max_retries,
            )

        self._ensure_session_cookies()
        logger.info(f"Fetching raw series '{series_id}' for '{indicator_id}' from {url}")
        retrieved_at = datetime.now(UTC).isoformat()

        cmd = [
            "curl",
            "--http1.1",
            "-s",
            "--max-time",
            "30",
            "-b",
            str(self.cookies_path),
            "-c",
            str(self.cookies_path),
            "-H",
            "Accept: text/csv,application/csv,*/*",
            url,
        ]

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                output = res.stdout
                if output and output.startswith("observation_date"):
                    break
                logger.warning(
                    f"Attempt {attempt}/{max_retries} received invalid response header: {output[:100]!r}"
                )
            except subprocess.CalledProcessError as e:
                last_err = e
                logger.warning(f"Attempt {attempt}/{max_retries} failed with exit code {e.returncode}")
                self._ensure_session_cookies()
        else:
            raise ConnectionError(
                f"Failed to fetch series {series_id} for indicator {indicator_id} after {max_retries} attempts: {last_err}"
            )

        lines = output.strip().split("\n")
        row_count = len(lines) - 1  # Excluding header
        sha256_hash = hashlib.sha256(output.encode("utf-8")).hexdigest()

        audit_meta = {
            "indicator_id": indicator_id,
            "source_series_id": series_id,
            "source_url": url,
            "retrieved_at": retrieved_at,
            "http_status": 200,
            "raw_row_count": row_count,
            "sha256_checksum": sha256_hash,
            "first_observation_date": lines[1].split(",")[0] if len(lines) > 1 else None,
            "last_observation_date": lines[-1].split(",")[0] if len(lines) > 1 else None,
            "data_format": "fred_csv",
        }

        logger.info(
            f"Successfully fetched {row_count} raw observations for '{indicator_id}' "
            f"({audit_meta['first_observation_date']} to {audit_meta['last_observation_date']})"
        )
        return output, audit_meta

    def save_raw_snapshot(
        self,
        indicator_id: str,
        raw_text: str,
        audit_meta: dict[str, Any],
    ) -> tuple[Path, Path]:
        """Save raw immutable file and JSON audit metadata to disk.

        Returns:
            Tuple of (raw_file_path, json_metadata_path).
        """
        dest_dir = self.raw_dir / indicator_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        is_json = raw_text.strip().startswith("{")
        ext = "json" if is_json else "csv"
        raw_path = dest_dir / f"{indicator_id}_snapshot.{ext}"
        meta_path = dest_dir / f"{indicator_id}_metadata.json"

        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        # If JSON, also write a companion CSV for easy visual inspection
        if is_json:
            try:
                data = json.loads(raw_text)
                time_idx = data["dimension"]["time"]["category"]["index"]
                vals = data["value"]
                csv_path = dest_dir / f"{indicator_id}_snapshot.csv"
                with open(csv_path, "w", encoding="utf-8") as cf:
                    cf.write("observation_date,value\n")
                    for p, idx in sorted(time_idx.items(), key=lambda x: x[1]):
                        if str(idx) in vals:
                            cf.write(f"{p}-01,{vals[str(idx)]}\n")
            except Exception as e:
                logger.warning(f"Could not write companion CSV for {indicator_id}: {e}")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(audit_meta, f, indent=2)

        logger.info(f"Persisted raw snapshot to {raw_path} and metadata to {meta_path}")
        return raw_path, meta_path
