"""Local storage management for raw and near-raw UN Comtrade API responses."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tr_ru_forecast.config import Settings, get_settings
from tr_ru_forecast.core.logging import get_logger

logger = get_logger("tr_ru_forecast.data.storage.raw_storage")


class RawSnapshotStore:
    """Manages raw snapshot files for reproducible trade ingestion."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_dir = self.settings.raw_data_dir / "comtrade"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_direction_dir(self, direction: str) -> Path:
        dir_path = self.base_dir / direction
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def save_snapshot(
        self,
        direction: str,
        period: str,
        raw_response: dict[str, Any],
        request_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a raw API response with ingestion provenance metadata.

        Args:
            direction: Canonical direction ('TR_RU' or 'RU_TR').
            period: Month period in YYYYMM format.
            raw_response: Full JSON dictionary returned by UN Comtrade.
            request_metadata: Optional metadata dictionary describing request context.

        Returns:
            Path to saved JSON file.
        """
        dir_path = self._get_direction_dir(direction)
        file_path = dir_path / f"{period}.json"

        snapshot = {
            "metadata": {
                "source": "UN Comtrade",
                "direction": direction,
                "period": period,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "record_count": len(raw_response.get("data", [])),
                **(request_metadata or {}),
            },
            "response": raw_response,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

        logger.debug("Raw snapshot saved to %s", file_path)
        return file_path

    def load_snapshot(self, direction: str, period: str) -> dict[str, Any] | None:
        """Load a previously saved raw snapshot.

        Args:
            direction: Canonical direction ('TR_RU' or 'RU_TR').
            period: Month period in YYYYMM format.

        Returns:
            Snapshot dictionary or None if not found.
        """
        file_path = self._get_direction_dir(direction) / f"{period}.json"
        if not file_path.exists():
            return None

        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    def list_periods(self, direction: str) -> list[str]:
        """List all period strings for which raw snapshots exist.

        Args:
            direction: Canonical direction.

        Returns:
            Sorted list of period strings (e.g. ['202301', '202302']).
        """
        dir_path = self._get_direction_dir(direction)
        periods: list[str] = []
        for p in dir_path.glob("*.json"):
            if not p.stem.startswith("."):
                periods.append(p.stem)
        return sorted(periods)
