"""Export deterministic, read-only M12 projections for the M13 static frontend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient

from api.main import app

OUTPUT_ROOT = Path("frontend/public/data")
API_ROOT = "/api/v1"


def _get_json(client: TestClient, path: str) -> dict[str, object]:
    response = client.get(path)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object response from {path}")
    return cast(dict[str, object], payload)


def _collect_mirror(client: TestClient) -> dict[str, object]:
    first = _get_json(client, f"{API_ROOT}/mirror-analytics?limit=10000")
    total = int(cast(int, first["total_observations"]))
    observations = list(cast(list[object], first["observations"]))
    offset = len(observations)
    while offset < total:
        page = _get_json(client, f"{API_ROOT}/mirror-analytics?offset={offset}&limit=10000")
        observations.extend(cast(list[object], page["observations"]))
        offset = len(observations)
    first["observations"] = observations
    first["offset"] = 0
    first["limit"] = total
    return first


def _build_bundle(client: TestClient) -> dict[str, object]:
    health = _get_json(client, f"{API_ROOT}/health")
    health.pop("timestamp", None)
    sectors = _get_json(client, f"{API_ROOT}/sectors")
    leaderboard = _get_json(client, f"{API_ROOT}/leaderboard")
    forecasts = {
        str(sector["sector_id"]): _get_json(client, f"{API_ROOT}/forecasts/{str(sector['sector_id']).replace(':', '__')}")
        for sector in cast(list[dict[str, object]], sectors["sectors"])
    }
    explanations = {
        f"{task['sector_id']}|{task['horizon']}": _get_json(
            client,
            f"{API_ROOT}/explainability/{str(task['sector_id']).replace(':', '__')}?horizon={task['horizon']}",
        )
        for task in cast(list[dict[str, object]], leaderboard["tasks"])
    }
    return {
        "health": health,
        "sectors": sectors,
        "leaderboard": leaderboard,
        "forecasts": forecasts,
        "explanations": explanations,
        "mirror": _collect_mirror(client),
        "overview": _get_json(client, f"{API_ROOT}/overview"),
        "methodology": _get_json(client, f"{API_ROOT}/methodology"),
    }


def _write_snapshot(bundle: dict[str, object]) -> str:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    content = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    snapshot_path = OUTPUT_ROOT / "snapshot.json"
    snapshot_path.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "snapshot_file": "snapshot.json",
        "snapshot_sha256": digest,
        "scope": "retrospective_latest_vintage_holdout_evaluation_under_deterministic_lag_rules",
        "health_timestamp": "omitted because it is request-time liveness metadata",
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return digest


def main() -> None:
    """Export the verified serving snapshot. Args: none. Returns: none. Raises: API or filesystem errors."""
    with TestClient(app) as client:
        digest = _write_snapshot(_build_bundle(client))
    print(f"M13 snapshot exported: {OUTPUT_ROOT / 'snapshot.json'} sha256={digest}")


if __name__ == "__main__":
    main()
