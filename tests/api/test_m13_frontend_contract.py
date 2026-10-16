"""Static dashboard bundle parity and protected M1-M12 artifact integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast


def test_exported_dashboard_snapshot_matches_frozen_api_dimensions() -> None:
    """The snapshot is checksummed and carries the complete frozen evaluation dimensions."""
    root = Path(__file__).parents[2]
    data_root = root / "frontend" / "public" / "data"
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    snapshot_path = data_root / str(manifest["snapshot_file"])
    snapshot_bytes = snapshot_path.read_bytes()
    assert hashlib.sha256(snapshot_bytes).hexdigest() == manifest["snapshot_sha256"]

    bundle = cast(dict[str, object], json.loads(snapshot_bytes))
    health = cast(dict[str, object], bundle["health"])
    sectors = cast(dict[str, object], bundle["sectors"])
    leaderboard = cast(dict[str, object], bundle["leaderboard"])
    forecasts = cast(dict[str, dict[str, object]], bundle["forecasts"])
    explanations = cast(dict[str, dict[str, object]], bundle["explanations"])
    assert health["total_holdout_predictions"] == 1512
    assert health["total_shap_rows"] == 9072
    assert len(cast(list[object], sectors["sectors"])) == 14
    tasks = cast(list[dict[str, object]], leaderboard["tasks"])
    assert len(tasks) == 42
    assert sum(len(cast(list[object], task["cv_candidates"])) for task in tasks) == 168
    assert sum(len(cast(list[object], forecast["tasks"])) for forecast in forecasts.values()) == 42
    assert len(explanations) == 42
    shap_rows = sum(
        len(cast(list[object], local["contributions"]))
        for explanation in explanations.values()
        if explanation["status"] == "available"
        for local in cast(list[dict[str, object]], explanation["local"])
    )
    assert shap_rows == 9072


def test_m13_did_not_change_any_m12_protected_processed_artifact() -> None:
    """All 102 protected processed bytes still match the pre-M12 and post-M12 pins."""
    root = Path(__file__).parents[2]
    integrity = json.loads((root / "docs" / "M12_ARTIFACT_INTEGRITY.json").read_text(encoding="utf-8"))
    before = cast(dict[str, str], integrity["sha256_before"])
    after = cast(dict[str, str], integrity["sha256_after"])
    assert len(before) == len(after) == 102
    assert before == after
    assert integrity["all_original_bytes_unchanged"] is True
    assert all(hashlib.sha256((root / path).read_bytes()).hexdigest() == digest for path, digest in after.items())
