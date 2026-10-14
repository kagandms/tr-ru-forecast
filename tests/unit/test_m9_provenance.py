"""Behavior tests for M9 run provenance."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tr_ru_forecast.models.ml.pipeline import compute_file_sha256
from tr_ru_forecast.models.ml.provenance import (
    compute_source_fingerprint,
    get_git_provenance,
)


def test_file_hash_fails_when_required_input_is_missing(tmp_path: Path) -> None:
    """A missing frozen input must fail instead of creating an empty checksum."""
    missing_path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError):
        compute_file_sha256(missing_path)


def test_source_fingerprint_changes_when_source_content_changes(tmp_path: Path) -> None:
    """Different model source contents must produce different run fingerprints."""
    source_path = tmp_path / "model.py"
    source_path.write_text("value = 1\n", encoding="utf-8")
    first_fingerprint = compute_source_fingerprint([source_path])

    source_path.write_text("value = 2\n", encoding="utf-8")
    second_fingerprint = compute_source_fingerprint([source_path])

    assert first_fingerprint != second_fingerprint


def test_git_provenance_reports_uncommitted_source_state(tmp_path: Path) -> None:
    """A dirty checkout reports its commit and uncommitted file paths."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "--allow-empty", "-m", "initial"],
        check=True,
        capture_output=True,
    )
    changed_path = tmp_path / "model.py"
    changed_path.write_text("value = 1\n", encoding="utf-8")

    provenance = get_git_provenance(tmp_path)

    assert len(provenance.commit) == 40
    assert provenance.is_dirty
    assert provenance.dirty_paths == ("model.py",)
