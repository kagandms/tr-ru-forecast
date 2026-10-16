"""Source-control and code-content provenance for model runs."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitProvenance:
    """Immutable source-control state captured at run start."""

    commit: str
    is_dirty: bool
    dirty_paths: tuple[str, ...]


def get_git_provenance(project_root: Path) -> GitProvenance:
    """Read the current checkout revision and its changed paths."""
    commit = _run_git(project_root, "rev-parse", "HEAD").strip()
    status = _run_git(project_root, "status", "--porcelain", "--untracked-files=all")
    dirty_paths = tuple(sorted(line[3:] for line in status.splitlines() if line))
    return GitProvenance(commit, bool(dirty_paths), dirty_paths)


def compute_source_fingerprint(source_paths: list[Path]) -> str:
    """Hash a deterministic set of source files, including each relative path."""
    digest = hashlib.sha256()
    for source_path in sorted(source_paths):
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Source file required for fingerprint is missing: {source_path}"
            )
        digest.update(source_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_git(project_root: Path, *arguments: str) -> str:
    """Run a read-only Git query and fail if the project root is not a checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Could not read Git provenance for {project_root}") from error
    return result.stdout
