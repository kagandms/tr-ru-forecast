"""Thread-safe per-process snapshot lifecycle with explicit degraded readiness."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from threading import Lock

from tr_ru_forecast.serving.catalog_schemas import ArtifactReadiness, HealthResponse
from tr_ru_forecast.serving.repository import SnapshotRepository
from tr_ru_forecast.serving.schemas import EVALUATION_SCOPE
from tr_ru_forecast.serving.service import ArtifactService

logger = logging.getLogger(__name__)


class ServiceUnavailable(Exception):
    """Verified snapshot is unavailable; a restart is required after repair."""


class ServingRuntime:
    """Load once; preserve either a fully published snapshot or a degraded state."""

    def __init__(self, repository: SnapshotRepository) -> None:
        """Configure lifecycle. Args: repository. Returns: none. Raises: none."""
        self._repository = repository
        self._lock = Lock()
        self._service: ArtifactService | None = None
        self._has_attempted_load = False

    def load(self) -> None:
        """Load once with logging. Args: self. Returns: none. Raises: none (readiness records failure)."""
        with self._lock:
            if self._has_attempted_load:
                return
            self._has_attempted_load = True
            try:
                self._service = ArtifactService(self._repository.load())
            except Exception:
                logger.exception("Frozen serving snapshot failed verification/loading")

    def get_service(self) -> ArtifactService:
        """Return verified service. Args: self. Returns: service. Raises: ServiceUnavailable."""
        self.load()
        if self._service is None:
            raise ServiceUnavailable("Frozen artifact snapshot is not ready")
        return self._service

    def health(self, identity: tuple[str, str]) -> HealthResponse:
        """Report cached readiness. Args: identity (version, environment). Returns: health. Raises: none."""
        self.load()
        snapshot = self._service.snapshot if self._service else None
        is_ready = snapshot is not None
        return HealthResponse(
            status="healthy" if is_ready else "degraded", timestamp=datetime.now(UTC),
            version=identity[0], environment=identity[1], evaluation_scope=EVALUATION_SCOPE,
            checksum_status="verified" if is_ready else "failed",
            verification_scope="pinned_serving_files_and_upstream_output_crosscheck_at_load",
            verified_file_count=snapshot.verified_file_count if snapshot else 0,
            serving_manifest_sha256=snapshot.manifest_sha256 if snapshot else None,
            total_tasks=len(snapshot.comparisons) if snapshot else 0,
            total_holdout_predictions=snapshot.prediction_count if snapshot else 0,
            total_shap_rows=snapshot.shap_row_count if snapshot else 0,
            readiness=ArtifactReadiness(m10=is_ready, m11=is_ready, history=is_ready,
                                       mirror=is_ready, macro=is_ready),
            m10=snapshot.m10 if snapshot else None, m11=snapshot.m11 if snapshot else None,
            error=None if is_ready else "artifact_snapshot_unavailable",
        )
