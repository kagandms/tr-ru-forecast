"""Typed runtime dependencies; business services do not import FastAPI."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from tr_ru_forecast.serving.runtime import ServingRuntime
from tr_ru_forecast.serving.service import ArtifactService


def get_runtime(request: Request) -> ServingRuntime:
    """Read app-scoped runtime. Args: request. Returns: runtime. Raises: RuntimeError."""
    runtime: object = request.app.state.serving_runtime
    if not isinstance(runtime, ServingRuntime):
        raise RuntimeError("Serving runtime is not configured")
    return runtime


RuntimeDependency = Annotated[ServingRuntime, Depends(get_runtime)]


def get_service(runtime: RuntimeDependency) -> ArtifactService:
    """Obtain verified service. Args: runtime. Returns: service. Raises: ServiceUnavailable."""
    return runtime.get_service()


ServiceDependency = Annotated[ArtifactService, Depends(get_service)]
