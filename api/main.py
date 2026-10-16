"""FastAPI application factory for immutable frozen evaluation serving."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from api.errors import register_error_handlers
from api.routes import health, router
from tr_ru_forecast import __version__
from tr_ru_forecast.config import get_settings
from tr_ru_forecast.core.logging import setup_logging
from tr_ru_forecast.serving.config import ServingSettings
from tr_ru_forecast.serving.repository import DiskArtifactRepository, SnapshotRepository
from tr_ru_forecast.serving.runtime import ServingRuntime
from tr_ru_forecast.serving.schemas import FrozenModel


class RootResponse(FrozenModel):
    """Immutable root response contract."""

    service: str = "TR-RU Trade Forecasting Platform API"
    version: str = __version__
    status: str = "online"
    docs_url: str = "/docs"


def root() -> RootResponse:
    """Return service identity. Args: none. Returns: identity. Raises: none."""
    return RootResponse()


def create_app(
    settings: ServingSettings | None = None, repository: SnapshotRepository | None = None
) -> FastAPI:
    """Create isolated API. Args: settings, repository. Returns: app. Raises: configuration errors."""
    configuration = settings or ServingSettings()
    runtime = ServingRuntime(repository or DiskArtifactRepository(configuration))

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Load off the event loop. Args: application. Yields: ready/degraded app. Raises: none."""
        try:
            await run_in_threadpool(runtime.load)
            yield
        finally:
            logging.getLogger(__name__).info("Frozen serving snapshot stopped")

    application = FastAPI(title="TR-RU Bilateral Trade Forecasting API", version=__version__,
                          description="Read-only frozen retrospective evaluation and mirror analytics.", lifespan=lifespan)
    application.state.serving_runtime = runtime
    application.add_middleware(CORSMiddleware, allow_origins=list(configuration.cors_origins),
                               allow_credentials=False, allow_methods=["GET"], allow_headers=["Accept", "Content-Type"])
    register_error_handlers(application)
    application.add_api_route("/", root, methods=["GET"], tags=["System"])
    application.add_api_route("/health", health, methods=["GET"], tags=["System"])
    application.include_router(router, prefix="/api/v1")
    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    configuration = get_settings()
    setup_logging(level=configuration.log_level)
    uvicorn.run(app, host=configuration.api_host, port=configuration.api_port)
