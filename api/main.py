"""FastAPI application entrypoint for the TR-RU Trade Forecasting Platform."""

from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tr_ru_forecast import __version__
from tr_ru_forecast.config import get_settings
from tr_ru_forecast.core.logging import get_logger, setup_logging

settings = get_settings()
setup_logging(level=settings.log_level)
logger = get_logger("tr_ru_forecast.api")

app = FastAPI(
    title="TR-RU Bilateral Trade Forecasting API",
    description="API foundation for trade analytics, forecasting, mirror statistics, and model lab.",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow CORS for Next.js frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
def root() -> dict[str, Any]:
    """Root endpoint returning service identity."""
    return {
        "service": "TR-RU Trade Forecasting Platform API",
        "version": __version__,
        "status": "online",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["System"])
def health() -> dict[str, Any]:
    """Health check endpoint for container and uptime monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": __version__,
        "environment": settings.app_env,
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting API server on %s:%s", settings.api_host, settings.api_port)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
