"""Structured public errors without raw request values or internal exceptions."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from tr_ru_forecast.serving.runtime import ServiceUnavailable
from tr_ru_forecast.serving.schemas import ErrorDetail, ErrorResponse
from tr_ru_forecast.serving.service import CatalogNotFound

logger = logging.getLogger(__name__)


def _response(status: int, code: str, message: str) -> JSONResponse:
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(status_code=status, content=payload.model_dump(mode="json"))


def register_error_handlers(app: FastAPI) -> None:
    """Install error boundaries. Args: app. Returns: none. Raises: none."""
    app.add_exception_handler(CatalogNotFound, handle_not_found)
    app.add_exception_handler(ServiceUnavailable, handle_unavailable)
    app.add_exception_handler(RequestValidationError, handle_validation)
    app.add_exception_handler(HTTPException, handle_http_error)
    app.add_exception_handler(Exception, handle_unexpected)


async def handle_not_found(request: Request, error: CatalogNotFound) -> JSONResponse:
    """Map domain catalog errors. Args: request, error. Returns: 404. Raises: none."""
    return _response(404, "catalog_not_found", str(error))


async def handle_unavailable(request: Request, error: ServiceUnavailable) -> JSONResponse:
    """Map readiness errors. Args: request, error. Returns: 503. Raises: none."""
    return _response(503, "artifact_snapshot_unavailable", "Verified frozen artifacts are unavailable")


async def handle_validation(request: Request, error: RequestValidationError) -> JSONResponse:
    """Map query/path validation. Args: request, error. Returns: 422. Raises: none."""
    return _response(422, "invalid_request", "Invalid query or path parameters")


async def handle_http_error(request: Request, error: HTTPException) -> JSONResponse:
    """Keep framework errors structured. Args: request, error. Returns: HTTP error. Raises: none."""
    return _response(error.status_code, "http_error", "Request could not be served")


async def handle_unexpected(request: Request, error: Exception) -> JSONResponse:
    """Log internal failures, return redacted message. Args: request, error. Returns: 500. Raises: none."""
    logger.error("Unhandled serving error", exc_info=error)
    return _response(500, "internal_error", "An internal error occurred")
