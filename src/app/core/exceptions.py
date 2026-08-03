"""Domain exceptions and their HTTP translation."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger, get_request_id

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for expected, user-facing failures."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "configuration_error"


class ProviderError(AppError):
    """An upstream provider (LLM, embeddings, vector DB) failed."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "provider_error"


class RetrievalError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "retrieval_error"


class IngestionError(AppError):
    # Literal 422: Starlette renamed this constant, so referencing either name
    # emits a DeprecationWarning on one version or breaks on the other.
    status_code = 422
    code = "ingestion_error"


class UnsupportedDocumentError(IngestionError):
    code = "unsupported_document"


class AuthenticationError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"


class RateLimitError(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"


def _payload(code: str, message: str, details: dict[str, object] | None = None) -> dict:
    body: dict[str, object] = {
        "error": {"code": code, "message": message, "request_id": get_request_id()}
    }
    if details:
        body["error"]["details"] = details  # type: ignore[index]
    return body


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        log = logger.warning if exc.status_code < 500 else logger.error
        log("%s: %s %s", exc.code, exc.message, exc.details or "")
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_payload("internal_error", "An unexpected error occurred."),
        )
