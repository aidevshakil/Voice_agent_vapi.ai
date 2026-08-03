"""HTTP middleware: request correlation, access logs, and rate limiting."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import Settings
from app.core.logging import get_logger, set_request_id

logger = get_logger("app.access")

Next = Callable[[Request], Awaitable[Response]]

# Health checks and docs would drown the access log.
_QUIET_PATHS = frozenset({"/api/v1/health/live", "/api/v1/health/ready", "/favicon.ico"})


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, logs the request, and reports server timing."""

    async def dispatch(self, request: Request, call_next: Next) -> Response:
        request_id = set_request_id(request.headers.get("x-request-id"))
        started = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{elapsed_ms:.1f}"

        if request.url.path not in _QUIET_PATHS:
            log = logger.info if response.status_code < 500 else logger.error
            log(
                "%s %s -> %d in %.1fms",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window-per-client limiter.

    In-process and therefore per-worker — this is a guardrail against runaway
    clients, not a billing control. Put a real limiter at the edge for that.
    Vapi's own traffic is exempt: throttling a live call is worse than the load.
    """

    _EXEMPT_PREFIXES = ("/api/v1/vapi", "/api/v1/health")

    def __init__(self, app: FastAPI, requests_per_minute: int) -> None:
        super().__init__(app)
        self._limit = requests_per_minute
        self._window = 60.0
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Next) -> Response:
        if self._limit <= 0 or request.url.path.startswith(self._EXEMPT_PREFIXES):
            return await call_next(request)

        key = self._client_key(request)
        now = time.monotonic()
        hits = self._hits[key]
        while hits and hits[0] < now - self._window:
            hits.popleft()

        if len(hits) >= self._limit:
            retry_after = max(1, int(self._window - (now - hits[0])))
            logger.warning("rate limited client=%s path=%s", key, request.url.path)
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": f"Rate limit of {self._limit} requests/minute exceeded.",
                    }
                },
                headers={"retry-after": str(retry_after)},
            )

        hits.append(now)
        # Bound memory when many distinct clients appear.
        if len(self._hits) > 10_000:
            for stale in [k for k, v in self._hits.items() if not v or v[-1] < now - self._window]:
                del self._hits[stale]
        return await call_next(request)


def register_middleware(app: FastAPI, settings: Settings) -> None:
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware

    # Starlette applies middleware bottom-up, so this registration order means a
    # request hits CORS -> rate limit -> context -> route.
    app.add_middleware(RequestContextMiddleware)
    if settings.security.rate_limit_per_minute > 0:
        app.add_middleware(
            RateLimitMiddleware, requests_per_minute=settings.security.rate_limit_per_minute
        )
    # 1 KB floor keeps small SSE frames uncompressed.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins or ["http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["x-request-id", "x-response-time-ms"],
    )
