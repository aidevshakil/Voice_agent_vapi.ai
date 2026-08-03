"""Application factory and ASGI entrypoint.

Run with:  ``uvicorn app.main:app --app-dir src --reload``
or simply: ``python -m app.main``
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.middleware import register_middleware
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.container import Container
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

DESCRIPTION = """\
A real-time, knowledge-grounded voice assistant backend.

**Vapi integration — two modes, both supported:**

1. `POST /api/v1/vapi/chat/completions` — OpenAI-compatible SSE endpoint. Point a
   Vapi assistant's `custom-llm` model at `/api/v1/vapi` and *every* turn is
   RAG-grounded. This is the recommended mode.
2. `POST /api/v1/vapi/webhook` — server webhook. Handles `tool-calls` (the
   `search_knowledge_base` function), transcripts and call lifecycle events.

**Also useful:** `/api/v1/rag/query` for text chat, `/api/v1/rag/retrieve` to
inspect retrieval quality without paying for generation, and
`/api/v1/documents/*` to manage the knowledge base.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    logger.info("starting up env=%s", settings.app.env.value)
    container = await Container.create(settings)
    app.state.container = container
    try:
        yield
    finally:
        logger.info("shutting down")
        await container.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(level=settings.app.log_level, json_output=settings.app.log_json)

    app = FastAPI(
        title="Real-Time RAG Voice Assistant",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
        # Interactive docs are a liability on a public production host.
        docs_url=None if settings.app.is_production else "/docs",
        redoc_url=None if settings.app.is_production else "/redoc",
        openapi_url=None if settings.app.is_production else "/openapi.json",
    )
    app.state.settings = settings

    register_middleware(app, settings)
    register_exception_handlers(app)
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs" if not settings.app.is_production else "/api/v1/health/live")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    # --reload and multiple workers are mutually exclusive in uvicorn.
    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.reload and not settings.app.is_production,
        workers=settings.app.workers if not settings.app.reload else 1,
        log_config=None,  # we configure logging ourselves
        access_log=False,  # RequestContextMiddleware does this with request ids
    )


if __name__ == "__main__":
    main()
