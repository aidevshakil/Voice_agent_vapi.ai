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

DESCRIPTION = """A real-time, knowledge-grounded voice assistant backend."""


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
        docs_url=None if settings.app.is_production else "/docs",
        redoc_url=None if settings.app.is_production else "/redoc",
        openapi_url=None if settings.app.is_production else "/openapi.json",
    )
    app.state.settings = settings

    register_middleware(app, settings)
    register_exception_handlers(app)
    app.include_router(api_router)

    from pathlib import Path
    from fastapi.staticfiles import StaticFiles

    dist_dir = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if dist_dir.exists():
        app.mount("/web", StaticFiles(directory=str(dist_dir), html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs" if not settings.app.is_production else "/api/v1/health/live")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.reload and not settings.app.is_production,
        workers=settings.app.workers if not settings.app.reload else 1,
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
