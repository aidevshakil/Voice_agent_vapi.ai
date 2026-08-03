"""Health and readiness probes.

``/live`` answers without touching any dependency (Kubernetes liveness).
``/ready`` reports component state and turns "knowledge base is empty" into a
``degraded`` status, because a voice agent with no vectors will answer every
question with "I don't know" while looking perfectly healthy.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from app.api.deps import ContainerDep
from app.core.logging import get_logger
from app.schemas.rag import HealthResponse

logger = get_logger(__name__)

router = APIRouter(tags=["health"])

VERSION = "1.0.0"


@router.get("/live", summary="Liveness probe")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", response_model=HealthResponse, summary="Readiness and component detail")
async def ready(response: Response, container: ContainerDep) -> HealthResponse:
    components: dict[str, Any] = {}
    healthy = True

    try:
        components = await container.pipeline.health()
    except Exception as exc:
        logger.error("health check failed: %s", exc)
        components = {"error": str(exc)}
        healthy = False

    kb_ready = bool(components.get("knowledge_base_ready"))
    components["vapi"] = {
        "configured": container.vapi.configured,
        "assistant_id": container.settings.vapi.assistant_id,
        "webhook_secret_set": container.settings.vapi.webhook_secret is not None,
    }
    components["sessions"] = await container.conversations.stats()

    ok = healthy and kb_ready
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if ok else "degraded",
        version=VERSION,
        environment=container.settings.app.env.value,
        knowledge_base_ready=kb_ready,
        components=components,
    )


@router.get("", response_model=HealthResponse, summary="Alias for /ready")
async def health(response: Response, container: ContainerDep) -> HealthResponse:
    return await ready(response, container)
