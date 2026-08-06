"""FastAPI dependencies: container access and auth."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, Header, Request

from app.core.config import Settings
from app.core.container import Container
from app.core.exceptions import AuthenticationError
from app.core.logging import get_logger
from app.rag.pipeline import RAGPipeline
from app.services.conversation import ConversationStore
from app.services.vapi_client import VapiClient

logger = get_logger(__name__)


def get_container(request: Request) -> Container:
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - only reachable before startup
        raise RuntimeError("application container is not initialised")
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_settings_dep(container: ContainerDep) -> Settings:
    return container.settings


def get_pipeline(container: ContainerDep) -> RAGPipeline:
    return container.pipeline


def get_conversations(container: ContainerDep) -> ConversationStore:
    return container.conversations


def get_vapi_client(container: ContainerDep) -> VapiClient:
    return container.vapi


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]
ConversationsDep = Annotated[ConversationStore, Depends(get_conversations)]
VapiDep = Annotated[VapiClient, Depends(get_vapi_client)]


# ----------------------------------------------------------------------- auth
async def require_api_key(
    container: ContainerDep,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Bearer-token guard for the first-party API. No-op when API_KEYS is empty."""
    security = container.settings.security
    if not security.auth_enabled:
        return

    presented = x_api_key
    if not presented and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            presented = token.strip()

    if not presented or not any(
        hmac.compare_digest(presented, allowed) for allowed in security.api_keys
    ):
        raise AuthenticationError("missing or invalid API key")


async def verify_vapi_signature(
    container: ContainerDep,
    x_vapi_secret: Annotated[str | None, Header()] = None,
    x_vapi_signature: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    secret = container.settings.vapi.webhook_secret
    if secret is None:
        if container.settings.app.is_production:
            raise AuthenticationError(
                "VAPI_WEBHOOK_SECRET must be set when APP_ENV=production"
            )
        return

    expected = secret.get_secret_value()
    presented = x_vapi_secret or x_vapi_signature
    if not presented and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            presented = token.strip()

    if not presented or not hmac.compare_digest(presented, expected):
        logger.warning("rejected vapi webhook with invalid secret")
        raise AuthenticationError("invalid Vapi webhook secret")



AuthDep = Depends(require_api_key)
VapiAuthDep = Depends(verify_vapi_signature)
