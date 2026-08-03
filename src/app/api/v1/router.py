"""v1 route assembly."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import chat_completions, documents, health, rag, vapi_webhook

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router, prefix="/health")
api_router.include_router(rag.router, prefix="/rag")
api_router.include_router(documents.router, prefix="/documents")

# Both Vapi integration modes live under /api/v1/vapi:
#   custom-llm -> POST /api/v1/vapi/chat/completions
#   webhook    -> POST /api/v1/vapi/webhook
api_router.include_router(chat_completions.router, prefix="/vapi")
api_router.include_router(vapi_webhook.router, prefix="/vapi")

__all__ = ["api_router"]
