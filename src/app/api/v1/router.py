"""v1 route assembly."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import call_page, chat_completions, documents, health, rag, vapi_webhook

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router, prefix="/health")
api_router.include_router(rag.router, prefix="/rag")
api_router.include_router(documents.router, prefix="/documents")

# Both Vapi integration modes live under /api/v1/vapi:
#   custom-llm -> POST /api/v1/vapi/chat/completions
#   webhook    -> POST /api/v1/vapi/webhook
api_router.include_router(chat_completions.router, prefix="/vapi")
api_router.include_router(vapi_webhook.router, prefix="/vapi")
# Browser call UI: a top-level document, because a sandboxed Streamlit iframe
# cannot be granted microphone access.
api_router.include_router(call_page.router, prefix="/vapi")

__all__ = ["api_router"]
