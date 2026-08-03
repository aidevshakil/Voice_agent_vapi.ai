"""Vapi server webhook + assistant provisioning.

Handles three kinds of inbound message:

* ``tool-calls`` — the alternative integration mode: a hosted Vapi model calls our
  ``search_knowledge_base`` function and we return a grounded answer string.
* ``transcript`` / ``status-update`` — live call telemetry, mirrored into the
  session store so the UI can show the conversation as it happens.
* ``end-of-call-report`` — final transcript, summary and recording URL.

Unknown message types are acknowledged with 200. Returning an error would make
Vapi retry and, in the tool case, stall a live call.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks

from app.api.deps import ContainerDep, VapiAuthDep, VapiDep
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.rag import prompts
from app.schemas.vapi import (
    AssistantProvisionRequest,
    AssistantProvisionResponse,
    VapiToolResponse,
    VapiToolResult,
    VapiWebhookRequest,
    WebCallResponse,
)

logger = get_logger(__name__)

router = APIRouter(tags=["vapi"])

KNOWLEDGE_TOOL_NAMES = frozenset(
    {"search_knowledge_base", "searchKnowledgeBase", "knowledge_search", "rag_search"}
)


@router.post(
    "/webhook",
    dependencies=[VapiAuthDep],
    summary="Vapi server webhook (tool calls, transcripts, call lifecycle)",
)
async def vapi_webhook(
    payload: VapiWebhookRequest,
    background: BackgroundTasks,
    container: ContainerDep,
) -> dict[str, Any]:
    message = payload.message
    event = message.type
    call_id = message.call_id or "unknown"
    logger.debug("vapi webhook type=%s call=%s", event, call_id)

    if event in ("tool-calls", "function-call"):
        return (await _handle_tool_calls(payload, container)).model_dump()

    if event == "transcript":
        # Only final transcripts; partials fire many times per utterance.
        if message.transcriptType == "final" and message.transcript and message.role:
            role = "user" if message.role == "user" else "assistant"
            background.add_task(
                container.conversations.append, call_id, role, message.transcript
            )
        return {"received": True, "type": event}

    if event == "status-update":
        logger.info("call %s status=%s reason=%s", call_id, message.status, message.endedReason)
        background.add_task(
            container.conversations.set_metadata,
            call_id,
            status=message.status,
            ended_reason=message.endedReason,
        )
        return {"received": True, "type": event}

    if event == "end-of-call-report":
        artifact = message.artifact
        logger.info(
            "call %s ended reason=%s transcript_chars=%d",
            call_id,
            message.endedReason,
            len(artifact.transcript or "") if artifact else 0,
        )
        background.add_task(
            container.conversations.set_metadata,
            call_id,
            ended_reason=message.endedReason,
            summary=message.summary,
            recording_url=artifact.recordingUrl if artifact else None,
        )
        return {"received": True, "type": event}

    if event in ("assistant-request", "conversation-update", "speech-update", "hang"):
        return {"received": True, "type": event}

    logger.info("unhandled vapi webhook type=%r", event)
    return {"received": True, "type": event}


async def _handle_tool_calls(
    payload: VapiWebhookRequest, container: ContainerDep
) -> VapiToolResponse:
    """Answer each ``search_knowledge_base`` call with a grounded string."""
    message = payload.message
    call_id = message.call_id
    results: list[VapiToolResult] = []

    for tool_call in message.tool_calls():
        name = tool_call.function.name
        if name not in KNOWLEDGE_TOOL_NAMES:
            logger.warning("unknown tool requested: %r", name)
            results.append(
                VapiToolResult(
                    toolCallId=tool_call.id,
                    result=f"Tool {name!r} is not available on this server.",
                )
            )
            continue

        args = tool_call.function.args()
        query = str(args.get("query") or args.get("question") or "").strip()
        if not query:
            results.append(
                VapiToolResult(
                    toolCallId=tool_call.id, result=prompts.FALLBACK_ANSWERS["empty_question"]
                )
            )
            continue

        history = (
            await container.conversations.history(
                call_id, turns=container.settings.rag.history_turns
            )
            if call_id
            else []
        )
        answer = await container.pipeline.answer(query, mode="voice", history=history)
        if call_id:
            await container.conversations.append_exchange(call_id, query, answer.answer)

        logger.info(
            "tool search_knowledge_base grounded=%s chunks=%d q=%r",
            answer.grounded,
            answer.retrieved_count,
            query[:100],
        )
        results.append(VapiToolResult(toolCallId=tool_call.id, result=answer.answer))

    return VapiToolResponse(results=results)


# ------------------------------------------------------------- provisioning
@router.post(
    "/assistant",
    response_model=AssistantProvisionResponse,
    summary="Create or update the Vapi assistant from this repo's config",
)
async def provision_assistant(
    body: AssistantProvisionRequest, container: ContainerDep, vapi: VapiDep
) -> AssistantProvisionResponse:
    payload = vapi.build_assistant_payload(
        name=body.name,
        server_url=body.server_url,
        first_message=body.first_message,
        use_custom_llm=body.integration == "custom_llm",
    )
    # Attach the shared secret so Vapi authenticates itself back to us.
    if secret := container.settings.vapi.webhook_secret:
        payload["server"]["secret"] = secret.get_secret_value()

    target = body.assistant_id or container.settings.vapi.assistant_id
    if target:
        result = await vapi.update_assistant(target, payload)
        action = "updated"
    else:
        result = await vapi.create_assistant(payload)
        action = "created"

    assistant_id = str(result.get("id") or target or "")
    base = (body.server_url or container.settings.vapi.server_url or "").rstrip("/")
    return AssistantProvisionResponse(
        assistant_id=assistant_id,
        name=result.get("name"),
        action=action,  # type: ignore[arg-type]
        webhook_url=f"{base}/api/v1/vapi/webhook",
        custom_llm_url=(
            f"{base}/api/v1/vapi/chat/completions" if body.integration == "custom_llm" else None
        ),
    )


@router.get("/assistant", summary="List assistants on the configured Vapi account")
async def list_assistants(vapi: VapiDep) -> dict[str, Any]:
    assistants = await vapi.list_assistants()
    return {
        "count": len(assistants),
        "assistants": [
            {"id": a.get("id"), "name": a.get("name"), "model": (a.get("model") or {}).get("provider")}
            for a in assistants
        ],
    }


@router.post(
    "/web-call",
    response_model=WebCallResponse,
    summary="Credentials for starting a browser call with the Vapi Web SDK",
)
async def start_web_call(container: ContainerDep) -> WebCallResponse:
    cfg = container.settings.vapi
    if not cfg.public_key:
        raise ConfigurationError(
            "VAPI_PUBLIC_KEY is required for browser calls (it is safe to expose)."
        )
    if not cfg.assistant_id:
        raise ConfigurationError("VAPI_ASSISTANT_ID is not set; provision an assistant first.")
    # The browser SDK dials directly with the public key, so no server-side call
    # object is needed -- we just hand over the identifiers.
    return WebCallResponse(public_key=cfg.public_key, assistant_id=cfg.assistant_id)


@router.get("/calls/{call_id}", summary="Fetch a call record from Vapi")
async def get_call(call_id: str, vapi: VapiDep) -> dict[str, Any]:
    return await vapi.get_call(call_id)
