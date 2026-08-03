"""OpenAI-compatible ``/chat/completions`` — the Vapi ``custom-llm`` endpoint.

Set the assistant's model to ``{"provider": "custom-llm", "url": ".../api/v1/vapi"}``
and Vapi POSTs here for every turn, appending ``/chat/completions`` itself. Because
*we* own the prompt, retrieval runs on every single turn and the model physically
cannot answer from its own weights — which is the whole point of RAG for voice.

The response is Server-Sent Events in the exact OpenAI chunk format so Vapi
starts synthesising speech from the first sentence.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import ContainerDep, VapiAuthDep
from app.core.logging import get_logger
from app.rag.llm.base import ChatMessage
from app.schemas.openai_compat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
    _completion_id,
)
from app.utils.text import estimate_tokens

logger = get_logger(__name__)

router = APIRouter(tags=["vapi-custom-llm"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # Nginx buffers SSE by default, which would defeat streaming entirely.
    "X-Accel-Buffering": "no",
}


def _sse(payload: object) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


@router.post(
    "/chat/completions",
    dependencies=[VapiAuthDep],
    # The handler returns either SSE or JSON depending on `stream`, which is not a
    # single Pydantic model -- FastAPI must not try to infer one.
    response_model=None,
    summary="OpenAI-compatible completions backed by RAG (Vapi custom-llm)",
)
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    container: ContainerDep,
) -> StreamingResponse | JSONResponse:
    pipeline = container.pipeline
    settings = container.settings

    question = payload.latest_user_message()
    call_id = payload.call_id()
    history = [
        ChatMessage(role=m.role, content=m.text())  # type: ignore[arg-type]
        for m in payload.prior_turns(settings.rag.history_turns)
        if m.text()
    ]

    logger.info(
        "custom-llm turn call=%s stream=%s history=%d q=%r",
        call_id or "-",
        payload.stream,
        len(history),
        question[:120],
    )

    # Vapi's dashboard system prompt is appended to ours rather than replacing it,
    # so persona tweaks are possible without losing the grounding rules.
    extra_prompt = payload.system_prompt_override()
    model_name = payload.model or settings.llm.model

    common = {
        "mode": "voice",
        "history": history,
        "system_prompt_extra": extra_prompt,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
    }

    if not payload.stream:
        answer = await pipeline.answer(question, **common)  # type: ignore[arg-type]
        if call_id:
            await container.conversations.append_exchange(call_id, question, answer.answer)
        response = ChatCompletionResponse.of(content=answer.answer, model=model_name)
        response.usage = Usage(
            prompt_tokens=estimate_tokens(question),
            completion_tokens=estimate_tokens(answer.answer),
            total_tokens=estimate_tokens(question) + estimate_tokens(answer.answer),
        )
        return JSONResponse(content=response.model_dump())

    completion_id = _completion_id()

    async def event_stream() -> AsyncIterator[str]:
        spoken: list[str] = []
        # An immediate role-only frame tells Vapi the stream is alive before the
        # first token arrives (~200-400 ms of retrieval + prefill).
        yield _sse(ChatCompletionChunk.opening(completion_id, model_name).model_dump())
        try:
            async for delta in pipeline.stream_answer(question, **common):  # type: ignore[arg-type]
                if await request.is_disconnected():
                    logger.info("caller hung up mid-stream call=%s", call_id or "-")
                    return
                spoken.append(delta)
                yield _sse(
                    ChatCompletionChunk.delta(completion_id, model_name, delta).model_dump(
                        exclude_none=True
                    )
                )
            # exclude_none so the terminal frame is `"delta": {}`, matching OpenAI.
            yield _sse(
                ChatCompletionChunk.closing(completion_id, model_name).model_dump(
                    exclude_none=True
                )
            )
            yield "data: [DONE]\n\n"
        finally:
            # Persist the turn even on disconnect so the transcript stays complete.
            if call_id and spoken:
                await container.conversations.append_exchange(
                    call_id, question, "".join(spoken).strip()
                )

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@router.get("/models", summary="Model list (some OpenAI clients probe this)")
async def list_models(container: ContainerDep) -> dict[str, object]:
    model = container.settings.llm.model
    return {
        "object": "list",
        "data": [{"id": model, "object": "model", "owned_by": container.llm.name}],
    }
