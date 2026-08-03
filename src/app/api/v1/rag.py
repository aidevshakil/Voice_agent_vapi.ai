"""First-party RAG API: text chat, streaming chat, retrieval debugging, sessions."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import ContainerDep, require_api_key
from app.core.logging import get_logger
from app.schemas.rag import (
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    RetrieveRequest,
    RetrieveResponse,
    SessionResponse,
)

logger = get_logger(__name__)

router = APIRouter(tags=["rag"], dependencies=[Depends(require_api_key)])


def _where(source: str | None) -> dict[str, str] | None:
    return {"source": source} if source else None


@router.post("/query", response_model=QueryResponse, summary="Ask a grounded question")
async def query(body: QueryRequest, container: ContainerDep) -> QueryResponse:
    session_id = body.session_id or uuid.uuid4().hex[:16]
    history = await container.conversations.history(
        session_id, turns=container.settings.rag.history_turns
    )

    answer = await container.pipeline.answer(
        body.question,
        mode=body.mode,
        history=history,
        top_k=body.top_k,
        where=_where(body.source),
        use_cache=body.use_cache,
    )
    await container.conversations.append_exchange(session_id, body.question, answer.answer)
    return QueryResponse.from_answer(answer, session_id=session_id)


@router.post("/query/stream", summary="Ask a grounded question, streamed as SSE")
async def query_stream(body: QueryRequest, container: ContainerDep) -> StreamingResponse:
    session_id = body.session_id or uuid.uuid4().hex[:16]
    history = await container.conversations.history(
        session_id, turns=container.settings.rag.history_turns
    )

    async def event_stream() -> AsyncIterator[str]:
        collected: list[str] = []
        yield f"event: meta\ndata: {json.dumps({'session_id': session_id})}\n\n"
        async for delta in container.pipeline.stream_answer(
            body.question,
            mode=body.mode,
            history=history,
            top_k=body.top_k,
            where=_where(body.source),
            use_cache=body.use_cache,
        ):
            collected.append(delta)
            yield f"data: {json.dumps({'delta': delta})}\n\n"
        await container.conversations.append_exchange(
            session_id, body.question, "".join(collected).strip()
        )
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieval only — inspect chunks and scores without LLM cost",
)
async def retrieve(body: RetrieveRequest, container: ContainerDep) -> RetrieveResponse:
    outcome = await container.pipeline.retriever.retrieve(
        body.query,
        top_k=body.top_k,
        min_score=body.min_score,
        where=_where(body.source),
        use_cache=body.use_cache,
    )
    return RetrieveResponse(
        query=body.query,
        chunks=[
            RetrievedChunk(
                text=doc.text,
                score=round(doc.score, 4),
                source=doc.source,
                metadata=doc.metadata,
            )
            for doc in outcome.documents
        ],
        candidates_considered=outcome.candidates_considered,
        cache_hit=outcome.cache_hit,
        timings=outcome.timings,
    )


# ------------------------------------------------------------------- sessions
@router.get(
    "/sessions/{session_id}", response_model=SessionResponse, summary="Read a session transcript"
)
async def get_session(session_id: str, container: ContainerDep) -> SessionResponse:
    turns = await container.conversations.transcript(session_id)
    if not turns:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found")
    return SessionResponse(
        session_id=session_id,
        turns=[{"role": t.role, "content": t.content, "at": t.at} for t in turns],
    )


@router.delete("/sessions/{session_id}", summary="Forget a session")
async def delete_session(session_id: str, container: ContainerDep) -> dict[str, bool]:
    return {"deleted": await container.conversations.clear(session_id)}


@router.post("/cache/clear", summary="Flush the retrieval cache")
async def clear_cache(container: ContainerDep) -> dict[str, str]:
    await container.pipeline.retriever.clear_cache()
    return {"status": "cleared"}
