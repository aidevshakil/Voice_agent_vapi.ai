"""Per-call conversation memory.

Vapi's custom-LLM mode already replays the full message list on every turn, so
this store exists for the *other* paths — the tool-call webhook and the text
chat API — where the client sends only the latest utterance. Sessions expire so
a long-running server doesn't leak memory for calls that ended hours ago.

Backed by an in-process dict, which is correct for a single worker. For
multi-worker or multi-instance deployments swap the implementation for Redis;
the interface is deliberately narrow enough to make that a drop-in change.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.rag.llm.base import ChatMessage, Role

logger = get_logger(__name__)

_MAX_SESSIONS = 1000
_SESSION_TTL_SECONDS = 3600.0
_MAX_MESSAGES_PER_SESSION = 60


@dataclass(slots=True)
class Turn:
    role: Role
    content: str
    at: float = field(default_factory=time.time)


@dataclass(slots=True)
class Session:
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return len(self.turns)


class ConversationStore:
    """LRU + TTL session store."""

    def __init__(
        self,
        *,
        max_sessions: int = _MAX_SESSIONS,
        ttl_seconds: float = _SESSION_TTL_SECONDS,
        max_messages: int = _MAX_MESSAGES_PER_SESSION,
    ) -> None:
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._max_sessions = max_sessions
        self._ttl = ttl_seconds
        self._max_messages = max_messages
        self._lock = asyncio.Lock()

    async def _get_or_create(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        now = time.time()
        if session is not None and now - session.last_seen_at > self._ttl:
            del self._sessions[session_id]
            session = None
        if session is None:
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
        session.last_seen_at = now
        self._sessions.move_to_end(session_id)
        self._evict_locked()
        return session

    def _evict_locked(self) -> None:
        cutoff = time.time() - self._ttl
        for key in [k for k, s in self._sessions.items() if s.last_seen_at < cutoff]:
            del self._sessions[key]
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    async def append(self, session_id: str, role: Role, content: str) -> None:
        if not content.strip():
            return
        async with self._lock:
            session = await self._get_or_create(session_id)
            session.turns.append(Turn(role=role, content=content.strip()))
            if len(session.turns) > self._max_messages:
                del session.turns[: len(session.turns) - self._max_messages]

    async def append_exchange(self, session_id: str, question: str, answer: str) -> None:
        await self.append(session_id, "user", question)
        await self.append(session_id, "assistant", answer)

    async def history(self, session_id: str, *, turns: int = 6) -> list[ChatMessage]:
        """Return the last ``turns`` messages as LLM-ready messages."""
        if turns <= 0:
            return []
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            recent = session.turns[-turns:]
            return [ChatMessage(role=t.role, content=t.content) for t in recent]

    async def transcript(self, session_id: str) -> list[Turn]:
        async with self._lock:
            session = self._sessions.get(session_id)
            return list(session.turns) if session else []

    async def set_metadata(self, session_id: str, **fields: object) -> None:
        async with self._lock:
            session = await self._get_or_create(session_id)
            session.metadata.update(fields)

    async def clear(self, session_id: str) -> bool:
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    async def stats(self) -> dict[str, int]:
        async with self._lock:
            return {
                "sessions": len(self._sessions),
                "turns": sum(s.turn_count for s in self._sessions.values()),
            }
