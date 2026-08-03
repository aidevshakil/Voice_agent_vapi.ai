"""OpenAI ``/chat/completions`` wire schemas.

Vapi's ``custom-llm`` provider speaks the OpenAI Chat Completions protocol, so
implementing these exactly is what lets our RAG backend stand in for a model
provider. Extra fields are allowed because Vapi adds its own (``call``,
``metadata``, ``phoneNumber``) that we read but don't require.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def text(self) -> str:
        """Flatten OpenAI's multi-part content into a plain string."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            parts = [
                str(part.get("text", ""))
                for part in self.content
                if isinstance(part, dict) and part.get("type") in (None, "text")
            ]
            return " ".join(p for p in parts if p).strip()
        return ""


class ChatCompletionRequest(BaseModel):
    """Incoming request from Vapi (or any OpenAI-compatible client)."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatCompletionMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    # Vapi-specific envelope fields.
    call: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    # ---------------------------------------------------------------- helpers
    def latest_user_message(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user" and (text := message.text().strip()):
                return text
        return ""

    def prior_turns(self, limit: int) -> list[ChatCompletionMessage]:
        """User/assistant turns before the latest user message.

        Vapi replays the whole transcript every turn, so trimming here is what
        keeps prompt size (and therefore latency and cost) flat over a long call.
        """
        conversation = [m for m in self.messages if m.role in ("user", "assistant")]
        for index in range(len(conversation) - 1, -1, -1):
            if conversation[index].role == "user":
                conversation = conversation[:index]
                break
        return conversation[-limit:] if limit > 0 else []

    def system_prompt_override(self) -> str | None:
        parts = [m.text().strip() for m in self.messages if m.role == "system"]
        joined = "\n".join(p for p in parts if p)
        return joined or None

    def call_id(self) -> str | None:
        if self.call and (cid := self.call.get("id")):
            return str(cid)
        return None


# --------------------------------------------------------------------- output
class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=_completion_id)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)

    @classmethod
    def of(cls, *, content: str, model: str) -> ChatCompletionResponse:
        return cls(model=model, choices=[Choice(message=ChoiceMessage(content=content))])


class DeltaContent(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChunkChoice]

    @classmethod
    def opening(cls, completion_id: str, model: str) -> ChatCompletionChunk:
        return cls(
            id=completion_id,
            model=model,
            choices=[ChunkChoice(delta=DeltaContent(role="assistant", content=""))],
        )

    @classmethod
    def delta(cls, completion_id: str, model: str, text: str) -> ChatCompletionChunk:
        return cls(
            id=completion_id, model=model, choices=[ChunkChoice(delta=DeltaContent(content=text))]
        )

    @classmethod
    def closing(cls, completion_id: str, model: str) -> ChatCompletionChunk:
        return cls(
            id=completion_id,
            model=model,
            choices=[ChunkChoice(delta=DeltaContent(), finish_reason="stop")],
        )
