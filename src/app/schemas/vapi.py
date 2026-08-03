"""Vapi server-webhook schemas.

Vapi wraps everything in ``{"message": {...}}`` and its payloads evolve, so every
model here is permissive (``extra="allow"``) and reads defensively. Rejecting an
unknown field would drop a live call.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VapiToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = ""
    # Vapi sends a dict; some older payloads send a JSON string.
    arguments: dict[str, Any] | str = Field(default_factory=dict)

    def args(self) -> dict[str, Any]:
        if isinstance(self.arguments, dict):
            return self.arguments
        import json

        try:
            parsed = json.loads(self.arguments)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}


class VapiToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    type: str = "function"
    function: VapiToolCallFunction = Field(default_factory=VapiToolCallFunction)


class VapiArtifact(BaseModel):
    model_config = ConfigDict(extra="allow")

    transcript: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    recordingUrl: str | None = None


class VapiCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    orgId: str | None = None
    type: str | None = None
    status: str | None = None
    startedAt: str | None = None
    endedAt: str | None = None
    customer: dict[str, Any] | None = None


class VapiMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = ""
    timestamp: int | float | None = None
    call: VapiCall | None = None
    toolCalls: list[VapiToolCall] = Field(default_factory=list)
    toolCallList: list[VapiToolCall] = Field(default_factory=list)
    artifact: VapiArtifact | None = None
    status: str | None = None
    endedReason: str | None = None
    role: str | None = None
    transcript: str | None = None
    transcriptType: str | None = None
    summary: str | None = None

    def tool_calls(self) -> list[VapiToolCall]:
        """Vapi has used both field names; accept either."""
        return self.toolCalls or self.toolCallList

    @property
    def call_id(self) -> str | None:
        return self.call.id if self.call else None


class VapiWebhookRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: VapiMessage = Field(default_factory=VapiMessage)


# --------------------------------------------------------------------- output
class VapiToolResult(BaseModel):
    toolCallId: str
    result: str


class VapiToolResponse(BaseModel):
    """Response shape Vapi expects for ``tool-calls``."""

    results: list[VapiToolResult] = Field(default_factory=list)


class VapiWebhookAck(BaseModel):
    """Generic 200 body for events that need no reply."""

    received: bool = True
    type: str | None = None


# --------------------------------------------------------- provisioning I/O
class AssistantProvisionRequest(BaseModel):
    name: str = "RAG Voice Assistant"
    server_url: str | None = Field(
        default=None, description="Public HTTPS base URL; defaults to VAPI_SERVER_URL."
    )
    first_message: str | None = None
    integration: Literal["custom_llm", "tool"] = Field(
        default="custom_llm",
        description=(
            "custom_llm routes every turn through this backend (always grounded); "
            "tool gives a hosted model a knowledge-base function instead."
        ),
    )
    assistant_id: str | None = Field(
        default=None, description="Update this assistant instead of creating a new one."
    )


class AssistantProvisionResponse(BaseModel):
    assistant_id: str
    name: str | None = None
    action: Literal["created", "updated"]
    webhook_url: str
    custom_llm_url: str | None = None


class WebCallResponse(BaseModel):
    call_id: str | None = None
    web_call_url: str | None = None
    public_key: str | None = None
    assistant_id: str | None = None
