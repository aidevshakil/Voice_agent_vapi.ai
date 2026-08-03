"""LLM provider contract.

The RAG pipeline only ever needs two things from a model: a full completion and
a token stream. Keeping the surface this small is what makes Groq / OpenAI /
Gemini interchangeable behind one env var.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True, frozen=True)
class ChatMessage:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class CompletionResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    name: str = "base"

    def __init__(self, model: str, *, temperature: float, max_tokens: int) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Return the full answer in one shot."""

    @abstractmethod
    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield answer deltas as they arrive. Critical for voice latency."""

    async def aclose(self) -> None:  # pragma: no cover
        return None

    # ------------------------------------------------------------------ utils
    def _resolve(self, temperature: float | None, max_tokens: int | None) -> tuple[float, int]:
        return (
            self.temperature if temperature is None else temperature,
            self.max_tokens if max_tokens is None else max_tokens,
        )
