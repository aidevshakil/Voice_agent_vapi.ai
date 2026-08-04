"""Shared implementation for OpenAI-wire-compatible providers (OpenAI, Groq).

Both SDKs expose the identical ``chat.completions.create`` surface, so one class
covers them and each concrete provider just supplies a configured client.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.rag.llm.base import ChatMessage, CompletionResult, LLMProvider

logger = get_logger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        temperature: float,
        max_tokens: int,
        name: str = "openai_compatible",
        supports_stream_options: bool = False,
    ) -> None:
        super().__init__(model, temperature=temperature, max_tokens=max_tokens)
        self._client = client
        self.name = name
        # Opt-in, not assumed: `stream_options` is an OpenAI extension and the
        # groq SDK raises TypeError on the unexpected kwarg, which would fail
        # every streamed turn -- i.e. every Vapi custom-llm call.
        self._supports_stream_options = supports_stream_options

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        temp, tokens = self._resolve(temperature, max_tokens)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temp,
            "max_tokens": tokens,
            "stream": stream,
        }
        if stream and self._supports_stream_options:
            # Ask for usage on the final chunk so we can log cost per turn.
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        try:
            response = await self._client.chat.completions.create(
                **self._payload(messages, temperature, max_tokens, stream=False)
            )
        except Exception as exc:
            raise ProviderError(f"{self.name} completion failed: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        usage = getattr(response, "usage", None)
        return CompletionResult(
            text=(getattr(choice.message, "content", "") or "") if choice else "",
            model=getattr(response, "model", self.model),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                **self._payload(messages, temperature, max_tokens, stream=True)
            )
        except Exception as exc:
            raise ProviderError(f"{self.name} stream failed to start: {exc}") from exc

        try:
            async for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue  # usage-only trailer chunk
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text
        except Exception as exc:
            raise ProviderError(f"{self.name} stream interrupted: {exc}") from exc

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result
