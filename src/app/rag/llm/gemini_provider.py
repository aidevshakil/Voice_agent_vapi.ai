"""Google Gemini provider (``google-genai`` SDK).

Gemini's wire format differs from OpenAI's in two ways we normalise here:
the system prompt is a top-level ``system_instruction`` rather than a message,
and the assistant role is called ``model``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.core.exceptions import ConfigurationError, ProviderError
from app.core.logging import get_logger
from app.rag.llm.base import ChatMessage, CompletionResult, LLMProvider

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(model, temperature=temperature, max_tokens=max_tokens)
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "LLM_PROVIDER=gemini requires `pip install google-genai`."
            ) from exc

        self._genai = genai
        self._client = genai.Client(
            api_key=api_key, http_options={"timeout": int(timeout * 1000)}
        )

    def _split(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        temp, tokens = self._resolve(temperature, max_tokens)
        system_parts = [m.content for m in messages if m.role == "system"]
        contents = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role != "system"
        ]
        config: dict[str, Any] = {"temperature": temp, "max_output_tokens": tokens}
        if system_parts:
            config["system_instruction"] = "\n\n".join(system_parts)
        return contents, config

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        contents, config = self._split(messages, temperature, max_tokens)
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except Exception as exc:
            raise ProviderError(f"gemini completion failed: {exc}") from exc

        usage = getattr(response, "usage_metadata", None)
        return CompletionResult(
            text=getattr(response, "text", "") or "",
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        contents, config = self._split(messages, temperature, max_tokens)
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            )
            async for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception as exc:
            raise ProviderError(f"gemini stream failed: {exc}") from exc
