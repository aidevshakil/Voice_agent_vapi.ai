"""OpenAI provider."""

from __future__ import annotations

from app.core.exceptions import ConfigurationError
from app.rag.llm.openai_compatible import OpenAICompatibleProvider


def build_openai_provider(
    *,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    base_url: str | None = None,
) -> OpenAICompatibleProvider:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError("LLM_PROVIDER=openai requires `pip install openai`.") from exc

    client = AsyncOpenAI(
        api_key=api_key, timeout=timeout, max_retries=max_retries, base_url=base_url
    )
    return OpenAICompatibleProvider(
        client=client,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        name="openai",
    )
