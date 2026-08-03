"""Groq provider — the default, because time-to-first-token drives voice UX."""

from __future__ import annotations

from app.core.exceptions import ConfigurationError
from app.rag.llm.openai_compatible import OpenAICompatibleProvider


def build_groq_provider(
    *,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> OpenAICompatibleProvider:
    try:
        from groq import AsyncGroq
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError("LLM_PROVIDER=groq requires `pip install groq`.") from exc

    client = AsyncGroq(api_key=api_key, timeout=timeout, max_retries=max_retries)
    return OpenAICompatibleProvider(
        client=client,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        name="groq",
    )
