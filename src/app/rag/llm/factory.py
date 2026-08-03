"""LLM provider factory."""

from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.rag.llm.base import LLMProvider


def build_llm_provider(settings: Settings) -> LLMProvider:
    cfg = settings.llm
    api_key = settings.require_llm_api_key()

    if cfg.provider == "groq":
        from app.rag.llm.groq_provider import build_groq_provider

        return build_groq_provider(
            api_key=api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout_seconds,
            max_retries=cfg.max_retries,
        )

    if cfg.provider == "openai":
        from app.rag.llm.openai_provider import build_openai_provider

        return build_openai_provider(
            api_key=api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout_seconds,
            max_retries=cfg.max_retries,
        )

    if cfg.provider == "gemini":
        from app.rag.llm.gemini_provider import GeminiProvider

        return GeminiProvider(
            api_key=api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout_seconds,
        )

    raise ConfigurationError(f"unknown LLM_PROVIDER: {cfg.provider!r}")
