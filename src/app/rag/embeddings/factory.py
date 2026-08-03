"""Embedding provider factory."""

from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.rag.embeddings.base import EmbeddingProvider


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    cfg = settings.embedding
    provider = cfg.provider

    if provider == "fastembed":
        from app.rag.embeddings.fastembed_provider import FastEmbedProvider

        return FastEmbedProvider(model=cfg.model, batch_size=cfg.batch_size)

    if provider == "openai":
        from app.rag.embeddings.openai_provider import OpenAIEmbeddingProvider

        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAIEmbeddingProvider(
            model=cfg.model,
            api_key=key,
            dimensions=cfg.dimensions,
            batch_size=cfg.batch_size,
        )

    if provider == "sentence_transformers":
        from app.rag.embeddings.sentence_transformers_provider import (
            SentenceTransformersProvider,
        )

        return SentenceTransformersProvider(model=cfg.model, batch_size=cfg.batch_size)

    raise ConfigurationError(f"unknown EMBEDDING_PROVIDER: {provider!r}")
