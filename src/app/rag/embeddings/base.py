"""Embedding provider contract."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """Turns text into dense vectors.

    Implementations expose sync methods; the async wrappers offload to a thread
    so CPU-bound local models never block the event loop.
    """

    name: str = "base"

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector width. Must match the vector store's configured dimension."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of passages (indexing path)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (retrieval path)."""

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    def warmup(self) -> None:
        """Force lazy model loading so the first real request isn't slow."""
        self.embed_query("warmup")

    def close(self) -> None:  # pragma: no cover - most providers are stateless
        return None
