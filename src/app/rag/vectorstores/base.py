"""Vector store contract.

All implementations normalise their native distance metric into a **cosine
similarity in [0, 1]** so ``RAG_MIN_RELEVANCE_SCORE`` means the same thing no
matter which backend is configured.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


@dataclass(slots=True)
class ScoredDocument:
    """A retrieved chunk plus its normalised relevance score."""

    document: Document
    score: float

    @property
    def text(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata

    @property
    def source(self) -> str:
        return str(self.metadata.get("source") or self.metadata.get("title") or "unknown")


@dataclass(slots=True)
class StoreStats:
    provider: str
    collection: str
    vectors: int
    dimensions: int
    sources: list[str] = field(default_factory=list)


class VectorStore(ABC):
    """Minimal surface a RAG retriever needs from a vector database."""

    name: str = "base"

    @abstractmethod
    def upsert(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[str],
    ) -> int:
        """Insert or replace vectors. Returns the number written."""

    @abstractmethod
    def search(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredDocument]:
        """Nearest-neighbour search, highest similarity first."""

    @abstractmethod
    def delete(self, *, ids: Sequence[str] | None = None, where: dict[str, Any] | None = None) -> int:
        """Delete by id or metadata filter. Returns the number removed."""

    @abstractmethod
    def count(self) -> int:
        """Total vectors in the collection."""

    @abstractmethod
    def stats(self) -> StoreStats:
        """Collection summary for the /health and /documents endpoints."""

    @abstractmethod
    def reset(self) -> None:
        """Drop every vector in the collection."""

    def list_sources(self) -> list[str]:
        return self.stats().sources

    # ------------------------------------------------------------------ async
    async def aupsert(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[str],
    ) -> int:
        return await asyncio.to_thread(self.upsert, documents, embeddings, ids)

    async def asearch(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredDocument]:
        return await asyncio.to_thread(self.search, embedding, k, where=where)

    def close(self) -> None:  # pragma: no cover
        return None
