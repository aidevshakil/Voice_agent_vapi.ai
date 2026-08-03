"""ChromaDB-backed vector store (persistent, local-first)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.core.exceptions import ConfigurationError, ProviderError
from app.core.logging import get_logger
from app.rag.vectorstores.base import ScoredDocument, StoreStats, VectorStore

logger = get_logger(__name__)

_UPSERT_BATCH = 256


def _to_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate a flat equality dict into Chroma's filter syntax."""
    if not where:
        return None
    if len(where) == 1:
        key, value = next(iter(where.items()))
        return {key: {"$eq": value}}
    return {"$and": [{key: {"$eq": value}} for key, value in where.items()]}


class ChromaVectorStore(VectorStore):
    name = "chroma"

    def __init__(self, collection: str, path: Path) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "VECTOR_STORE_PROVIDER=chroma requires `pip install chromadb`. "
                "Set VECTOR_STORE_PROVIDER=memory to run without it."
            ) from exc

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection
        try:
            self._client = chromadb.PersistentClient(
                path=str(path),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection,
                # Cosine keeps scores comparable across embedding providers.
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise ProviderError(f"failed to open Chroma collection {collection!r}: {exc}") from exc
        logger.info("chroma ready collection=%s path=%s vectors=%d", collection, path, self.count())

    # ------------------------------------------------------------ operations
    def upsert(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[str],
    ) -> int:
        if not documents:
            return 0
        try:
            for start in range(0, len(documents), _UPSERT_BATCH):
                stop = start + _UPSERT_BATCH
                self._collection.upsert(
                    ids=list(ids[start:stop]),
                    embeddings=[list(map(float, e)) for e in embeddings[start:stop]],
                    documents=[d.page_content for d in documents[start:stop]],
                    metadatas=[_sanitize(d.metadata) for d in documents[start:stop]],
                )
        except Exception as exc:
            raise ProviderError(f"chroma upsert failed: {exc}") from exc
        return len(documents)

    def search(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredDocument]:
        total = self.count()
        if total == 0:
            return []
        try:
            result = self._collection.query(
                query_embeddings=[list(map(float, embedding))],
                n_results=min(k, total),
                where=_to_where(where),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise ProviderError(f"chroma query failed: {exc}") from exc

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        out: list[ScoredDocument] = []
        for doc_id, text, meta, distance in zip(ids, docs, metas, distances, strict=False):
            # Chroma cosine distance is 1 - cosine_similarity, i.e. [0, 2].
            similarity = 1.0 - (float(distance) / 2.0)
            out.append(
                ScoredDocument(
                    document=Document(page_content=text or "", metadata={**(meta or {}), "id": doc_id}),
                    score=max(0.0, min(1.0, similarity)),
                )
            )
        return out

    def delete(self, *, ids: Sequence[str] | None = None, where: dict[str, Any] | None = None) -> int:
        if not ids and not where:
            return 0
        try:
            if where and not ids:
                existing = self._collection.get(where=_to_where(where), include=[])
                ids = existing.get("ids") or []
            if not ids:
                return 0
            self._collection.delete(ids=list(ids))
        except Exception as exc:
            raise ProviderError(f"chroma delete failed: {exc}") from exc
        return len(ids)

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:  # pragma: no cover
            raise ProviderError(f"chroma count failed: {exc}") from exc

    def stats(self) -> StoreStats:
        vectors = self.count()
        sources: list[str] = []
        dimensions = 0
        if vectors:
            try:
                sample = self._collection.get(limit=vectors, include=["metadatas"])
                sources = sorted(
                    {str((m or {}).get("source", "unknown")) for m in sample.get("metadatas") or []}
                )
                peek = self._collection.peek(limit=1)
                embeds = peek.get("embeddings")
                if embeds is not None and len(embeds) > 0:
                    dimensions = len(embeds[0])
            except Exception:  # stats are best-effort
                pass
        return StoreStats(
            provider=self.name,
            collection=self._collection_name,
            vectors=vectors,
            dimensions=dimensions,
            sources=sources,
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:
            raise ProviderError(f"chroma reset failed: {exc}") from exc


def _sanitize(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return clean
