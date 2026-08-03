"""Pinecone serverless vector store (managed / multi-instance deployments)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document

from app.core.exceptions import ConfigurationError, ProviderError
from app.core.logging import get_logger
from app.rag.vectorstores.base import ScoredDocument, StoreStats, VectorStore

logger = get_logger(__name__)

_UPSERT_BATCH = 100
_TEXT_KEY = "_text"


class PineconeVectorStore(VectorStore):
    """Uses one Pinecone namespace per logical collection."""

    name = "pinecone"

    def __init__(
        self,
        *,
        api_key: str | None,
        index_name: str,
        namespace: str,
        dimensions: int,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        if not api_key:
            raise ConfigurationError("VECTOR_STORE_PROVIDER=pinecone requires PINECONE_API_KEY.")
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError("Install `pinecone` to use the Pinecone store.") from exc

        self._namespace = namespace
        self._index_name = index_name
        self._dimensions = dimensions
        try:
            client = Pinecone(api_key=api_key)
            existing = {idx["name"] for idx in client.list_indexes()}
            if index_name not in existing:
                logger.info("creating pinecone index=%s dim=%d", index_name, dimensions)
                client.create_index(
                    name=index_name,
                    dimension=dimensions,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if client.describe_index(index_name).status.get("ready"):
                        break
                    time.sleep(2)
            self._index = client.Index(index_name)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ProviderError(f"pinecone initialisation failed: {exc}") from exc

    # ------------------------------------------------------------ operations
    def upsert(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[str],
    ) -> int:
        if not documents:
            return 0
        vectors = [
            {
                "id": doc_id,
                "values": list(map(float, embedding)),
                "metadata": {**_sanitize(doc.metadata), _TEXT_KEY: doc.page_content},
            }
            for doc, embedding, doc_id in zip(documents, embeddings, ids, strict=True)
        ]
        try:
            for start in range(0, len(vectors), _UPSERT_BATCH):
                self._index.upsert(
                    vectors=vectors[start : start + _UPSERT_BATCH], namespace=self._namespace
                )
        except Exception as exc:
            raise ProviderError(f"pinecone upsert failed: {exc}") from exc
        return len(vectors)

    def search(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredDocument]:
        try:
            response = self._index.query(
                vector=list(map(float, embedding)),
                top_k=k,
                namespace=self._namespace,
                include_metadata=True,
                filter={key: {"$eq": value} for key, value in where.items()} if where else None,
            )
        except Exception as exc:
            raise ProviderError(f"pinecone query failed: {exc}") from exc

        out: list[ScoredDocument] = []
        for match in response.get("matches", []):
            metadata = dict(match.get("metadata") or {})
            text = metadata.pop(_TEXT_KEY, "")
            metadata["id"] = match.get("id")
            # Pinecone cosine scores are already in [-1, 1]; clamp to [0, 1].
            out.append(
                ScoredDocument(
                    document=Document(page_content=text, metadata=metadata),
                    score=max(0.0, min(1.0, float(match.get("score", 0.0)))),
                )
            )
        return out

    def delete(self, *, ids: Sequence[str] | None = None, where: dict[str, Any] | None = None) -> int:
        try:
            if ids:
                self._index.delete(ids=list(ids), namespace=self._namespace)
                return len(ids)
            if where:
                self._index.delete(
                    filter={key: {"$eq": value} for key, value in where.items()},
                    namespace=self._namespace,
                )
                return -1  # Pinecone does not report a count for filtered deletes.
        except Exception as exc:
            raise ProviderError(f"pinecone delete failed: {exc}") from exc
        return 0

    def count(self) -> int:
        try:
            stats = self._index.describe_index_stats()
            namespaces = stats.get("namespaces") or {}
            return int(namespaces.get(self._namespace, {}).get("vector_count", 0))
        except Exception as exc:  # pragma: no cover
            raise ProviderError(f"pinecone stats failed: {exc}") from exc

    def stats(self) -> StoreStats:
        return StoreStats(
            provider=self.name,
            collection=f"{self._index_name}/{self._namespace}",
            vectors=self.count(),
            dimensions=self._dimensions,
            sources=[],  # Pinecone has no cheap distinct-metadata scan.
        )

    def reset(self) -> None:
        try:
            self._index.delete(delete_all=True, namespace=self._namespace)
        except Exception as exc:
            raise ProviderError(f"pinecone reset failed: {exc}") from exc


def _sanitize(metadata: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, (str, int, float, bool, list)) else str(value)
    return clean
