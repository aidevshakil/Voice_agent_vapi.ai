"""Vector store factory with a safe fallback.

If the configured backend cannot be constructed (missing wheel, unreachable
service) we fall back to the in-process numpy store rather than failing
start-up, and log loudly. Production deployments should set
``APP_ENV=production``, which turns the fallback off.
"""

from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger
from app.rag.vectorstores.base import VectorStore

logger = get_logger(__name__)


def _construct(settings: Settings, dimensions: int) -> VectorStore:
    cfg = settings.vector_store

    if cfg.provider == "chroma":
        from app.rag.vectorstores.chroma_store import ChromaVectorStore

        return ChromaVectorStore(collection=cfg.collection, path=cfg.path)

    if cfg.provider == "pinecone":
        from app.rag.vectorstores.pinecone_store import PineconeVectorStore

        key = settings.pinecone.api_key.get_secret_value() if settings.pinecone.api_key else None
        return PineconeVectorStore(
            api_key=key,
            index_name=settings.pinecone.index,
            namespace=cfg.collection,
            dimensions=dimensions,
            cloud=settings.pinecone.cloud,
            region=settings.pinecone.region,
        )

    if cfg.provider == "memory":
        from app.rag.vectorstores.memory_store import MemoryVectorStore

        return MemoryVectorStore(collection=cfg.collection, path=cfg.path)

    raise ConfigurationError(f"unknown VECTOR_STORE_PROVIDER: {cfg.provider!r}")


def build_vector_store(settings: Settings, dimensions: int) -> VectorStore:
    try:
        return _construct(settings, dimensions)
    except Exception as exc:
        if settings.app.is_production or settings.vector_store.provider == "memory":
            raise
        logger.error(
            "vector store %r unavailable (%s) -- falling back to the in-process "
            "numpy store. Fix the backend or set VECTOR_STORE_PROVIDER=memory.",
            settings.vector_store.provider,
            exc,
        )
        from app.rag.vectorstores.memory_store import MemoryVectorStore

        return MemoryVectorStore(
            collection=settings.vector_store.collection, path=settings.vector_store.path
        )
