"""Dependency container.

One object graph, built once at start-up and stored on ``app.state``. Routes get
it through FastAPI dependencies, tests construct it directly with fakes. Nothing
in the codebase reaches for a global singleton.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.core.config import Settings
from app.core.logging import get_logger
from app.rag.embeddings import EmbeddingProvider, build_embedding_provider
from app.rag.ingestion import IngestionService
from app.rag.llm import LLMProvider, build_llm_provider
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.rag.vectorstores import VectorStore, build_vector_store
from app.services.conversation import ConversationStore
from app.services.vapi_client import VapiClient

logger = get_logger(__name__)


@dataclass(slots=True)
class Container:
    settings: Settings
    embeddings: EmbeddingProvider
    vector_store: VectorStore
    llm: LLMProvider
    retriever: Retriever
    ingestion: IngestionService
    pipeline: RAGPipeline
    conversations: ConversationStore
    vapi: VapiClient

    # ------------------------------------------------------------------ build
    @classmethod
    async def create(cls, settings: Settings) -> Container:
        logger.info(
            "building container llm=%s/%s embeddings=%s/%s store=%s",
            settings.llm.provider,
            settings.llm.model,
            settings.embedding.provider,
            settings.embedding.model,
            settings.vector_store.provider,
        )

        embeddings = build_embedding_provider(settings)
        # Load the embedding model before serving traffic; otherwise the first
        # caller pays a multi-second cold start mid-conversation.
        await asyncio.to_thread(embeddings.warmup)
        dimensions = embeddings.dimensions
        logger.info("embeddings ready dim=%d", dimensions)

        vector_store = build_vector_store(settings, dimensions)
        retriever = Retriever(
            embeddings=embeddings, vector_store=vector_store, config=settings.rag
        )
        ingestion = IngestionService(
            embeddings=embeddings, vector_store=vector_store, config=settings.rag
        )
        llm = build_llm_provider(settings)
        pipeline = RAGPipeline(
            settings=settings,
            embeddings=embeddings,
            vector_store=vector_store,
            llm=llm,
            retriever=retriever,
            ingestion=ingestion,
        )

        container = cls(
            settings=settings,
            embeddings=embeddings,
            vector_store=vector_store,
            llm=llm,
            retriever=retriever,
            ingestion=ingestion,
            pipeline=pipeline,
            conversations=ConversationStore(),
            vapi=VapiClient(settings),
        )

        vectors = vector_store.count()
        if vectors == 0:
            logger.warning(
                "knowledge base is empty -- run `python scripts/ingest.py` or POST "
                "/api/v1/documents/reindex before taking calls"
            )
        else:
            logger.info("knowledge base ready vectors=%d", vectors)
        return container

    # --------------------------------------------------------------- teardown
    async def aclose(self) -> None:
        for name, closer in (
            ("llm", self.llm.aclose()),
            ("vapi", self.vapi.aclose()),
        ):
            try:
                await closer
            except Exception as exc:  # pragma: no cover
                logger.warning("error closing %s: %s", name, exc)
        for name, obj in (("vector_store", self.vector_store), ("embeddings", self.embeddings)):
            try:
                obj.close()
            except Exception as exc:  # pragma: no cover
                logger.warning("error closing %s: %s", name, exc)
        logger.info("container closed")
