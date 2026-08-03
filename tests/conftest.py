"""Test fixtures.

Everything runs against fakes: a deterministic hash-based embedder, the numpy
vector store, and a scripted LLM. No API keys, no network, no model downloads.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.core.config import Settings
from app.core.container import Container
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.ingestion import IngestionService
from app.rag.llm.base import ChatMessage, CompletionResult, LLMProvider
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.rag.vectorstores.memory_store import MemoryVectorStore
from app.services.conversation import ConversationStore
from app.services.vapi_client import VapiClient

DIMENSIONS = 64


class FakeEmbeddings(EmbeddingProvider):
    """Deterministic bag-of-words hashing embedder.

    Not semantic, but *consistent*: texts sharing vocabulary get high cosine
    similarity, which is exactly the property retrieval tests need.
    """

    name = "fake"

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % DIMENSIONS] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def warmup(self) -> None:
        return None


class FakeLLM(LLMProvider):
    """Echoes back the context it was given, so grounding is directly assertable."""

    name = "fake"

    def __init__(self, reply: str | None = None, fail: bool = False) -> None:
        super().__init__("fake-model", temperature=0.0, max_tokens=100)
        self.reply = reply
        self.fail = fail
        self.calls: list[list[ChatMessage]] = []

    def _answer(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        if self.fail:
            from app.core.exceptions import ProviderError

            raise ProviderError("simulated provider outage")
        if self.reply is not None:
            return self.reply
        last = messages[-1].content
        return "Support is open nine to six Pacific." if "KNOWLEDGE" in last else "Hello there."

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        return CompletionResult(text=self._answer(messages), model=self.model)

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        for word in self._answer(messages).split(" "):
            yield word + " "


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        app={"env": "development", "log_level": "WARNING", "reload": False},  # type: ignore[arg-type]
        llm={"provider": "groq", "model": "fake-model", "max_tokens": 100},  # type: ignore[arg-type]
        embedding={"provider": "fastembed", "model": "fake", "dimensions": DIMENSIONS},  # type: ignore[arg-type]
        vector_store={"provider": "memory", "collection": "test_kb", "path": tmp_path / "vs"},  # type: ignore[arg-type]
        rag={  # type: ignore[arg-type]
            "chunk_size": 300,
            "chunk_overlap": 40,
            "top_k": 3,
            "fetch_k": 8,
            "min_relevance_score": 0.0,
            "documents_dir": tmp_path / "docs",
            "query_cache_size": 16,
            "query_cache_ttl_seconds": 60,
        },
        security={"api_keys": [], "rate_limit_per_minute": 0},  # type: ignore[arg-type]
        groq_api_key="test-key",  # type: ignore[arg-type]
    )


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def store(settings: Settings) -> MemoryVectorStore:
    return MemoryVectorStore(collection="test_kb", path=settings.vector_store.path)


@pytest.fixture
def container(
    settings: Settings, embeddings: FakeEmbeddings, llm: FakeLLM, store: MemoryVectorStore
) -> Container:
    retriever = Retriever(embeddings=embeddings, vector_store=store, config=settings.rag)
    ingestion = IngestionService(embeddings=embeddings, vector_store=store, config=settings.rag)
    pipeline = RAGPipeline(
        settings=settings,
        embeddings=embeddings,
        vector_store=store,
        llm=llm,
        retriever=retriever,
        ingestion=ingestion,
    )
    return Container(
        settings=settings,
        embeddings=embeddings,
        vector_store=store,
        llm=llm,
        retriever=retriever,
        ingestion=ingestion,
        pipeline=pipeline,
        conversations=ConversationStore(),
        vapi=VapiClient(settings),
    )


@pytest.fixture
def knowledge_documents() -> list[Document]:
    return [
        Document(
            page_content=(
                "Support hours are Monday through Friday, nine in the morning to six "
                "in the evening Pacific Time. Enterprise customers have twenty four "
                "seven phone support through a dedicated success manager."
            ),
            metadata={"source": "handbook.md", "page": 1},
        ),
        Document(
            page_content=(
                "Refund policy: new customers may request a full refund within thirty "
                "days of the first payment. After thirty days refunds are prorated for "
                "the unused portion of the billing period."
            ),
            metadata={"source": "handbook.md", "page": 2},
        ),
        Document(
            page_content=(
                "Pricing: the Starter plan costs twenty nine dollars per month, Growth "
                "costs ninety nine dollars per month, and Enterprise pricing is custom."
            ),
            metadata={"source": "pricing.md", "page": 1},
        ),
    ]


@pytest.fixture
async def seeded_container(container: Container, knowledge_documents: list[Document]) -> Container:
    await container.ingestion.ingest_documents(knowledge_documents)
    return container


def _build_client(container: Container) -> TestClient:
    """App wired to the fixture container, with real start-up bypassed.

    ``create_app``'s lifespan would build the production object graph (downloading
    an embedding model, opening Chroma). Swapping in a no-op lifespan and setting
    ``state.container`` directly gives the same routes over fakes.
    """
    from contextlib import asynccontextmanager

    from app.main import create_app

    @asynccontextmanager
    async def noop_lifespan(app):
        app.state.container = container
        yield

    app = create_app(container.settings)
    app.router.lifespan_context = noop_lifespan  # type: ignore[assignment]
    return TestClient(app)


@pytest.fixture
def client(container: Container):
    with _build_client(container) as test_client:
        yield test_client


@pytest.fixture
def seeded_client(container: Container, knowledge_documents: list[Document]):
    """Client whose knowledge base already contains the sample documents.

    Seeds through the sync path deliberately: driving the async one here would
    bind the service's asyncio primitives to a loop that TestClient then replaces.
    """
    container.ingestion.index_sync(knowledge_documents)
    with _build_client(container) as test_client:
        yield test_client
