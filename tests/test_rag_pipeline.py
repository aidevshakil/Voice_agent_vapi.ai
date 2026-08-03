"""Pipeline, retriever and ingestion behaviour."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.core.container import Container
from app.rag.pipeline import RAGPipeline
from tests.conftest import FakeLLM


# ------------------------------------------------------------------- ingestion
async def test_ingestion_indexes_and_reports(container: Container, knowledge_documents):
    result = await container.ingestion.ingest_documents(knowledge_documents)

    assert result.chunks_indexed > 0
    assert result.chunks_created >= result.chunks_indexed
    assert set(result.sources) == {"handbook.md", "pricing.md"}
    assert container.vector_store.count() == result.chunks_indexed


async def test_ingestion_is_idempotent(container: Container, knowledge_documents):
    """Re-ingesting identical content must not duplicate vectors."""
    first = await container.ingestion.ingest_documents(knowledge_documents)
    await container.ingestion.ingest_documents(knowledge_documents)

    assert container.vector_store.count() == first.chunks_indexed


async def test_edited_content_creates_a_new_chunk(container: Container, knowledge_documents):
    await container.ingestion.ingest_documents(knowledge_documents)
    before = container.vector_store.count()

    await container.ingestion.ingest_documents(
        [Document(page_content="Brand new paragraph about widgets and gadgets in stock.",
                  metadata={"source": "handbook.md"})]
    )
    assert container.vector_store.count() == before + 1


async def test_short_fragments_are_dropped(container: Container):
    result = await container.ingestion.ingest_documents(
        [Document(page_content="Hi.", metadata={"source": "tiny.md"})]
    )
    assert result.chunks_indexed == 0


async def test_delete_source_removes_only_that_source(container: Container, knowledge_documents):
    await container.ingestion.ingest_documents(knowledge_documents)
    total = container.vector_store.count()

    deleted = await container.ingestion.delete_source("pricing.md")

    assert deleted > 0
    assert container.vector_store.count() == total - deleted
    assert "pricing.md" not in container.vector_store.stats().sources


# ------------------------------------------------------------------- retrieval
async def test_retrieval_ranks_the_relevant_chunk_first(seeded_container: Container):
    outcome = await seeded_container.retriever.retrieve("refund within thirty days")

    assert outcome.documents
    assert "refund" in outcome.documents[0].text.lower()
    assert 0.0 <= outcome.top_score <= 1.0


async def test_retrieval_respects_source_filter(seeded_container: Container):
    outcome = await seeded_container.retriever.retrieve(
        "plan pricing per month", where={"source": "pricing.md"}
    )
    assert outcome.documents
    assert {d.source for d in outcome.documents} == {"pricing.md"}


async def test_retrieval_honours_top_k(seeded_container: Container):
    outcome = await seeded_container.retriever.retrieve("support", top_k=1)
    assert len(outcome.documents) == 1


async def test_retrieval_cache_hits_on_repeat(seeded_container: Container):
    first = await seeded_container.retriever.retrieve("support hours")
    second = await seeded_container.retriever.retrieve("  SUPPORT   Hours  ")

    assert first.cache_hit is False
    assert second.cache_hit is True  # case and whitespace are normalised into the key


async def test_empty_query_returns_nothing(seeded_container: Container):
    outcome = await seeded_container.retriever.retrieve("   ")
    assert outcome.documents == []


# -------------------------------------------------------------------- answers
async def test_answer_is_grounded_and_cites_sources(seeded_container: Container):
    answer = await seeded_container.pipeline.answer("What are the support hours?")

    assert answer.grounded is True
    assert answer.citations
    assert answer.citations[0].source in {"handbook.md", "pricing.md"}
    assert "retrieval_ms" in answer.timings


async def test_context_block_reaches_the_model(seeded_container: Container, llm: FakeLLM):
    await seeded_container.pipeline.answer("What is the refund policy?")

    system, *_rest = llm.calls[-1]
    final_user = llm.calls[-1][-1]
    assert system.role == "system"
    assert "GROUNDING RULES" in system.content
    assert "KNOWLEDGE" in final_user.content
    assert "refund" in final_user.content.lower()


async def test_small_talk_skips_retrieval(seeded_container: Container):
    answer = await seeded_container.pipeline.answer("hello")

    assert answer.retrieved_count == 0
    assert answer.grounded is False


async def test_empty_question_short_circuits(seeded_container: Container, llm: FakeLLM):
    before = len(llm.calls)
    answer = await seeded_container.pipeline.answer("")

    assert "didn't catch that" in answer.answer
    assert len(llm.calls) == before  # no LLM call at all


async def test_voice_mode_strips_markdown(
    seeded_container: Container, settings, embeddings, store
):
    from app.rag.ingestion import IngestionService
    from app.rag.retriever import Retriever

    markdown_llm = FakeLLM(reply="**Support** is open `9-6` PT. See [docs](http://x.io) & more.")
    pipeline = RAGPipeline(
        settings=settings,
        embeddings=embeddings,
        vector_store=store,
        llm=markdown_llm,
        retriever=Retriever(embeddings=embeddings, vector_store=store, config=settings.rag),
        ingestion=IngestionService(embeddings=embeddings, vector_store=store, config=settings.rag),
    )

    answer = await pipeline.answer("support hours", mode="voice")

    assert "**" not in answer.answer
    assert "`" not in answer.answer
    assert "http" not in answer.answer
    assert "&" not in answer.answer
    assert "and" in answer.answer


async def test_provider_failure_returns_a_spoken_fallback(
    seeded_container: Container, settings, embeddings, store
):
    from app.rag.ingestion import IngestionService
    from app.rag.retriever import Retriever

    pipeline = RAGPipeline(
        settings=settings,
        embeddings=embeddings,
        vector_store=store,
        llm=FakeLLM(fail=True),
        retriever=Retriever(embeddings=embeddings, vector_store=store, config=settings.rag),
        ingestion=IngestionService(embeddings=embeddings, vector_store=store, config=settings.rag),
    )

    answer = await pipeline.answer("support hours")

    # A live caller must hear something, never a 500.
    assert "trouble" in answer.answer.lower()


async def test_streaming_yields_deltas(seeded_container: Container):
    chunks = [c async for c in seeded_container.pipeline.stream_answer("support hours")]

    assert chunks
    assert "support" in "".join(chunks).lower()


async def test_history_is_passed_through(seeded_container: Container, llm: FakeLLM):
    from app.rag.llm.base import ChatMessage

    history = [
        ChatMessage(role="user", content="Do you offer refunds?"),
        ChatMessage(role="assistant", content="Yes, within thirty days."),
    ]
    await seeded_container.pipeline.answer("And after that?", history=history)

    contents = [m.content for m in llm.calls[-1]]
    assert "Do you offer refunds?" in contents
    assert "Yes, within thirty days." in contents


# --------------------------------------------------------------------- health
async def test_pipeline_health_reports_components(seeded_container: Container):
    health = await seeded_container.pipeline.health()

    assert health["knowledge_base_ready"] is True
    assert health["vector_store"]["vectors"] > 0
    assert health["llm"]["model"] == "fake-model"


@pytest.mark.parametrize("utterance", ["hi", "thanks", "ok", "bye", "yep"])
async def test_common_small_talk_is_recognised(seeded_container: Container, utterance: str):
    answer = await seeded_container.pipeline.answer(utterance)
    assert answer.retrieved_count == 0
