"""RAG orchestration: the single entry point every API route funnels through.

Two generation modes:

* :meth:`RAGPipeline.answer` — buffered, used by the text API and the Vapi tool
  webhook (both need a complete string).
* :meth:`RAGPipeline.stream_answer` — token stream, used by the Vapi custom-LLM
  endpoint. Streaming is what keeps time-to-first-audio low: Vapi starts
  synthesising speech from the first sentence instead of waiting for the whole
  completion.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from app.core.config import Settings
from app.core.exceptions import AppError, ProviderError
from app.core.logging import get_logger
from app.rag import prompts
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.ingestion import IngestionService
from app.rag.llm.base import ChatMessage, LLMProvider
from app.rag.retriever import RetrievalOutcome, Retriever
from app.rag.vectorstores.base import ScoredDocument, VectorStore
from app.utils.text import SentenceBuffer, to_speech_friendly
from app.utils.timing import Stopwatch

logger = get_logger(__name__)

Mode = Literal["voice", "text"]

# Short utterances that carry no retrievable intent. Running the full pipeline on
# them wastes ~300 ms per turn and returns irrelevant chunks.
_SMALL_TALK = {
    "hi", "hello", "hey", "yo", "thanks", "thank you", "thankyou", "ok", "okay",
    "yes", "no", "yeah", "yep", "nope", "bye", "goodbye", "cool", "nice", "sure",
    "got it", "great", "hmm", "uh huh", "right",
}


@dataclass(slots=True)
class Citation:
    source: str
    score: float
    page: int | None = None
    snippet: str = ""

    @classmethod
    def from_scored(cls, scored: ScoredDocument, *, snippet_chars: int = 240) -> Citation:
        page = scored.metadata.get("page")
        return cls(
            source=scored.source,
            score=round(scored.score, 4),
            page=int(page) if isinstance(page, (int, float, str)) and str(page).isdigit() else None,
            snippet=scored.text[:snippet_chars].strip(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "score": self.score, "page": self.page, "snippet": self.snippet}


@dataclass(slots=True)
class RAGAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = False
    retrieved_count: int = 0
    top_score: float = 0.0
    cache_hit: bool = False
    model: str = ""
    timings: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.as_dict() for c in self.citations],
            "grounded": self.grounded,
            "retrieved_count": self.retrieved_count,
            "top_score": round(self.top_score, 4),
            "cache_hit": self.cache_hit,
            "model": self.model,
            "timings": self.timings,
        }


class RAGPipeline:
    """Wires retriever + LLM + prompts into answer generation."""

    def __init__(
        self,
        *,
        settings: Settings,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        llm: LLMProvider,
        retriever: Retriever,
        ingestion: IngestionService,
    ) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._store = vector_store
        self._llm = llm
        self._retriever = retriever
        self._ingestion = ingestion

    # -------------------------------------------------------------- accessors
    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def ingestion(self) -> IngestionService:
        return self._ingestion

    @property
    def vector_store(self) -> VectorStore:
        return self._store

    @property
    def embeddings(self) -> EmbeddingProvider:
        return self._embeddings

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _is_small_talk(question: str) -> bool:
        cleaned = question.lower().strip(" .,!?~")
        return bool(cleaned) and (cleaned in _SMALL_TALK or len(cleaned) <= 2)

    def _system_prompt(self, mode: Mode, extra: str | None = None) -> str:
        base = prompts.VOICE_SYSTEM_PROMPT if mode == "voice" else prompts.TEXT_SYSTEM_PROMPT
        return f"{base}\n\n{extra.strip()}" if extra else base

    def _build_messages(
        self,
        question: str,
        outcome: RetrievalOutcome,
        *,
        mode: Mode,
        history: Sequence[ChatMessage] = (),
        system_prompt_extra: str | None = None,
    ) -> list[ChatMessage]:
        context = prompts.build_context_block(
            outcome.documents,
            max_chars=self._settings.rag.max_context_chars,
            include_scores=mode == "text",
        )
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self._system_prompt(mode, system_prompt_extra))
        ]
        # Only user/assistant turns; any system prompt in history is the caller's
        # old copy and would conflict with ours.
        messages.extend(m for m in history if m.role in ("user", "assistant"))
        messages.append(ChatMessage(role="user", content=prompts.build_user_turn(question, context)))
        return messages

    def _postprocess(self, text: str, mode: Mode) -> str:
        text = text.strip()
        return to_speech_friendly(text) if mode == "voice" else text

    async def _retrieve(
        self,
        question: str,
        *,
        top_k: int | None,
        where: dict[str, Any] | None,
        use_cache: bool,
    ) -> RetrievalOutcome:
        if self._is_small_talk(question):
            logger.debug("skipping retrieval for small talk: %r", question)
            return RetrievalOutcome([], 0, False, {})
        return await self._retriever.retrieve(
            question, top_k=top_k, where=where, use_cache=use_cache
        )

    # ---------------------------------------------------------------- buffered
    async def answer(
        self,
        question: str,
        *,
        mode: Mode = "voice",
        history: Sequence[ChatMessage] = (),
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        use_cache: bool = True,
        system_prompt_extra: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> RAGAnswer:
        watch = Stopwatch()
        question = (question or "").strip()
        if not question:
            return RAGAnswer(answer=prompts.FALLBACK_ANSWERS["empty_question"])

        outcome = await self._retrieve(question, top_k=top_k, where=where, use_cache=use_cache)
        retrieval_ms = watch.mark("retrieval_ms")

        messages = self._build_messages(
            question, outcome, mode=mode, history=history, system_prompt_extra=system_prompt_extra
        )

        try:
            with watch.stage("generation_ms"):
                completion = await self._llm.complete(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
            text = self._postprocess(completion.text, mode)
            model = completion.model
        except AppError as exc:
            logger.error("generation failed: %s", exc.message)
            text, model = prompts.FALLBACK_ANSWERS["provider_error"], self._llm.model
        except Exception as exc:  # never let an LLM failure become a 500 on a live call
            logger.exception("unexpected generation failure: %s", exc)
            text, model = prompts.FALLBACK_ANSWERS["provider_error"], self._llm.model

        timings = {
            "retrieval_ms": retrieval_ms,
            "generation_ms": watch.stages.get("generation_ms", 0.0),
            **{f"retrieval_{k}": v for k, v in outcome.timings.items() if k != "total_ms"},
            "total_ms": watch.total_ms,
        }
        answer = RAGAnswer(
            answer=text or prompts.FALLBACK_ANSWERS["no_context"],
            citations=[Citation.from_scored(d) for d in outcome.documents],
            grounded=bool(outcome.documents),
            retrieved_count=len(outcome.documents),
            top_score=outcome.top_score,
            cache_hit=outcome.cache_hit,
            model=model,
            timings=timings,
        )
        logger.info(
            "answered grounded=%s chunks=%d top=%.2f cache=%s total=%.0fms",
            answer.grounded,
            answer.retrieved_count,
            answer.top_score,
            outcome.cache_hit,
            timings["total_ms"],
        )
        return answer

    # --------------------------------------------------------------- streaming
    async def stream_answer(
        self,
        question: str,
        *,
        mode: Mode = "voice",
        history: Sequence[ChatMessage] = (),
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        use_cache: bool = True,
        system_prompt_extra: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield answer deltas. Errors surface as a spoken fallback, not a raise.

        Once the HTTP response has started streaming there is no way to send an
        error status, and a silent assistant is the worst possible outcome on a
        phone call — so failures mid-stream emit an apology the caller can hear.
        """
        watch = Stopwatch()
        question = (question or "").strip()
        if not question:
            yield prompts.FALLBACK_ANSWERS["empty_question"]
            return

        try:
            outcome = await self._retrieve(question, top_k=top_k, where=where, use_cache=use_cache)
        except AppError as exc:
            logger.error("retrieval failed mid-stream: %s", exc.message)
            yield prompts.FALLBACK_ANSWERS["provider_error"]
            return

        retrieval_ms = watch.mark("retrieval_ms")
        messages = self._build_messages(
            question, outcome, mode=mode, history=history, system_prompt_extra=system_prompt_extra
        )

        first_token_ms: float | None = None
        emitted = False
        # Voice mode releases whole normalised sentences so markdown split across
        # deltas can't leak into TTS; text mode passes deltas through untouched.
        buffer = SentenceBuffer(transform=to_speech_friendly) if mode == "voice" else None
        try:
            async for delta in self._llm.stream(
                messages, temperature=temperature, max_tokens=max_tokens
            ):
                if first_token_ms is None:
                    first_token_ms = watch.mark("first_token_ms")
                if buffer is None:
                    emitted = True
                    yield delta
                    continue
                for sentence in buffer.push(delta):
                    emitted = True
                    yield sentence + " "
            if buffer is not None and (tail := buffer.drain()):
                emitted = True
                yield tail
        except (AppError, ProviderError) as exc:
            logger.error("stream failed: %s", exc)
            if not emitted:
                yield prompts.FALLBACK_ANSWERS["provider_error"]
            return
        except Exception as exc:
            logger.exception("unexpected stream failure: %s", exc)
            if not emitted:
                yield prompts.FALLBACK_ANSWERS["provider_error"]
            return

        if not emitted:
            yield prompts.FALLBACK_ANSWERS["no_context"]

        logger.info(
            "streamed grounded=%s chunks=%d retrieval=%.0fms ttft=%.0fms total=%.0fms",
            bool(outcome.documents),
            len(outcome.documents),
            retrieval_ms,
            first_token_ms or -1,
            watch.total_ms,
        )

    # -------------------------------------------------------------- diagnostics
    async def health(self) -> dict[str, Any]:
        stats = self._store.stats()
        return {
            "llm": {"provider": self._llm.name, "model": self._llm.model},
            "embeddings": {
                "provider": self._embeddings.name,
                "model": self._settings.embedding.model,
            },
            "vector_store": {
                "provider": stats.provider,
                "collection": stats.collection,
                "vectors": stats.vectors,
                "dimensions": stats.dimensions,
                "sources": stats.sources,
            },
            "retrieval_cache": self._retriever.cache_stats,
            "knowledge_base_ready": stats.vectors > 0,
        }
