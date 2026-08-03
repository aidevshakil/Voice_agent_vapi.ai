"""Retrieval: embed query -> over-fetch -> MMR re-rank -> score filter -> cache.

Over-fetching ``fetch_k`` candidates and re-ranking down to ``top_k`` with
Maximal Marginal Relevance is what stops the model from seeing four
near-identical chunks of the same paragraph — a common failure mode that wastes
context and makes answers repetitive.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.core.config import RAGSettings
from app.core.exceptions import ProviderError, RetrievalError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.vectorstores.base import ScoredDocument, VectorStore
from app.services.cache import TTLCache
from app.utils.text import content_hash
from app.utils.timing import Stopwatch

logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievalOutcome:
    documents: list[ScoredDocument]
    candidates_considered: int
    cache_hit: bool
    timings: dict[str, float]

    @property
    def top_score(self) -> float:
        return self.documents[0].score if self.documents else 0.0

    @property
    def sources(self) -> list[str]:
        seen: dict[str, None] = {}
        for doc in self.documents:
            seen.setdefault(doc.source, None)
        return list(seen)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _token_set(text: str) -> set[str]:
    return {token for token in text.lower().split() if len(token) > 3}


def _mmr_rerank(
    candidates: list[ScoredDocument], k: int, lambda_mult: float
) -> list[ScoredDocument]:
    """Greedy MMR using lexical overlap as the redundancy signal.

    Chunk vectors aren't returned by every backend, so similarity between
    candidates is approximated with Jaccard overlap of their content words. It's
    coarse but reliably catches the near-duplicate case MMR exists to solve.
    """
    if len(candidates) <= k:
        return candidates

    tokens = [_token_set(c.text) for c in candidates]
    selected: list[int] = [0]  # highest-relevance candidate always goes first
    remaining = set(range(1, len(candidates)))

    while len(selected) < k and remaining:
        best_index, best_score = None, -math.inf
        for index in remaining:
            redundancy = 0.0
            for chosen in selected:
                union = tokens[index] | tokens[chosen]
                if union:
                    overlap = len(tokens[index] & tokens[chosen]) / len(union)
                    redundancy = max(redundancy, overlap)
            score = lambda_mult * candidates[index].score - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_index, best_score = index, score
        if best_index is None:
            break
        selected.append(best_index)
        remaining.discard(best_index)

    return [candidates[i] for i in selected]


class Retriever:
    """Query-time half of the RAG pipeline."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        config: RAGSettings,
    ) -> None:
        self._embeddings = embeddings
        self._store = vector_store
        self._config = config
        self._cache: TTLCache[list[ScoredDocument]] = TTLCache(
            max_size=config.query_cache_size, ttl_seconds=config.query_cache_ttl_seconds
        )

    @property
    def cache_stats(self) -> dict[str, float | int]:
        return {**self._cache.stats.as_dict(), "size": len(self._cache)}

    async def clear_cache(self) -> None:
        await self._cache.clear()

    def _cache_key(self, query: str, k: int, where: dict[str, Any] | None) -> str:
        # Normalising case/whitespace turns trivially different phrasings of the
        # same question into one cache entry.
        return content_hash(" ".join(query.lower().split()), str(k), repr(sorted((where or {}).items())))

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        where: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> RetrievalOutcome:
        cfg = self._config
        k = top_k or cfg.top_k
        floor = cfg.min_relevance_score if min_score is None else min_score
        query = query.strip()
        if not query:
            return RetrievalOutcome([], 0, False, {})

        watch = Stopwatch()
        cache_key = self._cache_key(query, k, where)
        if use_cache and (cached := await self._cache.get(cache_key)):
            return RetrievalOutcome(cached, len(cached), True, {"total_ms": watch.total_ms})

        try:
            with watch.stage("embed_ms"):
                vector = await self._embeddings.aembed_query(query)
            with watch.stage("search_ms"):
                fetch_k = max(cfg.fetch_k, k)
                candidates = await self._store.asearch(vector, fetch_k, where=where)
        except ProviderError as exc:
            raise RetrievalError(f"retrieval failed: {exc.message}") from exc

        considered = len(candidates)
        with watch.stage("rerank_ms"):
            kept = [c for c in candidates if c.score >= floor]
            # A too-aggressive floor is worse than a weak answer: if everything is
            # filtered out but the store did return something, keep the single best
            # hit and let the prompt's grounding rules handle low confidence.
            if not kept and candidates:
                kept = candidates[:1]
            documents = (
                _mmr_rerank(kept, k, cfg.mmr_lambda) if cfg.use_mmr else kept[:k]
            )[:k]

        if use_cache and documents:
            await self._cache.set(cache_key, documents)

        timings = watch.summary()
        logger.debug(
            "retrieved k=%d/%d top=%.3f %s",
            len(documents),
            considered,
            documents[0].score if documents else 0.0,
            watch,
        )
        return RetrievalOutcome(documents, considered, False, timings)
