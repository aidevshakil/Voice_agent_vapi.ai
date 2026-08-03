"""Request/response models for the first-party RAG and document APIs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.rag.pipeline import RAGAnswer


class CitationModel(BaseModel):
    source: str
    score: float
    page: int | None = None
    snippet: str = ""


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None, description="Reuse a session id to keep conversational context."
    )
    mode: Literal["voice", "text"] = "text"
    top_k: int | None = Field(default=None, ge=1, le=20)
    source: str | None = Field(default=None, description="Restrict retrieval to one source file.")
    use_cache: bool = True
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationModel] = Field(default_factory=list)
    grounded: bool = False
    retrieved_count: int = 0
    top_score: float = 0.0
    cache_hit: bool = False
    model: str = ""
    session_id: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_answer(cls, answer: RAGAnswer, session_id: str | None = None) -> QueryResponse:
        return cls(
            answer=answer.answer,
            citations=[CitationModel(**c.as_dict()) for c in answer.citations],
            grounded=answer.grounded,
            retrieved_count=answer.retrieved_count,
            top_score=round(answer.top_score, 4),
            cache_hit=answer.cache_hit,
            model=answer.model,
            session_id=session_id,
            timings=answer.timings,
        )


class RetrieveRequest(BaseModel):
    """Retrieval-only, for debugging chunking and relevance without LLM cost."""

    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None
    use_cache: bool = False


class RetrievedChunk(BaseModel):
    text: str
    score: float
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    candidates_considered: int = 0
    cache_hit: bool = False
    timings: dict[str, float] = Field(default_factory=dict)


# ------------------------------------------------------------------ documents
class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(min_length=1, max_length=200, description="Label used in citations.")
    metadata: dict[str, Any] | None = None


class ReindexRequest(BaseModel):
    directory: str | None = Field(
        default=None, description="Defaults to RAG_DOCUMENTS_DIR."
    )
    recursive: bool = True
    reset: bool = Field(
        default=False, description="Drop all existing vectors before indexing."
    )


class IngestionResponse(BaseModel):
    files_processed: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    duplicates_skipped: int = 0
    sources: list[str] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)
    duration_ms: float = 0.0
    total_vectors: int = 0


class SourceSummary(BaseModel):
    source: str


class DocumentListResponse(BaseModel):
    provider: str
    collection: str
    total_vectors: int
    dimensions: int
    sources: list[str] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    deleted: int
    source: str | None = None
    total_vectors: int = 0


# --------------------------------------------------------------------- health
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    knowledge_base_ready: bool = False
    components: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session_id: str
    turns: list[dict[str, Any]] = Field(default_factory=list)
