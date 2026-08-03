"""Ingestion: load -> chunk -> embed -> upsert.

Chunk ids are content hashes, which makes ingestion **idempotent**: re-running it
over an unchanged corpus rewrites the same ids instead of duplicating vectors,
and an edited paragraph produces a new id while its neighbours stay put.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import RAGSettings
from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.loaders import SUPPORTED_EXTENSIONS, load_document
from app.rag.vectorstores.base import VectorStore
from app.utils.text import content_hash
from app.utils.timing import Stopwatch

logger = get_logger(__name__)

# Splitting on structural boundaries first keeps paragraphs and sentences whole,
# which matters more for retrieval quality than hitting an exact chunk size.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

_MIN_CHUNK_CHARS = 40


@dataclass(slots=True)
class IngestionResult:
    files_processed: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    duplicates_skipped: int = 0
    sources: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    duration_ms: float = 0.0

    def merge(self, other: IngestionResult) -> None:
        self.files_processed += other.files_processed
        self.files_failed += other.files_failed
        self.chunks_created += other.chunks_created
        self.chunks_indexed += other.chunks_indexed
        self.duplicates_skipped += other.duplicates_skipped
        self.sources.extend(other.sources)
        self.errors.extend(other.errors)

    def as_dict(self) -> dict[str, object]:
        return {
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "chunks_created": self.chunks_created,
            "chunks_indexed": self.chunks_indexed,
            "duplicates_skipped": self.duplicates_skipped,
            "sources": sorted(set(self.sources)),
            "errors": self.errors,
            "duration_ms": self.duration_ms,
        }


class IngestionService:
    """Builds and maintains the vector index."""

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
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=_SEPARATORS,
            length_function=len,
            keep_separator=True,
        )
        # Serialise writes: embedding models and Chroma collections are not
        # guaranteed safe under concurrent upserts from multiple requests.
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------ chunk
    def chunk(self, documents: Sequence[Document]) -> list[Document]:
        """Split documents and attach positional metadata to each chunk."""
        chunks: list[Document] = []
        for document in documents:
            for index, piece in enumerate(self._splitter.split_documents([document])):
                text = piece.page_content.strip()
                if len(text) < _MIN_CHUNK_CHARS:
                    continue  # headings and stray fragments retrieve poorly
                chunks.append(
                    Document(
                        page_content=text,
                        metadata={**piece.metadata, "chunk_index": index, "chunk_chars": len(text)},
                    )
                )
        return chunks

    @staticmethod
    def chunk_id(chunk: Document) -> str:
        """Content-addressed id: same text under the same source -> same id."""
        return content_hash(
            str(chunk.metadata.get("source", "")),
            str(chunk.metadata.get("page", "")),
            chunk.page_content,
        )

    def index_sync(self, documents: Sequence[Document]) -> int:
        """Blocking chunk-embed-upsert, for scripts and tests.

        Skips the async write lock, so only call it when nothing else is writing.
        """
        chunks = self.chunk(documents)
        if not chunks:
            return 0
        unique = {self.chunk_id(c): c for c in chunks}
        payload = list(unique.values())
        vectors = self._embeddings.embed_documents([c.page_content for c in payload])
        return self._store.upsert(payload, vectors, list(unique))

    # ----------------------------------------------------------------- ingest
    async def ingest_documents(self, documents: Sequence[Document]) -> IngestionResult:
        """Chunk, embed and upsert already-loaded documents."""
        watch = Stopwatch()
        result = IngestionResult()
        if not documents:
            return result

        with watch.stage("chunk_ms"):
            chunks = self.chunk(documents)
        result.chunks_created = len(chunks)
        if not chunks:
            result.duration_ms = watch.total_ms
            return result

        # Deduplicate within the batch; content hashing handles cross-batch dupes.
        unique: dict[str, Document] = {}
        for chunk in chunks:
            unique.setdefault(self.chunk_id(chunk), chunk)
        result.duplicates_skipped = len(chunks) - len(unique)

        ids = list(unique)
        payload = list(unique.values())

        async with self._write_lock:
            with watch.stage("embed_ms"):
                vectors = await self._embeddings.aembed_documents([c.page_content for c in payload])
            with watch.stage("upsert_ms"):
                result.chunks_indexed = await self._store.aupsert(payload, vectors, ids)

        result.sources = sorted({str(c.metadata.get("source", "unknown")) for c in payload})
        result.duration_ms = watch.total_ms
        logger.info(
            "ingested chunks=%d sources=%s %s",
            result.chunks_indexed,
            result.sources,
            watch,
        )
        return result

    async def ingest_file(self, path: Path) -> IngestionResult:
        path = Path(path)
        try:
            documents = await asyncio.to_thread(load_document, path)
        except IngestionError as exc:
            return IngestionResult(files_failed=1, errors=[{"file": path.name, "error": exc.message}])

        if not documents:
            return IngestionResult(
                files_failed=1, errors=[{"file": path.name, "error": "no extractable text"}]
            )

        result = await self.ingest_documents(documents)
        result.files_processed = 1
        return result

    async def ingest_text(
        self, text: str, *, source: str, metadata: dict[str, object] | None = None
    ) -> IngestionResult:
        if not text.strip():
            raise IngestionError("cannot ingest empty text")
        document = Document(
            page_content=text,
            metadata={"source": source, "file_type": "text", **(metadata or {})},
        )
        result = await self.ingest_documents([document])
        result.files_processed = 1
        return result

    async def ingest_directory(
        self, directory: Path | None = None, *, recursive: bool = True
    ) -> IngestionResult:
        """Ingest every supported file under ``directory``, one file at a time.

        Sequential by design: embedding is CPU-bound for local models, so fanning
        out would only cause thread contention, and per-file isolation means one
        corrupt document can't fail the whole run.
        """
        directory = Path(directory or self._config.documents_dir)
        if not directory.is_dir():
            raise IngestionError(f"documents directory not found: {directory}")

        pattern = "**/*" if recursive else "*"
        files = sorted(
            path
            for path in directory.glob(pattern)
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not path.name.startswith((".", "~"))
        )
        if not files:
            logger.warning("no supported documents found in %s", directory)
            return IngestionResult()

        logger.info("ingesting %d file(s) from %s", len(files), directory)
        watch = Stopwatch()
        total = IngestionResult()
        for path in files:
            total.merge(await self.ingest_file(path))
        total.duration_ms = watch.total_ms
        logger.info(
            "ingestion complete files=%d failed=%d chunks=%d in %.0fms",
            total.files_processed,
            total.files_failed,
            total.chunks_indexed,
            total.duration_ms,
        )
        return total

    # ----------------------------------------------------------------- delete
    async def delete_source(self, source: str) -> int:
        async with self._write_lock:
            return await asyncio.to_thread(self._store.delete, where={"source": source})

    async def reset(self) -> None:
        async with self._write_lock:
            await asyncio.to_thread(self._store.reset)
