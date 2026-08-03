"""Knowledge-base management: upload, ingest, reindex, inspect, delete."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.api.deps import ContainerDep, require_api_key
from app.core.exceptions import IngestionError, UnsupportedDocumentError
from app.core.logging import get_logger
from app.rag.ingestion import IngestionResult
from app.rag.loaders import SUPPORTED_EXTENSIONS
from app.schemas.rag import (
    DeleteResponse,
    DocumentListResponse,
    IngestionResponse,
    IngestTextRequest,
    ReindexRequest,
)

logger = get_logger(__name__)

router = APIRouter(tags=["documents"], dependencies=[Depends(require_api_key)])

MAX_UPLOAD_BYTES = 32 * 1024 * 1024  # 32 MB
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    """Strip directory components and hostile characters from an upload name."""
    cleaned = _UNSAFE_NAME.sub("_", Path(name).name).strip("._")
    return cleaned or "upload"


def _response(result: IngestionResult, total_vectors: int) -> IngestionResponse:
    return IngestionResponse(**result.as_dict(), total_vectors=total_vectors)  # type: ignore[arg-type]


@router.get("", response_model=DocumentListResponse, summary="Knowledge-base contents")
async def list_documents(container: ContainerDep) -> DocumentListResponse:
    stats = container.vector_store.stats()
    return DocumentListResponse(
        provider=stats.provider,
        collection=stats.collection,
        total_vectors=stats.vectors,
        dimensions=stats.dimensions,
        sources=stats.sources,
    )


@router.get("/supported-types", summary="File extensions this server can ingest")
async def supported_types() -> dict[str, list[str]]:
    return {"extensions": sorted(SUPPORTED_EXTENSIONS)}


@router.post("/upload", response_model=IngestionResponse, summary="Upload and index a file")
async def upload_document(
    container: ContainerDep,
    file: UploadFile = File(...),
    keep_copy: bool = Query(
        default=True,
        description="Also store the file in RAG_DOCUMENTS_DIR so a reindex can rebuild from it.",
    ),
) -> IngestionResponse:
    filename = _safe_filename(file.filename or "upload")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"unsupported file type {suffix!r}",
            details={"supported": sorted(SUPPORTED_EXTENSIONS)},
        )

    documents_dir = container.settings.rag.documents_dir
    scratch: tempfile.TemporaryDirectory[str] | None = None
    if keep_copy:
        documents_dir.mkdir(parents=True, exist_ok=True)
        target = documents_dir / filename
    else:
        # A temp *directory* rather than a prefixed temp file: the loader derives
        # `metadata["source"]` from the filename, and that name is what callers
        # see (and the assistant cites), so it must stay exactly as uploaded.
        scratch = tempfile.TemporaryDirectory(prefix="vapi_upload_")
        target = Path(scratch.name) / filename

    # Stream to disk with a size cap so a huge upload can't exhaust memory.
    written = 0
    try:
        with target.open("wb") as sink:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise IngestionError(
                        f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
                    )
                sink.write(chunk)
    except IngestionError:
        target.unlink(missing_ok=True)
        if scratch is not None:
            scratch.cleanup()
        raise
    finally:
        await file.close()

    logger.info("uploaded %s (%d bytes)", filename, written)
    try:
        result = await container.ingestion.ingest_file(target)
    finally:
        if scratch is not None:
            scratch.cleanup()

    return _response(result, container.vector_store.count())


@router.post("/text", response_model=IngestionResponse, summary="Index a raw text snippet")
async def ingest_text(body: IngestTextRequest, container: ContainerDep) -> IngestionResponse:
    result = await container.ingestion.ingest_text(
        body.text, source=body.source, metadata=body.metadata
    )
    return _response(result, container.vector_store.count())


@router.post(
    "/reindex",
    response_model=IngestionResponse,
    summary="Re-ingest the documents directory",
)
async def reindex(body: ReindexRequest, container: ContainerDep) -> IngestionResponse:
    if body.reset:
        logger.warning("resetting the vector store before reindex")
        await container.ingestion.reset()
        await container.pipeline.retriever.clear_cache()

    directory = Path(body.directory) if body.directory else None
    result = await container.ingestion.ingest_directory(directory, recursive=body.recursive)
    # Chunk ids are content hashes, so cached retrievals can point at text that
    # was just replaced. Flushing is cheap insurance.
    await container.pipeline.retriever.clear_cache()
    return _response(result, container.vector_store.count())


@router.delete("/{source}", response_model=DeleteResponse, summary="Delete one source's vectors")
async def delete_source(source: str, container: ContainerDep) -> DeleteResponse:
    deleted = await container.ingestion.delete_source(source)
    await container.pipeline.retriever.clear_cache()
    logger.info("deleted source=%s vectors=%d", source, deleted)
    return DeleteResponse(
        deleted=deleted, source=source, total_vectors=container.vector_store.count()
    )


@router.post("/reset", response_model=DeleteResponse, summary="Delete the entire knowledge base")
async def reset_knowledge_base(container: ContainerDep) -> DeleteResponse:
    before = container.vector_store.count()
    await container.ingestion.reset()
    await container.pipeline.retriever.clear_cache()
    logger.warning("knowledge base reset (%d vectors removed)", before)
    return DeleteResponse(deleted=before, total_vectors=container.vector_store.count())


@router.post("/copy-local", response_model=IngestionResponse, summary="Ingest a server-local path")
async def ingest_local_path(
    container: ContainerDep,
    path: str = Query(description="Path on the server, inside RAG_DOCUMENTS_DIR."),
) -> IngestionResponse:
    documents_dir = container.settings.rag.documents_dir.resolve()
    candidate = (documents_dir / path).resolve()
    # Confine to the documents directory; otherwise this is an arbitrary-file-read.
    if not candidate.is_relative_to(documents_dir):
        raise IngestionError("path must be inside the configured documents directory")
    if not candidate.is_file():
        raise IngestionError(f"file not found: {path}")

    result = await container.ingestion.ingest_file(candidate)
    return _response(result, container.vector_store.count())
