#!/usr/bin/env python
"""Build the vector index from ./data/documents (or any path you pass).

    python scripts/ingest.py                     # ingest RAG_DOCUMENTS_DIR
    python scripts/ingest.py --reset             # rebuild from scratch
    python scripts/ingest.py --path ./my-docs    # ingest somewhere else
    python scripts/ingest.py --stats             # just report what's indexed
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.rag.embeddings import build_embedding_provider
from app.rag.ingestion import IngestionService
from app.rag.vectorstores import build_vector_store

logger = get_logger("ingest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, default=None, help="Directory or single file to ingest.")
    parser.add_argument("--reset", action="store_true", help="Drop all existing vectors first.")
    parser.add_argument("--no-recursive", action="store_true", help="Don't descend into subdirectories.")
    parser.add_argument("--stats", action="store_true", help="Print index stats and exit.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(level="DEBUG" if args.verbose else "INFO", json_output=False)

    embeddings = build_embedding_provider(settings)
    logger.info("warming up %s/%s", settings.embedding.provider, settings.embedding.model)
    await asyncio.to_thread(embeddings.warmup)

    store = build_vector_store(settings, embeddings.dimensions)
    service = IngestionService(embeddings=embeddings, vector_store=store, config=settings.rag)

    if args.stats:
        stats = store.stats()
        print(f"\nprovider:   {stats.provider}")
        print(f"collection: {stats.collection}")
        print(f"vectors:    {stats.vectors}")
        print(f"dimensions: {stats.dimensions}")
        print(f"sources:    {len(stats.sources)}")
        for source in stats.sources:
            print(f"  - {source}")
        return 0

    if args.reset:
        logger.warning("resetting vector store")
        await service.reset()

    target = args.path or settings.rag.documents_dir
    if target.is_file():
        result = await service.ingest_file(target)
    else:
        result = await service.ingest_directory(target, recursive=not args.no_recursive)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"files processed   : {result.files_processed}")
    print(f"files failed      : {result.files_failed}")
    print(f"chunks created    : {result.chunks_created}")
    print(f"chunks indexed    : {result.chunks_indexed}")
    print(f"duplicates skipped: {result.duplicates_skipped}")
    print(f"duration          : {result.duration_ms / 1000:.2f}s")
    print(f"total vectors     : {store.count()}")
    if result.sources:
        print("\nsources:")
        for source in sorted(set(result.sources)):
            print(f"  - {source}")
    if result.errors:
        print("\nerrors:")
        for error in result.errors:
            print(f"  ! {error.get('file')}: {error.get('error')}")
    print("=" * 60)

    if result.chunks_indexed == 0:
        print(
            f"\nNothing was indexed. Put .pdf/.md/.txt/.docx/.html/.csv files in "
            f"{settings.rag.documents_dir} and run this again."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
