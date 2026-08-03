"""Dependency-free numpy vector store with on-disk persistence.

Useful as (a) a zero-setup default when chromadb has no wheel for the running
Python, (b) the store used in tests, and (c) a fast path for small knowledge
bases -- an exact cosine scan over ~50k chunks of 384-d vectors is <10 ms.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document

from app.core.logging import get_logger
from app.rag.vectorstores.base import ScoredDocument, StoreStats, VectorStore

logger = get_logger(__name__)


def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    return all(metadata.get(key) == value for key, value in where.items())


class MemoryVectorStore(VectorStore):
    name = "memory"

    def __init__(self, collection: str, path: Path | None = None) -> None:
        self._collection = collection
        self._dir = Path(path) / collection if path else None
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._load()

    # ----------------------------------------------------------- persistence
    @property
    def _vector_file(self) -> Path | None:
        return self._dir / "vectors.npy" if self._dir else None

    @property
    def _meta_file(self) -> Path | None:
        return self._dir / "records.json" if self._dir else None

    def _load(self) -> None:
        if not (self._vector_file and self._meta_file):
            return
        if not (self._vector_file.exists() and self._meta_file.exists()):
            return
        try:
            vectors = np.load(self._vector_file)
            records = json.loads(self._meta_file.read_text(encoding="utf-8"))
            self._vectors = vectors.astype(np.float32, copy=False)
            self._ids = records["ids"]
            self._texts = records["texts"]
            self._metadatas = records["metadatas"]
            self._index = {doc_id: i for i, doc_id in enumerate(self._ids)}
            logger.info("memory store loaded vectors=%d", len(self._ids))
        except Exception as exc:  # corrupt cache is not fatal -- start empty
            logger.warning("memory store could not load snapshot (%s); starting empty", exc)
            self._reset_state()

    def _persist(self) -> None:
        if not (self._vector_file and self._meta_file):
            return
        self._dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        np.save(self._vector_file, self._vectors)
        self._meta_file.write_text(
            json.dumps(
                {"ids": self._ids, "texts": self._texts, "metadatas": self._metadatas},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _reset_state(self) -> None:
        self._ids, self._texts, self._metadatas = [], [], []
        self._index = {}
        self._vectors = np.zeros((0, 0), dtype=np.float32)

    # ------------------------------------------------------------ operations
    def upsert(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        ids: Sequence[str],
    ) -> int:
        if not documents:
            return 0
        matrix = np.asarray(embeddings, dtype=np.float32)
        # Pre-normalise so search is a single matmul.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-12, None)

        with self._lock:
            if self._vectors.size == 0:
                self._vectors = np.zeros((0, matrix.shape[1]), dtype=np.float32)
            elif self._vectors.shape[1] != matrix.shape[1]:
                raise ValueError(
                    f"embedding dimension changed ({self._vectors.shape[1]} -> "
                    f"{matrix.shape[1]}); reset the store before re-indexing"
                )

            new_rows: list[np.ndarray] = []
            for doc, vector, doc_id in zip(documents, matrix, ids, strict=True):
                existing = self._index.get(doc_id)
                if existing is not None:
                    self._vectors[existing] = vector
                    self._texts[existing] = doc.page_content
                    self._metadatas[existing] = dict(doc.metadata)
                    continue
                self._index[doc_id] = len(self._ids)
                self._ids.append(doc_id)
                self._texts.append(doc.page_content)
                self._metadatas.append(dict(doc.metadata))
                new_rows.append(vector)

            if new_rows:
                self._vectors = np.vstack([self._vectors, np.asarray(new_rows, dtype=np.float32)])
            self._persist()
        return len(documents)

    def search(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredDocument]:
        with self._lock:
            if not self._ids:
                return []
            query = np.asarray(embedding, dtype=np.float32)
            query /= max(float(np.linalg.norm(query)), 1e-12)
            similarities = self._vectors @ query

            candidate_idx = range(len(self._ids))
            if where:
                candidate_idx = [i for i in candidate_idx if _matches(self._metadatas[i], where)]
                if not candidate_idx:
                    return []

            scored = sorted(candidate_idx, key=lambda i: float(similarities[i]), reverse=True)[:k]
            return [
                ScoredDocument(
                    document=Document(
                        page_content=self._texts[i],
                        metadata={**self._metadatas[i], "id": self._ids[i]},
                    ),
                    # Map cosine [-1, 1] onto [0, 1].
                    score=(float(similarities[i]) + 1.0) / 2.0,
                )
                for i in scored
            ]

    def delete(self, *, ids: Sequence[str] | None = None, where: dict[str, Any] | None = None) -> int:
        with self._lock:
            if not self._ids:
                return 0
            drop: set[int] = set()
            if ids:
                drop.update(self._index[i] for i in ids if i in self._index)
            if where:
                drop.update(i for i, meta in enumerate(self._metadatas) if _matches(meta, where))
            if not drop:
                return 0

            keep = [i for i in range(len(self._ids)) if i not in drop]
            self._ids = [self._ids[i] for i in keep]
            self._texts = [self._texts[i] for i in keep]
            self._metadatas = [self._metadatas[i] for i in keep]
            self._vectors = self._vectors[keep] if keep else np.zeros((0, self._vectors.shape[1]), dtype=np.float32)
            self._index = {doc_id: i for i, doc_id in enumerate(self._ids)}
            self._persist()
            return len(drop)

    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def stats(self) -> StoreStats:
        with self._lock:
            sources = sorted({str(m.get("source", "unknown")) for m in self._metadatas})
            return StoreStats(
                provider=self.name,
                collection=self._collection,
                vectors=len(self._ids),
                dimensions=int(self._vectors.shape[1]) if self._vectors.size else 0,
                sources=sources,
            )

    def reset(self) -> None:
        with self._lock:
            self._reset_state()
            if self._vector_file and self._vector_file.exists():
                self._vector_file.unlink()
            if self._meta_file and self._meta_file.exists():
                self._meta_file.unlink()
