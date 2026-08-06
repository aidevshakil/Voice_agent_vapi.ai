"""Local ONNX embeddings via fastembed.

Default choice for a voice assistant: no network hop on the retrieval hot path,
no torch dependency, ~10-20 ms per query on CPU for bge-small.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Lock

from app.core.exceptions import ConfigurationError, ProviderError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

# bge/e5 models are trained with asymmetric prefixes; using them measurably
# improves retrieval quality and costs nothing.
_QUERY_PREFIXES = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
}
_DOC_PREFIXES = {"e5": "passage: "}


def _prefix_for(model: str, table: dict[str, str]) -> str:
    lowered = model.lower()
    for family, prefix in table.items():
        if family in lowered:
            return prefix
    return ""


class FastEmbedProvider(EmbeddingProvider):
    name = "fastembed"

    def __init__(self, model: str, batch_size: int = 64, cache_dir: Path | None = None) -> None:
        self._model_name = model
        self._batch_size = batch_size
        self._cache_dir = cache_dir
        self._query_prefix = _prefix_for(model, _QUERY_PREFIXES)
        self._doc_prefix = _prefix_for(model, _DOC_PREFIXES)
        self._model: object | None = None
        self._dimensions: int | None = None
        self._lock = Lock()

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover
                raise ConfigurationError(
                    "EMBEDDING_PROVIDER=fastembed requires `pip install fastembed`."
                ) from exc
            logger.info("loading fastembed model=%s (first run downloads weights)", self._model_name)
            try:
                kwargs: dict[str, object] = {"model_name": self._model_name}
                if self._cache_dir is not None:
                    self._cache_dir.mkdir(parents=True, exist_ok=True)
                    kwargs["cache_dir"] = str(self._cache_dir)
                self._model = TextEmbedding(**kwargs)  # type: ignore[arg-type]
            except Exception as exc:
                raise ProviderError(f"failed to load fastembed model: {exc}") from exc
            return self._model

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._dimensions = len(self.embed_query("dimension probe"))
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        prepared = [f"{self._doc_prefix}{t}" for t in texts]
        try:
            vectors = model.embed(prepared, batch_size=self._batch_size)  # type: ignore[attr-defined]
            return [vec.tolist() for vec in vectors]
        except Exception as exc:
            raise ProviderError(f"fastembed document embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        try:
            vectors = list(model.query_embed(f"{self._query_prefix}{text}"))  # type: ignore[attr-defined]
            if not vectors:
                raise ProviderError("fastembed returned no vector for query")
            return vectors[0].tolist()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"fastembed query embedding failed: {exc}") from exc
