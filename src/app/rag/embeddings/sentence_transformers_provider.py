"""sentence-transformers embeddings (opt-in; pulls in torch)."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

from app.core.exceptions import ConfigurationError, ProviderError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)


class SentenceTransformersProvider(EmbeddingProvider):
    name = "sentence_transformers"

    def __init__(self, model: str, batch_size: int = 32) -> None:
        self._model_name = model
        self._batch_size = batch_size
        self._model: object | None = None
        self._lock = Lock()

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ConfigurationError(
                    "EMBEDDING_PROVIDER=sentence_transformers requires "
                    "`pip install sentence-transformers`."
                ) from exc
            logger.info("loading sentence-transformers model=%s", self._model_name)
            try:
                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:
                raise ProviderError(f"failed to load model {self._model_name}: {exc}") from exc
            return self._model

    @property
    def dimensions(self) -> int:
        model = self._ensure_model()
        return int(model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        vector = model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
        return vector.tolist()
