"""OpenAI embeddings (``text-embedding-3-*``)."""

from __future__ import annotations

from collections.abc import Sequence

from app.core.exceptions import ConfigurationError, ProviderError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

_NATIVE_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None,
        *,
        dimensions: int | None = None,
        batch_size: int = 64,
    ) -> None:
        if not api_key:
            raise ConfigurationError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError("Install the `openai` package to use OpenAI embeddings.") from exc

        self._client = OpenAI(api_key=api_key, max_retries=3)
        self._model = model
        self._batch_size = batch_size
        native = _NATIVE_DIMENSIONS.get(model)
        # `dimensions` truncation is only supported by the v3 models.
        self._truncate = bool(dimensions and native and dimensions < native and "-3-" in model)
        self._dimensions = dimensions or native or 1536

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        kwargs: dict[str, object] = {"model": self._model, "input": inputs}
        if self._truncate:
            kwargs["dimensions"] = self._dimensions
        try:
            response = self._client.embeddings.create(**kwargs)  # type: ignore[arg-type]
        except Exception as exc:
            raise ProviderError(f"OpenAI embedding request failed: {exc}") from exc
        # The API preserves input order, but sorting by index is free insurance.
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [t or " " for t in texts[start : start + self._batch_size]]
            out.extend(self._embed(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text or " "])[0]
