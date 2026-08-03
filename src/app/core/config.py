"""Typed application settings.

Every tunable lives here so the rest of the codebase never touches ``os.environ``
directly. Settings are grouped into nested models by concern; env vars use the
group prefix (``LLM_MODEL`` -> ``settings.llm.model``).
"""

from __future__ import annotations

import sys
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _csv_list(value: object) -> object:
    """Accept either a JSON array or a plain comma-separated env value."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):  # let pydantic parse JSON arrays
            return value
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return value


CsvList = Annotated[list[str], BeforeValidator(_csv_list)]


class AppEnv(StrEnum):
    development = "development"
    staging = "staging"
    production = "production"


class AppSettings(BaseModel):
    env: AppEnv = AppEnv.development
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_json: bool = False
    reload: bool = True
    workers: int = 1
    cors_origins: CsvList = Field(default_factory=lambda: ["http://localhost:8501"])

    @property
    def is_production(self) -> bool:
        return self.env is AppEnv.production


class LLMSettings(BaseModel):
    provider: Literal["groq", "openai", "gemini"] = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=400, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)


class EmbeddingSettings(BaseModel):
    provider: Literal["fastembed", "openai", "sentence_transformers"] = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = Field(default=384, gt=0)
    batch_size: int = Field(default=64, gt=0)


class VectorStoreSettings(BaseModel):
    provider: Literal["chroma", "pinecone", "memory"] = "chroma"
    collection: str = "voice_kb"
    path: Path = PROJECT_ROOT / "data" / "vector_store"


class PineconeSettings(BaseModel):
    api_key: SecretStr | None = None
    index: str = "voice-kb"
    cloud: str = "aws"
    region: str = "us-east-1"


class RAGSettings(BaseModel):
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=4, gt=0)
    fetch_k: int = Field(default=16, gt=0)
    # Tuned for bge-small (the default embedder), whose cosine scores for English
    # text bottom out around 0.65 -- a lower floor would filter nothing at all.
    min_relevance_score: float = Field(default=0.75, ge=0.0, le=1.0)
    use_mmr: bool = True
    mmr_lambda: float = Field(default=0.6, ge=0.0, le=1.0)
    max_context_chars: int = Field(default=6000, gt=0)
    history_turns: int = Field(default=6, ge=0)
    query_cache_size: int = Field(default=512, ge=0)
    query_cache_ttl_seconds: int = Field(default=300, ge=0)
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"

    @model_validator(mode="after")
    def _check_invariants(self) -> RAGSettings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if self.fetch_k < self.top_k:
            raise ValueError("RAG_FETCH_K must be >= RAG_TOP_K")
        return self


class VapiSettings(BaseModel):
    api_key: SecretStr | None = None
    public_key: str | None = None
    assistant_id: str | None = None
    webhook_secret: SecretStr | None = None
    voice_provider: str = "11labs"
    voice_id: str = "burt"
    transcriber_provider: str = "deepgram"
    transcriber_model: str = "nova-3"
    server_url: str | None = None
    base_url: str = "https://api.vapi.ai"


class SecuritySettings(BaseModel):
    api_keys: CsvList = Field(default_factory=list)
    rate_limit_per_minute: int = Field(default=120, ge=0)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


class Settings(BaseSettings):
    """Root settings object. Instantiate through :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    vapi: VapiSettings = Field(default_factory=VapiSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    # Provider credentials are flat because the SDKs read these names too.
    groq_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    # ---------------------------------------------------------------- helpers
    def llm_api_key(self) -> str | None:
        """Return the raw key for the currently selected LLM provider."""
        mapping = {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "gemini": self.google_api_key,
        }
        secret = mapping.get(self.llm.provider)
        return secret.get_secret_value() if secret else None

    def require_llm_api_key(self) -> str:
        key = self.llm_api_key()
        if not key:
            env_name = {
                "groq": "GROQ_API_KEY",
                "openai": "OPENAI_API_KEY",
                "gemini": "GOOGLE_API_KEY",
            }[self.llm.provider]
            raise ValueError(
                f"LLM_PROVIDER={self.llm.provider} requires {env_name} to be set."
            )
        return key


# --------------------------------------------------------------------------- #
# Env vars are flat and prefixed (LLM_MODEL, RAG_TOP_K, ...). pydantic-settings
# maps nested models via a delimiter, so we translate the flat prefixes into the
# nested shape once, at load time.
# --------------------------------------------------------------------------- #
_GROUP_PREFIXES = {
    "APP_": "app",
    "LLM_": "llm",
    "EMBEDDING_": "embedding",
    "VECTOR_STORE_": "vector_store",
    "PINECONE_": "pinecone",
    "RAG_": "rag",
    "VAPI_": "vapi",
}
_SECURITY_KEYS = {"API_KEYS": "api_keys", "RATE_LIMIT_PER_MINUTE": "rate_limit_per_minute"}


def _read_env() -> dict[str, str]:
    import os

    env_path = PROJECT_ROOT / ".env"
    values: dict[str, str] = {}
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip().upper()] = val.split(" #")[0].strip().strip("'\"")
    # Real environment wins over the file.
    values.update({k.upper(): v for k, v in os.environ.items()})
    return values


def _build_nested(env: dict[str, str]) -> dict[str, object]:
    nested: dict[str, dict[str, str]] = {group: {} for group in _GROUP_PREFIXES.values()}
    nested["security"] = {}
    flat: dict[str, object] = {}

    for key, value in env.items():
        if value == "":
            continue
        if key in _SECURITY_KEYS:
            nested["security"][_SECURITY_KEYS[key]] = value
            continue
        for prefix, group in _GROUP_PREFIXES.items():
            if key.startswith(prefix):
                nested[group][key[len(prefix) :].lower()] = value
                break
        else:
            lowered = key.lower()
            if lowered in {"groq_api_key", "openai_api_key", "google_api_key"}:
                flat[lowered] = value

    # VAPI_API_KEY collides with the VAPI_ prefix rule above and lands correctly
    # as vapi.api_key, which is what we want -- no special-casing needed.
    return {**flat, **{group: data for group, data in nested.items() if data}}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings. Fails fast with a readable message."""
    try:
        return Settings(**_build_nested(_read_env()))  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - startup guard
        print(f"[config] invalid configuration: {exc}", file=sys.stderr)
        raise


settings = get_settings()
