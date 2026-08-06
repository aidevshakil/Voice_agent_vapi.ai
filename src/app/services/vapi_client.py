"""Outbound Vapi REST client.

Used for provisioning (create/update the assistant so its config lives in this
repo rather than being hand-edited in a dashboard) and for placing outbound
calls. Inbound traffic from Vapi is handled by the API routes, not here.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, ProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class VapiClient:
    """Thin async wrapper over the Vapi API with a lazily-created HTTP client."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    @property
    def configured(self) -> bool:
        return self._settings.vapi.api_key is not None

    def _require_client(self) -> httpx.AsyncClient:
        if not self.configured:
            raise ConfigurationError("VAPI_API_KEY is not set.")
        if self._client is None:
            key = self._settings.vapi.api_key.get_secret_value()  # type: ignore[union-attr]
            self._client = httpx.AsyncClient(
                base_url=self._settings.vapi.base_url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        client = self._require_client()
        try:
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise ProviderError(
                f"vapi {method} {path} -> {exc.response.status_code}",
                details={"body": detail},
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"vapi {method} {path} failed: {exc}") from exc
        return response.json() if response.content else {}

    # ------------------------------------------------------------- assistants
    def build_assistant_payload(
        self,
        *,
        name: str = "RAG Voice Assistant",
        server_url: str | None = None,
        first_message: str | None = None,
        use_custom_llm: bool = True,
    ) -> dict[str, Any]:
        """Assistant config as code.

        ``use_custom_llm=True`` points Vapi's model at our own OpenAI-compatible
        endpoint, so every single turn is RAG-grounded. ``False`` instead gives a
        hosted model a ``search_knowledge_base`` tool, which is cheaper but lets
        the model answer from its own weights when it decides not to call the tool.
        """
        cfg = self._settings.vapi
        base = (server_url or cfg.server_url or "").rstrip("/")
        if not base:
            raise ConfigurationError(
                "VAPI_SERVER_URL must be a public HTTPS URL (e.g. an ngrok tunnel)."
            )

        payload: dict[str, Any] = {
            "name": name,
            "firstMessage": first_message
            or "Hi, I'm Aria. Ask me anything about the documents I've been given.",
            "transcriber": {
                "provider": cfg.transcriber_provider,
                "model": cfg.transcriber_model,
                "language": "en",
            },
            "voice": {"provider": cfg.voice_provider, "voiceId": cfg.voice_id},
            "server": {"url": f"{base}/api/v1/vapi/webhook"},
            "serverMessages": [
                "status-update",
                "end-of-call-report",
                "transcript",
                "tool-calls",
            ],
            # Keep replies short and let the caller interrupt -- both matter more
            # for perceived quality than raw model capability.
            "silenceTimeoutSeconds": 300,
            "responseDelaySeconds": 0.05,
            "llmRequestDelaySeconds": 0.05,
            # Interruption sensitivity moved under stopSpeakingPlan; the old
            # top-level numWordsToInterruptAssistantSpeech is now rejected with
            # "property should not exist" and fails the whole request.
            "stopSpeakingPlan": {"numWords": 2},
            "backgroundDenoisingEnabled": True,
            "endCallMessage": "Thanks for calling. Goodbye!",
        }

        if use_custom_llm:
            payload["model"] = {
                "provider": "custom-llm",
                "url": f"{base}/api/v1/vapi",
                "model": self._settings.llm.model,
                "temperature": self._settings.llm.temperature,
                "messages": [
                    {
                        "role": "system",
                        "content": "Answer strictly from the retrieved knowledge base.",
                    }
                ],
            }
        else:
            payload["model"] = {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": self._settings.llm.temperature,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Aria, a voice assistant. You MUST call "
                            "search_knowledge_base before answering any factual "
                            "question, and answer only from what it returns. Keep "
                            "replies to 1-3 spoken sentences."
                        ),
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_knowledge_base",
                            "description": (
                                "Search the private document knowledge base and return "
                                "a grounded answer. Call this for every factual question."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The user's question, as a standalone query.",
                                    }
                                },
                                "required": ["query"],
                            },
                        },
                        "server": {"url": f"{base}/api/v1/vapi/webhook"},
                    }
                ],
            }
        return payload

    async def create_assistant(self, payload: dict[str, Any]) -> dict[str, Any]:
        logger.info("creating vapi assistant name=%s", payload.get("name"))
        return await self._request("POST", "/assistant", json=payload)

    async def update_assistant(self, assistant_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        logger.info("updating vapi assistant id=%s", assistant_id)
        # `name` is immutable on update and Vapi rejects the whole request if sent.
        body = {k: v for k, v in payload.items() if k != "name"}
        return await self._request("PATCH", f"/assistant/{assistant_id}", json=body)

    async def get_assistant(self, assistant_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/assistant/{assistant_id}")

    async def list_assistants(self) -> list[dict[str, Any]]:
        result = await self._request("GET", "/assistant")
        return result if isinstance(result, list) else result.get("results", [])

    # ------------------------------------------------------------------ calls
    async def create_web_call(self, assistant_id: str | None = None) -> dict[str, Any]:
        target = assistant_id or self._settings.vapi.assistant_id
        if not target:
            raise ConfigurationError("VAPI_ASSISTANT_ID is not set.")
        return await self._request("POST", "/call/web", json={"assistantId": target})

    async def create_phone_call(
        self, *, to_number: str, phone_number_id: str, assistant_id: str | None = None
    ) -> dict[str, Any]:
        target = assistant_id or self._settings.vapi.assistant_id
        if not target:
            raise ConfigurationError("VAPI_ASSISTANT_ID is not set.")
        return await self._request(
            "POST",
            "/call/phone",
            json={
                "assistantId": target,
                "phoneNumberId": phone_number_id,
                "customer": {"number": to_number},
            },
        )

    async def get_call(self, call_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/call/{call_id}")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
