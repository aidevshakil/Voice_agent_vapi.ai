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


def _keyterm_field(model: str) -> str:
    """Deepgram renamed keyword boosting between model generations.

    nova-3 uses ``keyterm``, which accepts multi-word phrases. Older models use
    ``keywords``, which Vapi validates against a single-token regex -- sending a
    phrase there fails the whole assistant update.
    """
    return "keyterm" if model.startswith("nova-3") else "keywords"


def _server_block(url: str, secret: Any | None) -> dict[str, Any]:
    """A Vapi ``server`` object, with the shared secret when one is configured."""
    block: dict[str, Any] = {"url": url}
    if secret is not None:
        block["secret"] = secret.get_secret_value()
    return block


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

        transcriber: dict[str, Any] = {
            "provider": cfg.transcriber_provider,
            "model": cfg.transcriber_model,
            "language": cfg.transcriber_language,
            "confidenceThreshold": cfg.transcriber_confidence_threshold,
        }
        if cfg.transcriber_keyterms:
            field = _keyterm_field(cfg.transcriber_model)
            terms = cfg.transcriber_keyterms
            if field == "keywords":
                # Older models take single tokens only, so phrases are split.
                terms = [word for term in terms for word in term.split()]
            transcriber[field] = terms

        payload: dict[str, Any] = {
            "name": name,
            "firstMessage": first_message
            or "Hi, I'm Aria. Ask me anything about the documents I've been given.",
            "transcriber": transcriber,
            "voice": {"provider": cfg.voice_provider, "voiceId": cfg.voice_id},
            "server": _server_block(f"{base}/api/v1/vapi/webhook", cfg.webhook_secret),
            "serverMessages": [
                "status-update",
                "end-of-call-report",
                "transcript",
                "tool-calls",
            ],
            "silenceTimeoutSeconds": 300,
            # Turn-taking. The aggressive defaults chase latency and end the
            # caller's turn mid-sentence -- a pause for breath reads as "done",
            # so the model answers a fragment ("Is", "Please check"). These
            # values trade ~0.3 s of response time for whole questions.
            "startSpeakingPlan": {
                "waitSeconds": 0.7,
                # LiveKit predicts end-of-turn from the words themselves rather
                # than silence alone, which is what saves a mid-sentence pause.
                "smartEndpointingPlan": {"provider": "livekit"},
                "transcriptionEndpointingPlan": {
                    "onPunctuationSeconds": 0.3,
                    # The real fix: an unfinished sentence gets ~2 s of grace.
                    "onNoPunctuationSeconds": 2.0,
                    "onNumberSeconds": 0.6,
                },
            },
            # Interruption sensitivity moved under stopSpeakingPlan; the old
            # top-level numWordsToInterruptAssistantSpeech is now rejected with
            # "property should not exist" and fails the whole request.
            # numWords=2 let a cough or "mm-hmm" cut the assistant off.
            "stopSpeakingPlan": {"numWords": 3, "voiceSeconds": 0.3, "backoffSeconds": 1.0},
            # Replaces the retired `backgroundDenoisingEnabled` flag.
            "backgroundSpeechDenoisingPlan": {"smartDenoisingPlan": {"enabled": True}},
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
            # `server.secret` only guards the webhook URL -- Vapi sends nothing to
            # the custom-llm URL unless we ask for it here, and the endpoint's auth
            # dependency then 401s every turn, ending the call with
            # `pipeline-error-custom-llm-401-unauthorized`.
            if secret := cfg.webhook_secret:
                payload["model"]["headers"] = {"x-vapi-secret": secret.get_secret_value()}
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
                        # A tool-level server overrides the assistant's, secret
                        # included, so it has to carry the secret itself.
                        "server": _server_block(f"{base}/api/v1/vapi/webhook", cfg.webhook_secret),
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
