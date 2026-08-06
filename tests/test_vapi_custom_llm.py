"""The Vapi custom-LLM endpoint: OpenAI wire compatibility and SSE streaming."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

ENDPOINT = "/api/v1/vapi/chat/completions"


def _parse_sse(body: str) -> list[dict]:
    frames = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        frames.append(json.loads(payload))
    return frames


def test_non_streaming_response_matches_openai_shape(seeded_client: TestClient):
    response = seeded_client.post(
        ENDPOINT,
        json={
            "model": "gpt-4o-mini",
            "stream": False,
            "messages": [{"role": "user", "content": "What are the support hours?"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] > 0


def test_streaming_emits_valid_openai_chunks(seeded_client: TestClient):
    response = seeded_client.post(
        ENDPOINT,
        json={"stream": True, "messages": [{"role": "user", "content": "support hours?"}]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"  # or nginx would buffer it

    body = response.text
    assert body.rstrip().endswith("data: [DONE]")

    frames = _parse_sse(body)
    assert frames
    assert all(f["object"] == "chat.completion.chunk" for f in frames)
    # One stable id across the whole stream.
    assert len({f["id"] for f in frames}) == 1
    assert frames[0]["choices"][0]["delta"]["role"] == "assistant"
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"

    spoken = "".join(
        f["choices"][0]["delta"].get("content", "") for f in frames
    )
    assert "support" in spoken.lower()


def test_multipart_content_is_flattened(seeded_client: TestClient):
    """Vapi occasionally sends OpenAI's array-style content."""
    response = seeded_client.post(
        ENDPOINT,
        json={
            "stream": False,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "What is the refund policy?"}]}
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"]


def test_history_is_trimmed_to_configured_turns(seeded_client: TestClient, container):
    """A long call must not grow the prompt without bound."""
    turns = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(40)
    ]
    turns.append({"role": "user", "content": "What are the support hours?"})

    response = seeded_client.post(ENDPOINT, json={"stream": False, "messages": turns})

    assert response.status_code == 200
    sent = container.llm.calls[-1]
    # 1 system + at most history_turns of context + 1 grounded user turn.
    assert len(sent) <= container.settings.rag.history_turns + 2


def test_dashboard_system_prompt_is_appended_not_substituted(seeded_client: TestClient, container):
    seeded_client.post(
        ENDPOINT,
        json={
            "stream": False,
            "messages": [
                {"role": "system", "content": "Speak like a pirate."},
                {"role": "user", "content": "support hours?"},
            ],
        },
    )
    system = container.llm.calls[-1][0].content
    assert "Speak like a pirate." in system
    assert "GROUNDING RULES" in system  # our rules survive


def test_call_id_creates_a_session(seeded_client: TestClient, container):
    seeded_client.post(
        ENDPOINT,
        json={
            "stream": False,
            "call": {"id": "call_abc123"},
            "messages": [{"role": "user", "content": "support hours?"}],
        },
    )

    response = seeded_client.get("/api/v1/rag/sessions/call_abc123")
    assert response.status_code == 200
    assert len(response.json()["turns"]) == 2


def test_no_user_message_still_responds(seeded_client: TestClient):
    response = seeded_client.post(
        ENDPOINT, json={"stream": False, "messages": [{"role": "system", "content": "hi"}]}
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"]


def test_models_endpoint(seeded_client: TestClient):
    body = seeded_client.get("/api/v1/vapi/models").json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "fake-model"


# ------------------------------------------------------- assistant provisioning
def test_custom_llm_url_carries_the_shared_secret(container):
    """Vapi sends `server.secret` to the webhook only.

    Without an explicit header on the model block, every custom-llm turn arrives
    unauthenticated and the endpoint 401s, killing the call with
    `pipeline-error-custom-llm-401-unauthorized`.
    """
    from pydantic import SecretStr

    container.settings.vapi.webhook_secret = SecretStr("s3cret")
    payload = container.vapi.build_assistant_payload(server_url="https://example.test")

    assert payload["model"]["provider"] == "custom-llm"
    assert payload["model"]["headers"]["x-vapi-secret"] == "s3cret"
    assert payload["server"]["secret"] == "s3cret"


def test_tool_mode_server_override_carries_the_shared_secret(container):
    from pydantic import SecretStr

    container.settings.vapi.webhook_secret = SecretStr("s3cret")
    payload = container.vapi.build_assistant_payload(
        server_url="https://example.test", use_custom_llm=False
    )

    tool = payload["model"]["tools"][0]
    assert tool["server"]["secret"] == "s3cret"


def test_no_secret_configured_means_no_auth_fields(container):
    container.settings.vapi.webhook_secret = None
    payload = container.vapi.build_assistant_payload(server_url="https://example.test")

    assert "headers" not in payload["model"]
    assert "secret" not in payload["server"]


# ------------------------------------------------------------------- transcriber
def test_nova3_uses_keyterm_and_keeps_phrases(container):
    container.settings.vapi.transcriber_model = "nova-3"
    container.settings.vapi.transcriber_keyterms = ["Shakil Ahamed", "FastAPI"]

    transcriber = container.vapi.build_assistant_payload(
        server_url="https://example.test"
    )["transcriber"]

    assert transcriber["keyterm"] == ["Shakil Ahamed", "FastAPI"]
    assert "keywords" not in transcriber


def test_older_models_split_phrases_into_single_token_keywords(container):
    """Vapi validates `keywords` against a single-token regex; a phrase 400s."""
    container.settings.vapi.transcriber_model = "nova-2-phonecall"
    container.settings.vapi.transcriber_keyterms = ["Shakil Ahamed", "FastAPI"]

    transcriber = container.vapi.build_assistant_payload(
        server_url="https://example.test"
    )["transcriber"]

    assert transcriber["keywords"] == ["Shakil", "Ahamed", "FastAPI"]
    assert "keyterm" not in transcriber
    assert all(" " not in word for word in transcriber["keywords"])


def test_no_keyterms_configured_sends_no_boost_field(container):
    container.settings.vapi.transcriber_keyterms = []
    transcriber = container.vapi.build_assistant_payload(
        server_url="https://example.test"
    )["transcriber"]

    assert "keyterm" not in transcriber and "keywords" not in transcriber


def test_endpointing_gives_an_unfinished_sentence_room(container):
    """A pause for breath must not end the caller's turn mid-question."""
    payload = container.vapi.build_assistant_payload(server_url="https://example.test")

    plan = payload["startSpeakingPlan"]
    assert plan["transcriptionEndpointingPlan"]["onNoPunctuationSeconds"] >= 1.5
    assert plan["smartEndpointingPlan"]["provider"] == "livekit"
    # Retired fields fail schema validation on newer Vapi API versions.
    assert "responseDelaySeconds" not in payload
    assert "backgroundDenoisingEnabled" not in payload
