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
