"""Vapi server-webhook handling: tool calls, transcripts, lifecycle, auth."""

from __future__ import annotations

from fastapi.testclient import TestClient

WEBHOOK = "/api/v1/vapi/webhook"


def test_tool_call_returns_a_grounded_result(seeded_client: TestClient):
    response = seeded_client.post(
        WEBHOOK,
        json={
            "message": {
                "type": "tool-calls",
                "call": {"id": "call_1"},
                "toolCalls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {
                            "name": "search_knowledge_base",
                            "arguments": {"query": "What are the support hours?"},
                        },
                    }
                ],
            }
        },
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["toolCallId"] == "tc_1"
    assert results[0]["result"]


def test_tool_call_accepts_stringified_arguments(seeded_client: TestClient):
    response = seeded_client.post(
        WEBHOOK,
        json={
            "message": {
                "type": "tool-calls",
                "toolCallList": [
                    {
                        "id": "tc_2",
                        "function": {
                            "name": "search_knowledge_base",
                            "arguments": '{"query": "refund policy"}',
                        },
                    }
                ],
            }
        },
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["result"]


def test_unknown_tool_is_reported_not_crashed(seeded_client: TestClient):
    response = seeded_client.post(
        WEBHOOK,
        json={
            "message": {
                "type": "tool-calls",
                "toolCalls": [{"id": "tc_3", "function": {"name": "launch_rockets"}}],
            }
        },
    )
    assert response.status_code == 200
    assert "not available" in response.json()["results"][0]["result"]


def test_missing_query_returns_a_reprompt(seeded_client: TestClient):
    response = seeded_client.post(
        WEBHOOK,
        json={
            "message": {
                "type": "tool-calls",
                "toolCalls": [
                    {"id": "tc_4", "function": {"name": "search_knowledge_base", "arguments": {}}}
                ],
            }
        },
    )
    assert response.status_code == 200
    assert "didn't catch" in response.json()["results"][0]["result"]


def test_multiple_tool_calls_in_one_message(seeded_client: TestClient):
    response = seeded_client.post(
        WEBHOOK,
        json={
            "message": {
                "type": "tool-calls",
                "toolCalls": [
                    {
                        "id": f"tc_{i}",
                        "function": {"name": "search_knowledge_base", "arguments": {"query": q}},
                    }
                    for i, q in enumerate(["support hours", "refund policy"])
                ],
            }
        },
    )
    results = response.json()["results"]
    assert [r["toolCallId"] for r in results] == ["tc_0", "tc_1"]


def test_final_transcript_is_recorded(seeded_client: TestClient):
    seeded_client.post(
        WEBHOOK,
        json={
            "message": {
                "type": "transcript",
                "transcriptType": "final",
                "role": "user",
                "transcript": "Hello there",
                "call": {"id": "call_t"},
            }
        },
    )
    body = seeded_client.get("/api/v1/rag/sessions/call_t").json()
    assert body["turns"][0]["content"] == "Hello there"


def test_partial_transcripts_are_ignored(seeded_client: TestClient):
    seeded_client.post(
        WEBHOOK,
        json={
            "message": {
                "type": "transcript",
                "transcriptType": "partial",
                "role": "user",
                "transcript": "Hel",
                "call": {"id": "call_p"},
            }
        },
    )
    # Nothing stored, so the session doesn't exist.
    assert seeded_client.get("/api/v1/rag/sessions/call_p").status_code == 404


def test_lifecycle_events_are_acknowledged(seeded_client: TestClient):
    for event in ("status-update", "end-of-call-report", "speech-update", "conversation-update"):
        response = seeded_client.post(
            WEBHOOK, json={"message": {"type": event, "call": {"id": "call_x"}}}
        )
        assert response.status_code == 200, event
        assert response.json()["received"] is True


def test_unknown_event_type_is_acknowledged(seeded_client: TestClient):
    """A 4xx here would make Vapi retry and could stall a live call."""
    response = seeded_client.post(WEBHOOK, json={"message": {"type": "some-future-event"}})
    assert response.status_code == 200


def test_malformed_body_does_not_500(seeded_client: TestClient):
    response = seeded_client.post(WEBHOOK, json={})
    assert response.status_code == 200


# ----------------------------------------------------------------------- auth
def test_webhook_secret_is_enforced_when_configured(container, knowledge_documents):
    from pydantic import SecretStr

    from tests.conftest import _build_client

    container.settings.vapi.webhook_secret = SecretStr("s3cret")
    container.ingestion.index_sync(knowledge_documents)

    with _build_client(container) as client:
        payload = {"message": {"type": "status-update"}}

        assert client.post(WEBHOOK, json=payload).status_code == 401
        assert client.post(WEBHOOK, json=payload, headers={"x-vapi-secret": "wrong"}).status_code == 401
        assert client.post(WEBHOOK, json=payload, headers={"x-vapi-secret": "s3cret"}).status_code == 200
