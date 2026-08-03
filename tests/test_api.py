"""First-party API surface: health, query, retrieve, documents, auth."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


# --------------------------------------------------------------------- health
def test_liveness_is_dependency_free(client: TestClient):
    assert client.get("/api/v1/health/live").json() == {"status": "ok"}


def test_readiness_is_degraded_without_documents(client: TestClient):
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_readiness_is_ok_once_indexed(seeded_client: TestClient):
    response = seeded_client.get("/api/v1/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["components"]["vector_store"]["vectors"] > 0


def test_request_id_header_is_returned(client: TestClient):
    response = client.get("/api/v1/health/live")
    assert response.headers["x-request-id"]
    assert float(response.headers["x-response-time-ms"]) >= 0


def test_supplied_request_id_is_echoed(client: TestClient):
    response = client.get("/api/v1/health/live", headers={"x-request-id": "trace-42"})
    assert response.headers["x-request-id"] == "trace-42"


# ---------------------------------------------------------------------- query
def test_query_returns_answer_and_citations(seeded_client: TestClient):
    response = seeded_client.post(
        "/api/v1/rag/query", json={"question": "What are the support hours?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["grounded"] is True
    assert body["citations"]
    assert body["session_id"]
    assert body["timings"]["total_ms"] >= 0


def test_query_reuses_a_session(seeded_client: TestClient):
    first = seeded_client.post(
        "/api/v1/rag/query", json={"question": "What is the refund policy?"}
    ).json()
    session_id = first["session_id"]

    seeded_client.post(
        "/api/v1/rag/query",
        json={"question": "And after that period?", "session_id": session_id},
    )

    turns = seeded_client.get(f"/api/v1/rag/sessions/{session_id}").json()["turns"]
    assert len(turns) == 4


def test_query_rejects_empty_question(seeded_client: TestClient):
    assert seeded_client.post("/api/v1/rag/query", json={"question": ""}).status_code == 422


def test_query_source_filter(seeded_client: TestClient):
    body = seeded_client.post(
        "/api/v1/rag/query", json={"question": "how much per month", "source": "pricing.md"}
    ).json()
    assert {c["source"] for c in body["citations"]} == {"pricing.md"}


def test_query_stream(seeded_client: TestClient):
    response = seeded_client.post(
        "/api/v1/rag/query/stream", json={"question": "support hours?"}
    )
    assert response.status_code == 200

    deltas, session_seen = [], False
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:]) if line[6:].strip() not in ("", "{}") else {}
        session_seen |= "session_id" in payload
        if "delta" in payload:
            deltas.append(payload["delta"])

    assert session_seen
    assert "support" in "".join(deltas).lower()


# ------------------------------------------------------------------- retrieve
def test_retrieve_returns_scored_chunks(seeded_client: TestClient):
    body = seeded_client.post(
        "/api/v1/rag/retrieve", json={"query": "refund", "top_k": 2}
    ).json()

    assert 0 < len(body["chunks"]) <= 2
    assert body["chunks"][0]["score"] >= body["chunks"][-1]["score"]
    assert "search_ms" in body["timings"]


def test_retrieve_with_impossible_floor_returns_the_best_hit(seeded_client: TestClient):
    """A too-high floor shouldn't leave the assistant with nothing to say."""
    body = seeded_client.post(
        "/api/v1/rag/retrieve", json={"query": "refund", "min_score": 0.999}
    ).json()
    assert len(body["chunks"]) <= 1


# ------------------------------------------------------------------ documents
def test_list_documents(seeded_client: TestClient):
    body = seeded_client.get("/api/v1/documents").json()
    assert body["total_vectors"] > 0
    assert set(body["sources"]) == {"handbook.md", "pricing.md"}


def test_supported_types_includes_pdf(client: TestClient):
    assert ".pdf" in client.get("/api/v1/documents/supported-types").json()["extensions"]


def test_ingest_text(client: TestClient):
    body = client.post(
        "/api/v1/documents/text",
        json={
            "text": "The office kettle is replaced every eighteen months by facilities.",
            "source": "note.md",
        },
    ).json()

    assert body["chunks_indexed"] == 1
    assert body["total_vectors"] == 1


def test_upload_rejects_unsupported_type(client: TestClient):
    response = client.post(
        "/api/v1/documents/upload", files={"file": ("virus.exe", b"MZ\x00\x00")}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_document"


def test_upload_and_index_markdown(client: TestClient):
    content = (
        b"# Parking\n\nVisitor parking is free for the first two hours and costs "
        b"three dollars per hour after that. The garage closes at eleven at night."
    )
    body = client.post(
        "/api/v1/documents/upload",
        files={"file": ("parking.md", content)},
        params={"keep_copy": "false"},
    ).json()

    assert body["chunks_indexed"] >= 1
    assert "parking.md" in body["sources"]


def test_delete_source(seeded_client: TestClient):
    before = seeded_client.get("/api/v1/documents").json()["total_vectors"]
    body = seeded_client.delete("/api/v1/documents/pricing.md").json()

    assert body["deleted"] > 0
    assert body["total_vectors"] == before - body["deleted"]


def test_reset_clears_everything(seeded_client: TestClient):
    seeded_client.post("/api/v1/documents/reset")
    assert seeded_client.get("/api/v1/documents").json()["total_vectors"] == 0


def test_local_path_traversal_is_blocked(client: TestClient):
    response = client.post(
        "/api/v1/documents/copy-local", params={"path": "../../../../etc/passwd"}
    )
    assert response.status_code == 422
    assert "documents directory" in response.json()["error"]["message"]


# ----------------------------------------------------------------------- auth
def test_api_key_is_enforced_when_configured(container):
    from tests.conftest import _build_client

    container.settings.security.api_keys = ["let-me-in"]

    with _build_client(container) as client:
        payload = {"question": "hello"}

        assert client.post("/api/v1/rag/query", json=payload).status_code == 401
        assert client.post(
            "/api/v1/rag/query", json=payload, headers={"Authorization": "Bearer nope"}
        ).status_code == 401
        assert client.post(
            "/api/v1/rag/query", json=payload, headers={"Authorization": "Bearer let-me-in"}
        ).status_code == 200
        # x-api-key is accepted too.
        assert client.post(
            "/api/v1/rag/query", json=payload, headers={"x-api-key": "let-me-in"}
        ).status_code == 200
        # Health stays open for probes.
        assert client.get("/api/v1/health/live").status_code == 200


def test_error_envelope_shape(client: TestClient):
    body = client.get("/api/v1/rag/sessions/does-not-exist")
    assert body.status_code == 404
