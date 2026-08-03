"""Streamlit console for the RAG voice assistant.

Four tabs: text chat (streaming), live voice via the Vapi Web SDK, knowledge-base
management, and a retrieval inspector for tuning chunking and relevance.

    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Any

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
TIMEOUT = httpx.Timeout(120.0, connect=10.0)

st.set_page_config(page_title="RAG Voice Assistant", page_icon="🎙️", layout="wide")


# ----------------------------------------------------------------- http helpers
def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def api_get(path: str) -> dict[str, Any]:
    with httpx.Client(base_url=API_BASE, timeout=TIMEOUT) as client:
        response = client.get(path, headers=_headers())
        response.raise_for_status()
        return response.json()


def api_post(path: str, *, json_body: Any = None, files: Any = None, params: Any = None) -> dict[str, Any]:
    with httpx.Client(base_url=API_BASE, timeout=TIMEOUT) as client:
        response = client.post(path, json=json_body, files=files, params=params, headers=_headers())
        response.raise_for_status()
        return response.json()


def api_delete(path: str) -> dict[str, Any]:
    with httpx.Client(base_url=API_BASE, timeout=TIMEOUT) as client:
        response = client.delete(path, headers=_headers())
        response.raise_for_status()
        return response.json()


def stream_answer(body: dict[str, Any]):
    """Yield deltas from the SSE endpoint, surfacing the session id first."""
    with (
        httpx.Client(base_url=API_BASE, timeout=TIMEOUT) as client,
        client.stream("POST", "/api/v1/rag/query/stream", json=body, headers=_headers()) as response,
    ):
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if not payload or payload == "{}":
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if session_id := data.get("session_id"):
                st.session_state.session_id = session_id
            if delta := data.get("delta"):
                yield delta


# ------------------------------------------------------------------- app state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("session_id", None)
st.session_state.setdefault("last_citations", [])


# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎙️ RAG Voice Assistant")
    st.caption(f"backend: `{API_BASE}`")

    try:
        health = api_get("/api/v1/health/ready")
        components = health.get("components", {})
        store = components.get("vector_store", {})

        if health.get("status") == "ok":
            st.success(f"ready · {store.get('vectors', 0)} vectors")
        else:
            st.warning("degraded — is the knowledge base empty?")

        st.metric("Indexed chunks", store.get("vectors", 0))
        col_a, col_b = st.columns(2)
        col_a.caption(f"**LLM**\n\n{components.get('llm', {}).get('model', '?')}")
        col_b.caption(f"**Store**\n\n{store.get('provider', '?')}")
        st.caption(f"**Embeddings** · {components.get('embeddings', {}).get('model', '?')}")

        cache = components.get("retrieval_cache", {})
        st.caption(f"cache hit rate: {cache.get('hit_rate', 0):.0%} ({cache.get('hits', 0)} hits)")
        vapi_state = components.get("vapi", {})
    except httpx.HTTPStatusError as exc:
        st.error(f"backend returned {exc.response.status_code}")
        vapi_state = {}
    except Exception as exc:
        st.error(f"cannot reach backend: {exc}")
        st.info("Start it with `python -m app.main` (or `make run`).")
        vapi_state = {}

    st.divider()
    top_k = st.slider("Chunks to retrieve", 1, 12, 4)
    mode = st.radio("Answer style", ["text", "voice"], horizontal=True,
                    help="voice mode strips markdown and keeps replies to 1-3 spoken sentences")
    use_cache = st.checkbox("Use retrieval cache", value=True)

    if st.button("Clear conversation", width="stretch"):
        if st.session_state.session_id:
            # Best-effort: clearing the local view matters more than the server copy.
            with contextlib.suppress(Exception):
                api_delete(f"/api/v1/rag/sessions/{st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.last_citations = []
        st.rerun()


chat_tab, voice_tab, kb_tab, inspect_tab = st.tabs(
    ["💬 Chat", "📞 Live voice", "📚 Knowledge base", "🔍 Retrieval inspector"]
)


# -------------------------------------------------------------------- chat tab
with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander(f"{len(message['citations'])} source(s)"):
                    for citation in message["citations"]:
                        page = f" · p.{citation['page']}" if citation.get("page") else ""
                        st.caption(f"**{citation['source']}**{page} · relevance {citation['score']:.2f}")
                        st.text(citation.get("snippet", "")[:400])

    if prompt := st.chat_input("Ask something about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            body = {
                "question": prompt,
                "session_id": st.session_state.session_id,
                "mode": mode,
                "top_k": top_k,
                "use_cache": use_cache,
            }
            try:
                answer = st.write_stream(stream_answer(body))
            except Exception as exc:
                answer = f"Request failed: {exc}"
                st.error(answer)

            # The stream carries text only, so fetch citations for the same query.
            citations: list[dict[str, Any]] = []
            try:
                detail = api_post(
                    "/api/v1/rag/retrieve",
                    json_body={"query": prompt, "top_k": top_k, "use_cache": use_cache},
                )
                citations = [
                    {
                        "source": chunk["source"],
                        "score": chunk["score"],
                        "page": chunk.get("metadata", {}).get("page"),
                        "snippet": chunk["text"][:400],
                    }
                    for chunk in detail.get("chunks", [])
                ]
                if citations:
                    with st.expander(f"{len(citations)} source(s)"):
                        for citation in citations:
                            page = f" · p.{citation['page']}" if citation.get("page") else ""
                            st.caption(
                                f"**{citation['source']}**{page} · relevance {citation['score']:.2f}"
                            )
                            st.text(citation["snippet"])
            except Exception:
                pass

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "citations": citations}
        )


# ------------------------------------------------------------------- voice tab
with voice_tab:
    st.subheader("Talk to the assistant")
    public_key = os.getenv("VAPI_PUBLIC_KEY", "")
    assistant_id = os.getenv("VAPI_ASSISTANT_ID", "") or (vapi_state.get("assistant_id") or "")

    if not public_key or not assistant_id:
        st.info(
            "Set `VAPI_PUBLIC_KEY` and `VAPI_ASSISTANT_ID` in the environment of "
            "*this* Streamlit process to enable in-browser calling.\n\n"
            "Create the assistant first:\n"
            "```\npython scripts/setup_vapi_assistant.py --server-url https://<your-tunnel>\n```"
        )
        st.caption(
            "The browser needs your **public** key — it is designed to be exposed. "
            "Never put `VAPI_API_KEY` in client-side code."
        )
    else:
        st.caption(f"assistant `{assistant_id}`")
        # The Vapi Web SDK runs entirely in the browser and talks to Vapi directly;
        # Vapi then calls this backend for every turn.
        st.components.v1.html(
            f"""
            <div style="font-family:system-ui,-apple-system,sans-serif">
              <button id="call-btn" style="
                  padding:14px 28px;font-size:16px;font-weight:600;border:0;
                  border-radius:999px;background:#16a34a;color:#fff;cursor:pointer">
                Start call
              </button>
              <span id="status" style="margin-left:14px;color:#64748b;font-size:14px">idle</span>
              <div id="log" style="
                  margin-top:16px;padding:12px;border-radius:10px;background:#0f172a0d;
                  max-height:260px;overflow-y:auto;font-size:14px;line-height:1.5"></div>
            </div>
            <script type="module">
              const {{ default: Vapi }} = await import("https://esm.sh/@vapi-ai/web@2");
              const vapi = new Vapi("{public_key}");
              const btn = document.getElementById("call-btn");
              const status = document.getElementById("status");
              const log = document.getElementById("log");
              let active = false;

              const setStatus = (t, c) => {{ status.textContent = t; status.style.color = c || "#64748b"; }};
              const addLine = (who, text) => {{
                const row = document.createElement("div");
                row.innerHTML = `<b>${{who}}:</b> ${{text}}`;
                log.appendChild(row);
                log.scrollTop = log.scrollHeight;
              }};

              btn.onclick = async () => {{
                if (active) {{ vapi.stop(); return; }}
                setStatus("connecting...", "#d97706");
                try {{ await vapi.start("{assistant_id}"); }}
                catch (e) {{ setStatus("failed: " + e.message, "#dc2626"); }}
              }};

              vapi.on("call-start", () => {{
                active = true;
                btn.textContent = "End call";
                btn.style.background = "#dc2626";
                setStatus("connected — speak now", "#16a34a");
              }});
              vapi.on("call-end", () => {{
                active = false;
                btn.textContent = "Start call";
                btn.style.background = "#16a34a";
                setStatus("call ended");
              }});
              vapi.on("speech-start", () => setStatus("assistant speaking", "#2563eb"));
              vapi.on("speech-end", () => setStatus("listening", "#16a34a"));
              vapi.on("message", (m) => {{
                if (m.type === "transcript" && m.transcriptType === "final") {{
                  addLine(m.role === "user" ? "You" : "Aria", m.transcript);
                }}
              }});
              vapi.on("error", (e) => setStatus("error: " + (e?.message || e), "#dc2626"));
            </script>
            """,
            height=420,
        )
        st.caption(
            "Turn-by-turn transcripts also arrive at your webhook and are readable "
            "via `GET /api/v1/rag/sessions/{call_id}`."
        )


# -------------------------------------------------------- knowledge base tab
with kb_tab:
    st.subheader("Knowledge base")
    left, right = st.columns([3, 2])

    with left:
        try:
            documents = api_get("/api/v1/documents")
            st.caption(
                f"{documents['total_vectors']} vectors · {documents['dimensions']}-d · "
                f"{documents['provider']}/{documents['collection']}"
            )
            if documents["sources"]:
                for source in documents["sources"]:
                    row_a, row_b = st.columns([5, 1])
                    row_a.write(f"📄 {source}")
                    if row_b.button("Delete", key=f"del-{source}"):
                        api_delete(f"/api/v1/documents/{source}")
                        st.rerun()
            else:
                st.info("No documents indexed yet. Upload one, or run `python scripts/ingest.py`.")
        except Exception as exc:
            st.error(f"could not list documents: {exc}")

    with right:
        uploaded = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "md", "docx", "html", "csv", "json"],
        )
        if uploaded is not None and st.button("Index file", type="primary"):
            with st.spinner(f"indexing {uploaded.name}..."):
                try:
                    result = api_post(
                        "/api/v1/documents/upload",
                        files={"file": (uploaded.name, uploaded.getvalue())},
                    )
                    st.success(f"indexed {result['chunks_indexed']} chunk(s)")
                    if result.get("errors"):
                        st.warning(result["errors"])
                    st.rerun()
                except Exception as exc:
                    st.error(f"upload failed: {exc}")

        st.divider()
        with st.form("paste-text"):
            source_name = st.text_input("Source label", value="pasted-note")
            pasted = st.text_area("Paste text to index", height=140)
            if st.form_submit_button("Index text") and pasted.strip():
                try:
                    result = api_post(
                        "/api/v1/documents/text",
                        json_body={"text": pasted, "source": source_name},
                    )
                    st.success(f"indexed {result['chunks_indexed']} chunk(s)")
                    st.rerun()
                except Exception as exc:
                    st.error(f"failed: {exc}")

        st.divider()
        if st.button("Reindex documents directory"):
            with st.spinner("reindexing..."):
                try:
                    result = api_post("/api/v1/documents/reindex", json_body={"reset": False})
                    st.success(
                        f"{result['files_processed']} file(s), "
                        f"{result['chunks_indexed']} chunk(s) in {result['duration_ms'] / 1000:.1f}s"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"reindex failed: {exc}")


# -------------------------------------------------------------- inspector tab
with inspect_tab:
    st.subheader("Retrieval inspector")
    st.caption(
        "See exactly which chunks a query pulls back, and at what score — no LLM "
        "call, so it's free and instant. Use it to tune `RAG_CHUNK_SIZE`, "
        "`RAG_TOP_K` and `RAG_MIN_RELEVANCE_SCORE`."
    )

    probe = st.text_input("Query", placeholder="e.g. what is the refund policy?")
    col_k, col_score = st.columns(2)
    probe_k = col_k.slider("top_k", 1, 20, 6, key="probe-k")
    probe_floor = col_score.slider("min relevance", 0.0, 1.0, 0.0, 0.05, key="probe-floor")

    if probe:
        try:
            result = api_post(
                "/api/v1/rag/retrieve",
                json_body={
                    "query": probe,
                    "top_k": probe_k,
                    "min_score": probe_floor,
                    "use_cache": False,
                },
            )
            timings = result.get("timings", {})
            st.caption(
                f"{len(result['chunks'])} of {result['candidates_considered']} candidates · "
                f"embed {timings.get('embed_ms', 0):.0f}ms · "
                f"search {timings.get('search_ms', 0):.0f}ms · "
                f"rerank {timings.get('rerank_ms', 0):.0f}ms"
            )
            if not result["chunks"]:
                st.warning("Nothing retrieved. Lower the relevance floor or index more documents.")
            for rank, chunk in enumerate(result["chunks"], start=1):
                page = chunk.get("metadata", {}).get("page")
                label = f"#{rank} · {chunk['source']}"
                if page:
                    label += f" p.{page}"
                with st.expander(f"{label} · score {chunk['score']:.3f}"):
                    st.progress(min(chunk["score"], 1.0))
                    st.text(chunk["text"])
                    st.json(chunk.get("metadata", {}), expanded=False)
        except Exception as exc:
            st.error(f"retrieval failed: {exc}")
