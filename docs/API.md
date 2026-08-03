# API reference

Base URL: `http://localhost:8000`. Interactive docs at `/docs` (disabled when
`APP_ENV=production`).

---

## Conventions

**Auth.** When `API_KEYS` is set, `/api/v1/rag/*` and `/api/v1/documents/*` require
a bearer token:

```
Authorization: Bearer <key>
# or
x-api-key: <key>
```

`/api/v1/vapi/*` uses the Vapi shared secret instead (`x-vapi-secret`), and
`/api/v1/health/*` is always open so probes work.

**Errors.** Domain failures return a typed envelope:

```json
{
  "error": {
    "code": "unsupported_document",
    "message": "unsupported file type '.exe'",
    "request_id": "a3f9c2e1b8d40567",
    "details": { "supported": [".csv", ".docx", ".html", "..."] }
  }
}
```

| Code | Status |
|---|---|
| `unauthorized` | 401 |
| `ingestion_error`, `unsupported_document` | 422 |
| `rate_limited` | 429 |
| `configuration_error`, `internal_error` | 500 |
| `provider_error` | 502 |
| `retrieval_error` | 503 |

**Response headers.** Every response carries `x-request-id` (echoed from your
`x-request-id` if supplied) and `x-response-time-ms`.

---

## Vapi — custom LLM

### `POST /api/v1/vapi/chat/completions`

OpenAI Chat Completions, backed by RAG. This is what a Vapi `custom-llm` assistant
calls. Retrieval runs on every request.

**Request**

```json
{
  "model": "llama-3.3-70b-versatile",
  "stream": true,
  "temperature": 0.3,
  "max_tokens": 400,
  "messages": [
    { "role": "system", "content": "optional persona, appended to grounding rules" },
    { "role": "user", "content": "What is the refund policy?" }
  ],
  "call": { "id": "call_abc123" }
}
```

| Field | Notes |
|---|---|
| `messages` | Full transcript. Prior turns are trimmed to `RAG_HISTORY_TURNS`. `content` may be a string or OpenAI's multi-part array |
| `stream` | `true` → SSE (what Vapi uses); `false` → single JSON body |
| `call.id` | Optional. When present, the turn is recorded in that session |
| `system` messages | **Appended** to the grounding prompt, never substituted |

**Streaming response** — `text/event-stream`

```
data: {"id":"chatcmpl-b31be0","object":"chat.completion.chunk","created":1785783225,"model":"llama-3.3-70b-versatile","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-b31be0","object":"chat.completion.chunk","created":1785783225,"model":"llama-3.3-70b-versatile","choices":[{"index":0,"delta":{"content":"New customers can get a full refund within thirty days. "}}]}

data: {"id":"chatcmpl-b31be0","object":"chat.completion.chunk","created":1785783225,"model":"llama-3.3-70b-versatile","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

The id is stable across the stream. The opening frame carries the role so Vapi
knows the stream is alive before the first real token. Content is released at
sentence boundaries after speech normalisation.

**Non-streaming response**

```json
{
  "id": "chatcmpl-2e0512f9",
  "object": "chat.completion",
  "created": 1785783225,
  "model": "llama-3.3-70b-versatile",
  "choices": [{
    "index": 0,
    "message": { "role": "assistant", "content": "New customers can get a full refund within thirty days." },
    "finish_reason": "stop"
  }],
  "usage": { "prompt_tokens": 6, "completion_tokens": 24, "total_tokens": 30 }
}
```

**Always 200.** Provider outages produce a spoken apology in `content`, because a
5xx mid-call means silence for the caller.

### `GET /api/v1/vapi/models`

Some OpenAI clients probe this before connecting.

---

## Vapi — webhook

### `POST /api/v1/vapi/webhook`

Requires `x-vapi-secret` when `VAPI_WEBHOOK_SECRET` is set.

**Tool call**

```json
{
  "message": {
    "type": "tool-calls",
    "call": { "id": "call_abc123" },
    "toolCalls": [{
      "id": "tc_1",
      "type": "function",
      "function": {
        "name": "search_knowledge_base",
        "arguments": { "query": "What is the refund policy?" }
      }
    }]
  }
}
```

```json
{
  "results": [
    { "toolCallId": "tc_1", "result": "New customers can get a full refund within thirty days of their first payment." }
  ]
}
```

Accepted aliases for the tool name: `search_knowledge_base`,
`searchKnowledgeBase`, `knowledge_search`, `rag_search`. `arguments` may be an
object or a JSON string; `toolCalls` and `toolCallList` are both read. Multiple
tool calls in one message are all answered, in order.

**Other events**

```json
{ "message": { "type": "transcript", "transcriptType": "final", "role": "user",
               "transcript": "What are your hours?", "call": { "id": "call_abc" } } }
```

→ `{"received": true, "type": "transcript"}`

Handled: `transcript` (finals only), `status-update`, `end-of-call-report`,
`assistant-request`, `conversation-update`, `speech-update`, `hang`. Unrecognised
types also return 200 — a 4xx would trigger a Vapi retry and could stall a call.

---

## Vapi — provisioning

### `POST /api/v1/vapi/assistant`

Create or update the assistant from this repo's configuration.

```json
{
  "name": "RAG Voice Assistant",
  "server_url": "https://abc123.ngrok-free.app",
  "integration": "custom_llm",
  "first_message": "Hi, I'm Aria. What can I help you with?",
  "assistant_id": null
}
```

```json
{
  "assistant_id": "8f2c1b9e-4d3a-...",
  "name": "RAG Voice Assistant",
  "action": "created",
  "webhook_url": "https://abc123.ngrok-free.app/api/v1/vapi/webhook",
  "custom_llm_url": "https://abc123.ngrok-free.app/api/v1/vapi/chat/completions"
}
```

`integration` is `custom_llm` or `tool`. Omitting `assistant_id` falls back to
`VAPI_ASSISTANT_ID`; if that's also unset, a new assistant is created.

### `GET /api/v1/vapi/assistant`

```json
{ "count": 1, "assistants": [{ "id": "8f2c...", "name": "RAG Voice Assistant", "model": "custom-llm" }] }
```

### `POST /api/v1/vapi/web-call`

Credentials for a browser call. Returns the **public** key only.

```json
{ "public_key": "pk_...", "assistant_id": "8f2c1b9e-..." }
```

### `GET /api/v1/vapi/calls/{call_id}`

Proxies Vapi's call record.

---

## RAG

### `POST /api/v1/rag/query`

```json
{
  "question": "What is the refund policy?",
  "session_id": null,
  "mode": "text",
  "top_k": 4,
  "source": null,
  "use_cache": true
}
```

| Field | Default | Notes |
|---|---|---|
| `question` | required | 1–4000 chars |
| `session_id` | new id | Reuse to keep conversational context |
| `mode` | `"text"` | `"voice"` strips markdown and shortens to spoken length |
| `top_k` | `RAG_TOP_K` | 1–20 |
| `source` | — | Restrict retrieval to one source file |
| `use_cache` | `true` | |

```json
{
  "answer": "New customers can request a full refund within 30 days of their first payment. After 30 days, refunds are prorated for the unused portion of the billing period. [sample_knowledge_base.md]",
  "citations": [
    { "source": "sample_knowledge_base.md", "score": 0.889, "page": null,
      "snippet": "## Refunds and cancellation\n\nNew customers can request a full refund within 30 days..." }
  ],
  "grounded": true,
  "retrieved_count": 3,
  "top_score": 0.889,
  "cache_hit": false,
  "model": "llama-3.3-70b-versatile",
  "session_id": "d41f8a7c22b09e15",
  "timings": { "retrieval_ms": 10.43, "generation_ms": 412.7,
               "retrieval_embed_ms": 7.13, "retrieval_search_ms": 2.71,
               "retrieval_rerank_ms": 0.51, "total_ms": 423.2 }
}
```

`grounded: false` means no chunks were retrieved — either small talk or nothing
matched. Check `retrieved_count` and `top_score` to tell which.

### `POST /api/v1/rag/query/stream`

Same body. Returns SSE:

```
event: meta
data: {"session_id":"d41f8a7c22b09e15"}

data: {"delta":"New customers can request a full refund within thirty days. "}

event: done
data: {}
```

Text only — call `/rag/retrieve` for citations.

### `POST /api/v1/rag/retrieve`

Retrieval without generation. No LLM cost, ~10 ms. The tool for tuning chunking
and relevance.

```json
{ "query": "how long do I have to get a refund", "top_k": 3, "min_score": 0.0, "use_cache": false }
```

```json
{
  "query": "how long do I have to get a refund",
  "chunks": [
    { "text": "## Refunds and cancellation\n\nNew customers can request a full refund within 30 days...",
      "score": 0.889, "source": "sample_knowledge_base.md",
      "metadata": { "source": "sample_knowledge_base.md", "file_type": "md",
                    "chunk_index": 0, "chunk_chars": 486, "id": "9f3c..." } }
  ],
  "candidates_considered": 6,
  "cache_hit": false,
  "timings": { "embed_ms": 7.13, "search_ms": 2.71, "rerank_ms": 0.51, "total_ms": 10.43 }
}
```

`min_score: 0` disables the floor — use it to see what retrieval *could* return.
`candidates_considered` is the pre-filter count (`RAG_FETCH_K`).

### Sessions

```
GET    /api/v1/rag/sessions/{session_id}    → { session_id, turns: [{role, content, at}] }
DELETE /api/v1/rag/sessions/{session_id}    → { deleted: true }
POST   /api/v1/rag/cache/clear              → { status: "cleared" }
```

`GET` returns 404 for an unknown or expired session. Sessions expire after 1 hour
of inactivity.

---

## Documents

### `GET /api/v1/documents`

```json
{ "provider": "chroma", "collection": "voice_kb", "total_vectors": 6,
  "dimensions": 384, "sources": ["sample_knowledge_base.md"] }
```

### `GET /api/v1/documents/supported-types`

```json
{ "extensions": [".csv", ".docx", ".htm", ".html", ".json", ".jsonl", ".log",
                 ".markdown", ".md", ".pdf", ".rst", ".text", ".tsv", ".txt"] }
```

### `POST /api/v1/documents/upload`

`multipart/form-data`, field name `file`. Query param `keep_copy` (default `true`)
also saves the file to `RAG_DOCUMENTS_DIR` so a later reindex can rebuild from it.

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F 'file=@handbook.pdf' -F 'keep_copy=true'
```

```json
{ "files_processed": 1, "files_failed": 0, "chunks_created": 42,
  "chunks_indexed": 42, "duplicates_skipped": 0,
  "sources": ["handbook.pdf"], "errors": [], "duration_ms": 1834.2, "total_vectors": 48 }
```

32 MB cap, streamed to disk. Unsupported extensions return 422.

### `POST /api/v1/documents/text`

```json
{ "text": "Visitor parking is free for the first two hours.",
  "source": "parking-note", "metadata": { "team": "facilities" } }
```

`source` is what appears in citations. Custom `metadata` is filterable via the
`source` field on retrieval.

### `POST /api/v1/documents/reindex`

```json
{ "directory": null, "recursive": true, "reset": false }
```

`reset: true` drops all vectors first — use it after changing the embedding model
or chunk size. Because chunk ids are content hashes, `reset: false` is safe and
idempotent for ordinary re-runs.

### `DELETE /api/v1/documents/{source}`

```json
{ "deleted": 6, "source": "handbook.pdf", "total_vectors": 42 }
```

### `POST /api/v1/documents/reset`

Deletes everything. `{ "deleted": 48, "total_vectors": 0 }`

### `POST /api/v1/documents/copy-local?path=<relative-path>`

Ingests a file already on the server. The resolved path must sit inside
`RAG_DOCUMENTS_DIR`; `../` traversal returns 422.

---

## Health

### `GET /api/v1/health/live`

`{"status": "ok"}` — touches no dependencies. Use for liveness probes.

### `GET /api/v1/health/ready`

200 when ready, **503 with `"status": "degraded"`** when the knowledge base is
empty or a component failed.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "development",
  "knowledge_base_ready": true,
  "components": {
    "llm": { "provider": "groq", "model": "llama-3.3-70b-versatile" },
    "embeddings": { "provider": "fastembed", "model": "BAAI/bge-small-en-v1.5" },
    "vector_store": { "provider": "chroma", "collection": "voice_kb",
                      "vectors": 6, "dimensions": 384,
                      "sources": ["sample_knowledge_base.md"] },
    "retrieval_cache": { "hits": 12, "misses": 4, "evictions": 0, "hit_rate": 0.75 },
    "vapi": { "configured": true, "assistant_id": "8f2c...", "webhook_secret_set": true },
    "sessions": { "sessions": 3, "turns": 18 }
  }
}
```

An empty knowledge base counts as degraded on purpose: the service would answer
every question with "I don't know" while every other signal looked healthy.

---

## cURL cookbook

```bash
B=http://localhost:8000/api/v1

# is it ready?
curl -s $B/health/ready | python -m json.tool

# index everything in data/documents
curl -s -X POST $B/documents/reindex -H 'content-type: application/json' -d '{}'

# ask a question
curl -s -X POST $B/rag/query -H 'content-type: application/json' \
  -d '{"question":"What are your support hours?"}' | python -m json.tool

# stream an answer
curl -N -X POST $B/rag/query/stream -H 'content-type: application/json' \
  -d '{"question":"What are your support hours?"}'

# what does retrieval actually return? (free, no LLM)
curl -s -X POST $B/rag/retrieve -H 'content-type: application/json' \
  -d '{"query":"support hours","min_score":0,"top_k":6}' | python -m json.tool

# simulate a Vapi custom-LLM turn
curl -N -X POST $B/vapi/chat/completions -H 'content-type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":"support hours?"}]}'

# simulate a Vapi tool call
curl -s -X POST $B/vapi/webhook -H 'content-type: application/json' \
  -d '{"message":{"type":"tool-calls","toolCalls":[{"id":"tc1","function":{"name":"search_knowledge_base","arguments":{"query":"support hours"}}}]}}'
```
