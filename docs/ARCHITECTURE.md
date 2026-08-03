# Architecture

How a spoken question becomes a spoken, source-grounded answer — and why each
layer is built the way it is.

---

## 1. Request flow

### Custom-LLM mode (recommended)

```
caller ──speech──► Vapi (STT)
                     │
                     │  POST /api/v1/vapi/chat/completions
                     │  { stream: true, messages: [...full transcript...], call: {...} }
                     ▼
        ┌────────────────────────────────────────────────────┐
        │ chat_completions.py                                │
        │  · verify x-vapi-secret                            │
        │  · extract latest user message                     │
        │  · trim history to RAG_HISTORY_TURNS               │
        └───────────────────────┬────────────────────────────┘
                                ▼
        ┌────────────────────────────────────────────────────┐
        │ RAGPipeline.stream_answer()                        │
        │                                                    │
        │  small talk? ──yes──► skip retrieval               │
        │       │no                                          │
        │       ▼                                            │
        │  Retriever.retrieve()                              │
        │    cache hit? ──yes──► return chunks (~0 ms)       │
        │       │no                                          │
        │       ├─ embed query                     ~7 ms     │
        │       ├─ vector search (fetch_k=16)      ~3 ms     │
        │       ├─ score floor + MMR → top_k=4     ~1 ms     │
        │       └─ cache the result                          │
        │       ▼                                            │
        │  prompts.build_context_block()                     │
        │    numbered, source-labelled, char-budgeted        │
        │       ▼                                            │
        │  LLMProvider.stream()                              │
        │       ▼                                            │
        │  SentenceBuffer → to_speech_friendly()             │
        └───────────────────────┬────────────────────────────┘
                                ▼
                     SSE: chat.completion.chunk frames
                                │
                     Vapi (TTS) ──audio──► caller
```

### Tool-calling mode

```
caller ──speech──► Vapi (STT) ──► Vapi's hosted LLM
                                        │
                                        │ decides to call search_knowledge_base
                                        ▼
                     POST /api/v1/vapi/webhook  { message: { type: "tool-calls", ... } }
                                        │
                                        ▼
                     RAGPipeline.answer()  ──►  { results: [{ toolCallId, result }] }
                                        │
                     Vapi's LLM composes the reply ──► TTS ──► caller
```

The difference matters: in custom-LLM mode retrieval is unconditional and *we*
own the system prompt, so the model has no path to answering from its own
weights. In tool mode the hosted model decides whether to retrieve, and sometimes
decides not to.

---

## 2. Layering

```
┌──────────────────────────────────────────────────────────┐
│ api/          HTTP concerns only: parsing, auth, SSE     │
├──────────────────────────────────────────────────────────┤
│ rag/          retrieval + generation                     │
│ services/     conversation memory, outbound Vapi, cache  │
├──────────────────────────────────────────────────────────┤
│ core/         config, container, logging, exceptions     │
└──────────────────────────────────────────────────────────┘
```

Two rules make the codebase navigable:

1. **Dependencies point downward.** Nothing in `rag/` or `services/` imports from
   `api/`. The pipeline is a plain object you can drive from a script or a test —
   `scripts/evaluate.py` does exactly that.
2. **`core/config.py` is the only place that reads the environment.** Every knob
   is a typed field with a documented default, validated at start-up.

---

## 3. Design decisions

### Why an OpenAI-compatible endpoint rather than only a tool webhook

Vapi's `custom-llm` provider speaks the OpenAI Chat Completions protocol.
Implementing that protocol means Vapi treats this backend *as* the model — so
retrieval happens on every turn by construction, not by the model's choice. The
cost is that we must get the wire format exactly right, including SSE frame
shape, a stable completion id across the stream, `finish_reason` on the terminal
frame, and tolerating the usage-only trailer chunk (`choices: []`) that some
clients emit. All of that is covered by tests.

### Why streaming is non-negotiable

Vapi's TTS begins synthesising as soon as it has a sentence. Buffering the whole
completion adds the full generation time to time-to-first-audio — typically
400–800 ms of dead air per turn, which on a phone call reads as the assistant
having stopped working.

### Why deltas are buffered to sentence boundaries

The obvious approach — normalise each delta as it arrives — silently fails. A
`**` bold marker can arrive as two separate `*` tokens, and a period can arrive
before the space that terminates its sentence. Per-delta normalisation sees
neither. `SentenceBuffer` accumulates to the next real sentence boundary,
normalises the whole sentence, and releases it.

This costs nothing in perceived latency because TTS can't synthesise a partial
sentence anyway. Two refinements: abbreviations (`Dr.`, `e.g.`) and initials
(`J. Smith`) don't trigger a false split, and a `flush_at` cap bounds the wait for
models that emit long unpunctuated runs.

### Why over-fetch then re-rank

A plain top-4 vector search frequently returns four overlapping windows of the
same paragraph — chunk overlap guarantees adjacent chunks are similar, and a
query matching one matches its neighbours. The model then sees one fact repeated
four times and no supporting context.

Fetching 16 candidates and re-ranking with Maximal Marginal Relevance trades a
little relevance for diversity (`RAG_MMR_LAMBDA=0.6`). Redundancy between
candidates is measured as Jaccard overlap of content words rather than cosine
distance, because not every backend returns chunk vectors from a query. It's
coarse, but the near-duplicate case is exactly what it catches reliably.

### Why chunk ids are content hashes

`blake2b(source + page + text)` gives idempotent ingestion for free. Re-running
over an unchanged corpus rewrites identical ids — no duplicates, no dedup pass,
no "delete then re-add" window where the knowledge base is half-empty. Editing
one paragraph changes only that chunk's id; its neighbours keep theirs.

The trade-off: content-addressed ids mean a *moved* paragraph looks like a delete
plus an insert. For a knowledge base that's fine, and `--reset` handles the rare
case where you want a clean rebuild.

### Why the relevance floor keeps the best hit when everything is filtered

A hard floor with nothing above it leaves the model with zero context, which
produces a flat "I don't know" even when a marginally-scoring chunk held the
answer. Keeping the single best candidate and letting the prompt's grounding
rules judge confidence is strictly better: the model can say "I don't have that"
*and* it can recognise a partial match. The floor's job is noise reduction, not
gatekeeping.

### Why failures return 200 with a spoken apology

Once an SSE response has started there is no way to send an error status. More
fundamentally, a 500 on a live call produces **silence** — the caller hears
nothing and concludes the line dropped. An audible "I'm having trouble reaching
my knowledge system, could you try again?" is a recoverable turn. Errors are
logged at ERROR with the request id; the caller just doesn't pay for them.

### Why conversation memory is in-process

`ConversationStore` is an LRU+TTL dict. In custom-LLM mode Vapi replays the whole
transcript every turn, so the store is not on the critical path there at all — it
exists for the tool-calling path (where only the tool arguments arrive) and for
`/rag/sessions` inspection. That makes a process-local store the right default,
and the four-method interface makes Redis a drop-in when `APP_WORKERS>1`.

### Why fastembed is the default embedder

Retrieval sits directly in the turn latency budget. A hosted embedding API adds a
network round trip (30–100 ms) to every single question; `fastembed` runs ONNX
locally in ~7 ms with no torch dependency. Measured on the bundled sample
knowledge base: embed 7 ms + search 3 ms + rerank 1 ms ≈ **10 ms total**. The
model loads at start-up (`warmup()`), not on the first caller's turn.

### Why there's a numpy vector store

Three reasons, in order of how often they matter: it's what the tests run on (no
Chroma process, no cleanup); it's a working fallback when `chromadb` has no wheel
for the running Python; and for small knowledge bases an exact cosine scan is
genuinely faster than an approximate index. It implements the full `VectorStore`
interface including metadata filters and disk persistence — not a stub.

### Why the vector-store factory falls back in development but not production

A missing wheel during local setup should not be a hard stop — you want to keep
working and fix it later, and the log line says so loudly. In production a silent
downgrade to a process-local store would be a data-loss bug: each worker would
have its own index. So `APP_ENV=production` disables the fallback and the error
propagates.

---

## 4. Latency budget

Measured with the bundled sample knowledge base, `fastembed` + Chroma, warm process:

| Stage | Typical | Notes |
|---|---|---|
| Vapi STT (final transcript) | 100–300 ms | Vapi-side |
| Query embedding | ~7 ms | local ONNX |
| Vector search (fetch_k=16) | ~3 ms | |
| Score filter + MMR | ~1 ms | |
| **Total retrieval** | **~10 ms** | ~0 ms on a cache hit |
| LLM time-to-first-token | 200–500 ms | Groq at the low end |
| Vapi TTS first audio | 100–200 ms | Vapi-side |
| **Perceived first response** | **≈ 0.4–1.0 s** | |

Retrieval is ~1% of the budget, which is the point: the expensive part is the LLM,
so the design optimises for *starting* generation early (streaming) and *keeping
the prompt small* (history trimming, context char budget) rather than for
retrieval micro-optimisation.

Three mechanisms keep it there:

- **Query cache** — normalised (case- and whitespace-insensitive) key, TTL+LRU.
  Callers repeat themselves constantly, and repeats cost ~0 ms.
- **Small-talk short-circuit** — "hi", "thanks", "ok" skip retrieval entirely.
  Saves ~10 ms and, more importantly, avoids feeding irrelevant chunks to the model.
- **History trimming** — Vapi replays the full transcript each turn, so without a
  cap the prompt grows unboundedly over a long call, and with it both latency and cost.

---

## 5. Extension points

Each of these is a small interface with a factory; adding an implementation
requires no changes outside its own directory plus one line in the factory.

| Add a… | Implement | Register in |
|---|---|---|
| LLM provider | `LLMProvider` (`complete`, `stream`) | `rag/llm/factory.py` |
| Embedding provider | `EmbeddingProvider` (`embed_documents`, `embed_query`, `dimensions`) | `rag/embeddings/factory.py` |
| Vector store | `VectorStore` (`upsert`, `search`, `delete`, `count`, `stats`, `reset`) | `rag/vectorstores/factory.py` |
| File format | a `Loader` returning `list[Document]` | `LOADERS` in `rag/loaders.py` |
| Vapi tool | a handler branch | `KNOWLEDGE_TOOL_NAMES` in `api/v1/vapi_webhook.py` |

Anything OpenAI-wire-compatible (vLLM, Together, Fireworks, Ollama, LM Studio)
needs no new code at all — reuse `build_openai_provider` with a `base_url`.

Two contracts to respect when implementing a store:

- **Normalise scores to cosine similarity in `[0, 1]`.** Chroma reports cosine
  *distance* in `[0, 2]`; Pinecone reports similarity in `[-1, 1]`. Both are
  mapped so `RAG_MIN_RELEVANCE_SCORE` means one thing everywhere.
- **Blocking implementations are fine.** The base class wraps every method in
  `asyncio.to_thread`, so CPU-bound work never blocks the event loop.

---

## 6. Security posture

| Surface | Control |
|---|---|
| Vapi webhook + custom-LLM | Shared secret via `x-vapi-secret`, constant-time compare. **Required** when `APP_ENV=production` |
| First-party API | Bearer token or `x-api-key` from `API_KEYS`; constant-time compare; disabled when empty |
| Rate limiting | Per-client fixed window; Vapi and health routes exempt (throttling a live call is worse than the load) |
| Uploads | Extension allow-list, 32 MB streamed cap, filename sanitised to `[A-Za-z0-9._-]` |
| Local-path ingestion | Resolved path must be inside `RAG_DOCUMENTS_DIR` — blocks `../` traversal |
| Secrets in config | `SecretStr`; `--print-only` redacts the webhook secret |
| API docs | `/docs`, `/redoc`, `/openapi.json` disabled when `APP_ENV=production` |
| Error responses | Domain errors return a typed envelope; unhandled exceptions return a generic message and log the traceback server-side |

The rate limiter is in-process and therefore per-worker — a guardrail against a
runaway client, not a billing control. Put a real limiter at the edge for that.
