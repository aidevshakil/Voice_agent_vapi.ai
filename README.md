# Real-Time RAG Voice Assistant (Vapi.ai)

A production-shaped voice assistant that **listens, retrieves, thinks, and speaks**
— Vapi handles speech-to-text and text-to-speech in real time, while this backend
grounds every single answer in your own documents.

```
   caller speaks
        │
        ▼
┌───────────────────┐   streaming text    ┌──────────────────────────────┐
│  Vapi             │ ──────────────────► │  THIS BACKEND (FastAPI)      │
│  STT · TTS · VAD  │                     │                              │
│  turn-taking      │ ◄────────────────── │  1. embed the question       │
└───────────────────┘   SSE token stream  │  2. search the vector store  │
        │                                 │  3. re-rank (MMR) + filter   │
        ▼                                 │  4. build a grounded prompt  │
   caller hears                           │  5. stream the LLM's answer  │
                                          └──────────────────────────────┘
                                                       │
                            ┌──────────────────────────┼──────────────────────────┐
                            ▼                          ▼                          ▼
                     Embeddings                  Vector store                   LLM
                 fastembed · OpenAI          Chroma · Pinecone · memory   Groq · OpenAI · Gemini
```

Retrieval measures **~10 ms** end to end with the default local embeddings, so
essentially all of the response latency is the LLM — which is why the answer is
streamed sentence by sentence rather than buffered.

---

## Table of contents

- [What makes this different](#what-makes-this-different)
- [Quick start](#quick-start)
- [Connecting Vapi](#connecting-vapi)
- [Project layout](#project-layout)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Tuning retrieval quality](#tuning-retrieval-quality)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## What makes this different

**Two Vapi integration modes, and the recommended one can't hallucinate its way
around retrieval.**

| | Custom LLM *(recommended)* | Tool calling |
|---|---|---|
| Vapi calls | `POST /api/v1/vapi/chat/completions` | `POST /api/v1/vapi/webhook` |
| Who owns the prompt | **this backend** | Vapi's hosted model |
| Retrieval runs | **every turn, always** | only when the model decides to call the tool |
| Can answer from model weights | no | yes — this is the failure mode |
| Setup | `--mode custom-llm` | `--mode tool` |

Both are implemented. Use custom-LLM unless you specifically need a hosted
model's own tool-calling behaviour.

**Everything else that a voice pipeline actually needs:**

- **Speech-safe output.** LLMs emit markdown; TTS engines read it literally.
  Output passes through a normaliser that strips markup, expands `$1.2M` to
  "1.2 million dollars", `&` to "and", and drops citation markers and URLs.
- **Sentence-boundary streaming.** Deltas are buffered to sentence boundaries
  before normalising, so a `**` split across two tokens can't leak to the
  speaker — while abbreviations (`Dr.`) and decimals (`3.5`) don't false-split.
- **Never silent on a live call.** A provider outage returns a spoken apology
  with HTTP 200, never a 500. Silence is the worst possible call outcome.
- **Idempotent ingestion.** Chunk ids are content hashes, so re-running ingestion
  rewrites the same vectors instead of duplicating them; only edited paragraphs
  get new ids.
- **MMR re-ranking.** Over-fetch 16 candidates, re-rank to 4 with diversity, so
  the model doesn't see four near-copies of one paragraph.
- **Swap any layer with one env var.** Three LLM providers, three embedders,
  three vector stores, all behind small interfaces.
- **120 tests, no API keys required.** Fakes for every provider.

---

## Quick start

### 1. Install

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.11+ (verified on 3.14).

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` — the only value you *must* set is one LLM key:

```ini
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...        # free tier at console.groq.com, and the fastest option
```

Defaults for everything else need no keys or services: `fastembed` runs locally
(the model downloads on first use, ~130 MB) and Chroma persists to
`./data/vector_store`.

### 3. Add documents and index them

Drop files into `data/documents/` — `.pdf`, `.md`, `.txt`, `.docx`, `.html`,
`.csv`, `.json` are all supported. A sample knowledge base is included so you can
test immediately.

```bash
python scripts/ingest.py
```

```
files processed   : 1
chunks indexed    : 6
total vectors     : 6
```

### 4. Run

```bash
python -m app.main            # or: make run
```

Open <http://localhost:8000/docs>.

### 5. Try it without any voice setup

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H 'content-type: application/json' \
  -d '{"question":"What is the refund policy?"}'
```

```json
{
  "answer": "New customers can get a full refund within thirty days of their first payment. After that, refunds are prorated for the unused part of the billing period.",
  "citations": [{ "source": "sample_knowledge_base.md", "score": 0.889 }],
  "grounded": true,
  "timings": { "retrieval_ms": 10.4, "generation_ms": 412.0, "total_ms": 422.4 }
}
```

### 6. Optional UI

```bash
pip install -r requirements-dev.txt
streamlit run ui/streamlit_app.py
```

Four tabs: streaming text chat with citations, in-browser voice calling, knowledge-base
management, and a retrieval inspector for tuning.

---

## Connecting Vapi

Vapi calls *your* server, so it needs a public HTTPS URL. Locally that means a tunnel.

### 1. Expose the backend

```bash
ngrok http 8000
# → https://abc123.ngrok-free.app
```

### 2. Add your Vapi keys to `.env`

From <https://dashboard.vapi.ai> → Settings → API Keys:

```ini
VAPI_API_KEY=...              # private — server-side only
VAPI_PUBLIC_KEY=...           # safe to expose to browsers
VAPI_WEBHOOK_SECRET=<make-up-a-long-random-string>
VAPI_SERVER_URL=https://abc123.ngrok-free.app
```

`VAPI_WEBHOOK_SECRET` is a shared secret Vapi echoes back as `x-vapi-secret`.
Without it, anyone who learns your webhook URL can drive your LLM and read your
knowledge base. It is **required** when `APP_ENV=production`.

### 3. Create the assistant

```bash
python scripts/setup_vapi_assistant.py --server-url https://abc123.ngrok-free.app
```

```
assistant created: 8f2c1b9e-...
mode        : custom-llm
webhook     : https://abc123.ngrok-free.app/api/v1/vapi/webhook
custom llm  : https://abc123.ngrok-free.app/api/v1/vapi/chat/completions
```

Copy the id into `.env` as `VAPI_ASSISTANT_ID=...`. Re-running the script then
*updates* that assistant instead of creating another.

Useful flags: `--mode tool` for tool-calling instead of custom-LLM,
`--print-only` to inspect the payload without sending it, `--list` to see what
already exists.

### 4. Call it

- **Browser** — the Streamlit "Live voice" tab (needs `VAPI_PUBLIC_KEY` and
  `VAPI_ASSISTANT_ID` in the Streamlit process's environment).
- **Dashboard** — press *Talk to Assistant* on the assistant page.
- **Phone** — buy or import a number in Vapi and attach the assistant.

The assistant config lives in
[`build_assistant_payload`](src/app/services/vapi_client.py) — voice, transcriber,
interruption sensitivity, and timeouts are all versioned in this repo rather than
hand-edited in a dashboard.

---

## Project layout

```
Voice_agent_vapi.ai/
├── src/app/
│   ├── main.py                  # app factory, lifespan, entrypoint
│   ├── core/
│   │   ├── config.py            # typed settings, one place for every knob
│   │   ├── container.py         # dependency graph, built once at startup
│   │   ├── exceptions.py        # domain errors → HTTP envelopes
│   │   └── logging.py           # request-id correlation
│   ├── api/
│   │   ├── deps.py              # container access, API-key + Vapi-secret auth
│   │   ├── middleware.py        # request context, rate limit, CORS, gzip
│   │   └── v1/
│   │       ├── chat_completions.py  # ← Vapi custom-LLM (OpenAI-compatible SSE)
│   │       ├── vapi_webhook.py      # ← Vapi tool calls, transcripts, provisioning
│   │       ├── rag.py               # text chat, streaming chat, retrieval debug
│   │       ├── documents.py         # upload / reindex / delete
│   │       └── health.py            # liveness + readiness
│   ├── rag/
│   │   ├── pipeline.py          # orchestration: answer() and stream_answer()
│   │   ├── retriever.py         # embed → over-fetch → MMR → filter → cache
│   │   ├── ingestion.py         # load → chunk → embed → upsert (idempotent)
│   │   ├── loaders.py           # per-format text extraction
│   │   ├── prompts.py           # voice-tuned, grounding-enforcing prompts
│   │   ├── embeddings/          # fastembed | openai | sentence_transformers
│   │   ├── vectorstores/        # chroma | pinecone | memory
│   │   └── llm/                 # groq | openai | gemini
│   ├── schemas/
│   │   ├── openai_compat.py     # OpenAI chat-completions wire format
│   │   ├── vapi.py              # Vapi webhook payloads
│   │   └── rag.py               # first-party request/response models
│   ├── services/
│   │   ├── conversation.py      # per-call memory, TTL + LRU
│   │   ├── vapi_client.py       # outbound Vapi REST + assistant-as-code
│   │   └── cache.py             # async TTL/LRU cache
│   └── utils/
│       ├── text.py              # speech normalisation, SentenceBuffer
│       └── timing.py            # per-stage latency instrumentation
├── scripts/
│   ├── ingest.py                # build the index
│   ├── setup_vapi_assistant.py  # create/update the Vapi assistant
│   └── evaluate.py              # latency + grounding smoke eval
├── ui/streamlit_app.py
├── tests/                       # 120 tests, all provider-free
├── docs/                        # ARCHITECTURE · VAPI_SETUP · API
├── data/documents/              # ← your knowledge base goes here
├── Makefile
└── .env.example
```

The layering rule: `api/` → `rag/`+`services/` → `core/`. Nothing in `rag/` imports
from `api/`, and nothing reads `os.environ` outside `core/config.py`.

---

## Configuration

Full list with comments in [`.env.example`](.env.example). The ones that matter most:

### LLM

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` · `openai` · `gemini` |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | |
| `LLM_MAX_TOKENS` | `400` | Keep low — long spoken answers feel worse, not better |
| `LLM_TEMPERATURE` | `0.3` | Low, because answers should track the source text |

Groq is the default for one reason: time-to-first-token. On a phone call, a
300 ms head start is the difference between natural and awkward.

### Embeddings

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | `fastembed` | Local ONNX, no API key, no torch, ~7 ms/query |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | 384-d |

Changing the embedding model changes the vector width — **reset the store and
re-ingest** (`python scripts/ingest.py --reset`).

### Vector store

| Variable | Default | Notes |
|---|---|---|
| `VECTOR_STORE_PROVIDER` | `chroma` | `chroma` · `pinecone` · `memory` |
| `VECTOR_STORE_PATH` | `./data/vector_store` | |

`memory` is a numpy exact-cosine store with disk persistence — zero dependencies,
fine up to ~50k chunks, and what the tests use. Use `pinecone` when you need
multiple instances to share one index.

### Retrieval

| Variable | Default | Notes |
|---|---|---|
| `RAG_CHUNK_SIZE` | `800` | Chars. Smaller = sharper retrieval, less context per hit |
| `RAG_CHUNK_OVERLAP` | `120` | Stops answers being split across a boundary |
| `RAG_TOP_K` | `4` | Chunks sent to the model |
| `RAG_FETCH_K` | `16` | Candidates fetched before re-ranking |
| `RAG_MIN_RELEVANCE_SCORE` | `0.75` | **Model-dependent — see below** |
| `RAG_USE_MMR` | `true` | Diversity re-ranking |
| `RAG_HISTORY_TURNS` | `6` | Caps prompt growth over a long call |

### Security

| Variable | Default | Notes |
|---|---|---|
| `VAPI_WEBHOOK_SECRET` | — | Required when `APP_ENV=production` |
| `API_KEYS` | — | Comma-separated bearer tokens; empty disables auth |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per-worker guardrail; Vapi routes are exempt |

---

## API reference

Full detail in [docs/API.md](docs/API.md); interactive docs at `/docs`.

### Vapi-facing

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/vapi/chat/completions` | OpenAI-compatible SSE — the custom-LLM endpoint |
| `POST` | `/api/v1/vapi/webhook` | Tool calls, transcripts, call lifecycle |
| `POST` | `/api/v1/vapi/assistant` | Create/update the assistant from repo config |
| `GET` | `/api/v1/vapi/assistant` | List assistants |
| `POST` | `/api/v1/vapi/web-call` | Credentials for a browser call |

### RAG

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/rag/query` | Grounded answer with citations and timings |
| `POST` | `/api/v1/rag/query/stream` | Same, streamed as SSE |
| `POST` | `/api/v1/rag/retrieve` | **Retrieval only** — no LLM cost, for tuning |
| `GET` | `/api/v1/rag/sessions/{id}` | Read a conversation transcript |
| `DELETE` | `/api/v1/rag/sessions/{id}` | Forget a session |

### Documents

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/documents` | What's indexed |
| `POST` | `/api/v1/documents/upload` | Upload + index a file (32 MB cap) |
| `POST` | `/api/v1/documents/text` | Index a raw snippet |
| `POST` | `/api/v1/documents/reindex` | Re-ingest the documents directory |
| `DELETE` | `/api/v1/documents/{source}` | Delete one source's vectors |
| `POST` | `/api/v1/documents/reset` | Wipe the knowledge base |

### Health

`GET /api/v1/health/live` — dependency-free liveness.
`GET /api/v1/health/ready` — component detail; returns **503 + `degraded`** when
the knowledge base is empty, because an agent with no vectors answers every
question with "I don't know" while looking perfectly healthy.

---

## Tuning retrieval quality

`POST /api/v1/rag/retrieve` (or the Streamlit inspector) shows exactly which
chunks a question pulls and at what score — no LLM call, so it's free and instant.

```bash
curl -X POST http://localhost:8000/api/v1/rag/retrieve \
  -H 'content-type: application/json' \
  -d '{"query":"how long do I have to get a refund","top_k":3}'
```

### The relevance floor is model-dependent

This trips people up. `bge`/`e5` models rarely score *any* English text below
~0.65, so a floor of `0.25` filters nothing:

| Query | Best score | Kept at floor 0.75 |
|---|---|---|
| "how long do I have to get a refund" | 0.889 | 3 chunks ✅ |
| "who won the world cup in 1998" | 0.721 | 1 chunk (fallback) |

Measured values for the bundled sample knowledge base. Guidance:

- `bge-*` / `e5-*` → **0.72–0.80**
- OpenAI `text-embedding-3-*` → **0.30–0.40** (different score distribution)

If every candidate is filtered out, the single best hit is kept anyway — so a
high floor reduces noise rather than silencing the assistant, and the prompt's
grounding rules handle the "I don't have that" case.

### Chunk size

- Dense reference material (policies, specs) → 500–800 chars
- Narrative prose → 1000–1500
- Q&A pairs or FAQs → keep each pair whole; ingest as JSON/CSV, one record per row

### Measuring changes

```bash
python scripts/evaluate.py --retrieval-only     # fast, free
python scripts/evaluate.py                      # includes generation
python scripts/evaluate.py --file my_questions.json
```

```json
[{ "question": "What is the refund policy?",
   "expect_source": "handbook.pdf",
   "expect_terms": ["30 days"] }]
```

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest                                  # 120 tests, ~2s, no API keys needed
pytest --cov=src/app --cov-report=term-missing
ruff check . && mypy src                # lint + types
```

Every provider has a fake: a deterministic hash-based embedder, the numpy vector
store, and a scripted LLM. Tests cover OpenAI wire compatibility, SSE frame
shape, webhook payload variants, speech normalisation, idempotent ingestion,
auth enforcement, and path-traversal rejection.

---

## Deployment

Run it directly with uvicorn:

```bash
python -m uvicorn app.main:app --app-dir src --host 0.0.0.0 --port 8000 --workers 4 --no-access-log
```

Or `python -m app.main`, which reads `APP_HOST`, `APP_PORT` and `APP_WORKERS`
from your environment.

Behind a process manager (systemd, supervisor, pm2), set the working directory to
the project root and `PYTHONPATH=src`.

### Production checklist

```ini
APP_ENV=production          # disables /docs, turns off the vector-store fallback
APP_LOG_JSON=true
APP_RELOAD=false
APP_WORKERS=4
APP_CORS_ORIGINS=https://your-frontend.example
VAPI_WEBHOOK_SECRET=<long random string>    # enforced in production
API_KEYS=<long random string>
```

Two things to know before scaling out:

1. **Conversation memory is per-process.** `ConversationStore` is an in-process
   dict, correct for a single worker. With `APP_WORKERS>1`, swap it for Redis —
   the interface is four methods wide, deliberately. The custom-LLM path is
   unaffected (Vapi replays the transcript each turn); the tool-calling path and
   `/rag/sessions` are.
2. **Chroma's local mode is single-writer.** For multiple instances, use
   `VECTOR_STORE_PROVIDER=pinecone` or run Chroma in server mode.

Point `/api/v1/health/ready` at your load balancer's health check. Keep
`X-Accel-Buffering: no` intact through any reverse proxy or SSE will be buffered
and streaming breaks silently.

---

## Troubleshooting

**"knowledge base is empty" / `/health/ready` returns 503**
Run `python scripts/ingest.py`. Check with `python scripts/ingest.py --stats`.

**Assistant says "I don't have that information" for something clearly in a document**
Check what retrieval actually returns:
```bash
curl -X POST localhost:8000/api/v1/rag/retrieve -H 'content-type: application/json' \
  -d '{"query":"your question","min_score":0}'
```
If the right chunk is there but scoring below your floor, lower
`RAG_MIN_RELEVANCE_SCORE`. If it isn't there at all, the chunk boundary probably
split the answer — reduce `RAG_CHUNK_SIZE` or raise `RAG_CHUNK_OVERLAP` and
reindex with `--reset`.

**Vapi calls connect but the assistant never speaks**
Vapi cannot reach localhost. Confirm the tunnel is live and that
`VAPI_SERVER_URL` matches it, then check the assistant's `model.url` with
`python scripts/setup_vapi_assistant.py --list`.

**Webhook returns 401**
`VAPI_WEBHOOK_SECRET` in `.env` must match the assistant's `server.secret`.
Re-run `setup_vapi_assistant.py` after changing it — the script sends it along.

**Answers sound like they're reading markup aloud**
Confirm the request uses `mode: "voice"`. The custom-LLM endpoint always does;
`/rag/query` defaults to `"text"`.

**A PDF ingests with zero chunks**
It's a scanned image. Extract the text with OCR (e.g. `ocrmypdf`) first.

**`chromadb` won't install**
Set `VECTOR_STORE_PROVIDER=memory`. The numpy store is a full implementation of
the same interface, not a stub.

**Dimension mismatch after changing the embedding model**
`python scripts/ingest.py --reset`.

---

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — request flow, design decisions and their trade-offs
- [docs/VAPI_SETUP.md](docs/VAPI_SETUP.md) — Vapi walkthrough, both integration modes, phone setup
- [docs/API.md](docs/API.md) — every endpoint with request/response examples

## License

MIT
