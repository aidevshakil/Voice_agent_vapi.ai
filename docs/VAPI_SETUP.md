# Vapi setup guide

From zero to a working voice call, plus both integration modes explained and the
things that commonly go wrong.

---

## Prerequisites

1. A Vapi account — <https://dashboard.vapi.ai>
2. This backend running locally (`python -m app.main`) with documents indexed
   (`python scripts/ingest.py`)
3. A tunnel, because **Vapi calls your server** and cannot reach `localhost`

---

## Step 1 — expose your backend

```bash
ngrok http 8000
```

```
Forwarding  https://abc123.ngrok-free.app -> http://localhost:8000
```

Alternatives: `cloudflared tunnel --url http://localhost:8000`, `localtunnel`, or
just deploy somewhere with a real domain.

Verify Vapi will be able to reach you:

```bash
curl https://abc123.ngrok-free.app/api/v1/health/ready
```

You want `"status": "ok"`. If it says `degraded`, your knowledge base is empty —
run `python scripts/ingest.py`.

> The free ngrok URL changes every restart. Each time it does, re-run the
> assistant setup script with the new URL (Step 3).

---

## Step 2 — keys

Dashboard → **Settings → API Keys**. There are two, and the distinction matters:

| Key | Use | Exposure |
|---|---|---|
| **Private key** | Server-side REST calls (create assistants, place calls) | Never send to a browser |
| **Public key** | Browser Web SDK | Designed to be public |

```ini
# .env
VAPI_API_KEY=<private key>
VAPI_PUBLIC_KEY=<public key>
VAPI_WEBHOOK_SECRET=<invent a long random string>
VAPI_SERVER_URL=https://abc123.ngrok-free.app
```

Generate a secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

`VAPI_WEBHOOK_SECRET` is stored on the assistant as `server.secret`; Vapi sends it
back on every request as the `x-vapi-secret` header, and this backend compares it
in constant time. **Without it, anyone who discovers your webhook URL can drive
your LLM and read your knowledge base.** It is enforced when `APP_ENV=production`.

`server.secret` covers the *webhook* URL only. The `custom-llm` URL is a separate
destination that Vapi calls with no credentials of its own, so
`build_assistant_payload` also puts the same secret in `model.headers` — otherwise
every turn 401s and the call dies with
`pipeline-error-custom-llm-401-unauthorized`. Tool-mode does the same on the
tool's `server` block, which overrides the assistant's rather than inheriting it.
Change the secret in `.env` and you must re-run `setup_vapi_assistant.py`, since
the assistant holds its own copy.

---

## Step 3 — create the assistant

```bash
python scripts/setup_vapi_assistant.py --server-url https://abc123.ngrok-free.app
```

```
======================================================
assistant created: 8f2c1b9e-4d3a-...
======================================================
mode        : custom-llm
webhook     : https://abc123.ngrok-free.app/api/v1/vapi/webhook
custom llm  : https://abc123.ngrok-free.app/api/v1/vapi/chat/completions
voice       : 11labs/burt
transcriber : deepgram/nova-3

Add this to .env:
  VAPI_ASSISTANT_ID=8f2c1b9e-4d3a-...
```

Add that id to `.env`. From then on the script **updates** that assistant rather
than creating duplicates.

### Script options

| Flag | Effect |
|---|---|
| `--mode custom-llm` | Default. Every turn routed through this backend |
| `--mode tool` | Hosted Vapi model with a `search_knowledge_base` function |
| `--print-only` | Print the payload without calling Vapi (secret redacted) |
| `--list` | List existing assistants |
| `--assistant-id <id>` | Update a specific assistant |
| `--first-message "..."` | Override the greeting |
| `--name "..."` | Assistant name (set at creation; immutable on update) |

The configuration itself lives in
[`build_assistant_payload`](../src/app/services/vapi_client.py) — version it here
rather than editing the dashboard, so a rebuild is reproducible.

---

## Step 4 — make a call

### In the browser (Streamlit)

```bash
# Windows PowerShell
$env:VAPI_PUBLIC_KEY="<public key>"; $env:VAPI_ASSISTANT_ID="<id>"
streamlit run ui/streamlit_app.py

# macOS/Linux
VAPI_PUBLIC_KEY=<public key> VAPI_ASSISTANT_ID=<id> streamlit run ui/streamlit_app.py
```

Open the **📞 Live voice** tab and press *Start call*. Allow microphone access.
Live transcripts appear as you talk.

### From the dashboard

Open the assistant and press **Talk to Assistant**.

### Over the phone

1. Dashboard → **Phone Numbers** → buy a number (or import a Twilio one)
2. Attach your assistant to it
3. Call the number

Outbound calls:

```bash
curl -X POST https://api.vapi.ai/call/phone \
  -H "Authorization: Bearer $VAPI_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"assistantId":"<id>","phoneNumberId":"<phone-number-id>","customer":{"number":"+15551234567"}}'
```

---

## The two integration modes

### Custom LLM — recommended

Vapi points its `model` at your endpoint and speaks the OpenAI Chat Completions
protocol to it.

```json
{
  "model": {
    "provider": "custom-llm",
    "url": "https://your-host/api/v1/vapi",
    "model": "llama-3.3-70b-versatile"
  }
}
```

Vapi appends `/chat/completions` itself, so the URL is the **base** path.

```
Vapi ──POST /api/v1/vapi/chat/completions {stream:true, messages:[...]}──► backend
Vapi ◄──── SSE: chat.completion.chunk frames ──────────────────────────── backend
```

- Retrieval runs on **every turn**, unconditionally
- **This backend owns the system prompt** — the model has no route to answering
  from its own weights
- A system prompt set in the Vapi dashboard is *appended* to the grounding rules,
  not substituted, so persona tweaks are still possible
- One LLM call per turn

### Tool calling

Vapi runs its own hosted model and gives it a function.

```json
{
  "model": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "tools": [{
      "type": "function",
      "function": {
        "name": "search_knowledge_base",
        "parameters": {
          "type": "object",
          "properties": { "query": { "type": "string" } },
          "required": ["query"]
        }
      },
      "server": { "url": "https://your-host/api/v1/vapi/webhook" }
    }]
  }
}
```

```
Vapi's LLM ──"tool-calls" webhook──► backend ──{results:[...]}──► Vapi's LLM ──► TTS
```

- The hosted model **decides** whether to retrieve — and sometimes decides not to,
  which is the failure mode this mode carries
- Two LLM calls per grounded turn (one to choose the tool, one to compose)
- Useful when you specifically want the hosted model's own reasoning and
  multi-tool behaviour

Both handlers are implemented and tested. Switch with `--mode`.

---

## Webhook events handled

`POST /api/v1/vapi/webhook`, all wrapped as `{"message": {...}}`:

| `type` | Handling |
|---|---|
| `tool-calls` | Runs the RAG pipeline, returns `{results: [{toolCallId, result}]}` |
| `transcript` | Final transcripts stored in the session; partials ignored |
| `status-update` | Call status and end reason recorded |
| `end-of-call-report` | Final transcript, summary, recording URL recorded |
| `assistant-request`, `conversation-update`, `speech-update`, `hang` | Acknowledged |
| anything unrecognised | Acknowledged with 200 |

Unknown types return **200 deliberately**. A 4xx would make Vapi retry, and on a
`tool-calls` message that stalls a live call.

Read a call's transcript afterwards:

```bash
curl http://localhost:8000/api/v1/rag/sessions/<call-id>
```

---

## Voice and transcriber options

```ini
VAPI_VOICE_PROVIDER=11labs        # 11labs | playht | openai | azure | deepgram | cartesia
VAPI_VOICE_ID=burt
VAPI_TRANSCRIBER_PROVIDER=deepgram
VAPI_TRANSCRIBER_MODEL=nova-3
VAPI_TRANSCRIBER_LANGUAGE=en
VAPI_TRANSCRIBER_KEYTERMS=Acme Corp,FastAPI,Kubernetes
VAPI_TRANSCRIBER_CONFIDENCE_THRESHOLD=0.4
```

Re-run `setup_vapi_assistant.py` after changing these. Browse voices in the
dashboard under **Voice Library**.

### When the transcriber mishears you

Three levers, in the order worth trying:

| Lever | Fixes |
|---|---|
| `VAPI_TRANSCRIBER_KEYTERMS` | Names and jargon coming back as nonsense. Deepgram has never seen your document's proper nouns; listing them raises recall on those words dramatically. Cheap and safe — start here |
| `VAPI_TRANSCRIBER_LANGUAGE` | A consistent accent. `en` is generic English; `en-IN`, `en-GB`, `en-AU`, `en-US` are separately tuned and usually beat it for a matching speaker |
| `VAPI_TRANSCRIBER_CONFIDENCE_THRESHOLD` | Words vanishing entirely rather than coming out wrong. Deepgram drops anything below this; the 0.4 default is unkind to accented speech. Try `0.3`, then `0.25` |

`keyterm` is nova-3 only and accepts phrases. Older models fall back to
`keywords`, which takes single tokens — `build_assistant_payload` splits phrases
automatically so switching models can't produce a rejected payload.

### Turn-taking

Set in `build_assistant_payload`. These decide when your turn is considered
over, and getting them wrong looks exactly like bad transcription — the model
answers half a question because it stopped listening early.

| Setting | Value | Effect |
|---|---|---|
| `startSpeakingPlan.waitSeconds` | `0.7` | Silence before the assistant replies |
| `transcriptionEndpointingPlan.onNoPunctuationSeconds` | `2.0` | Grace on an unfinished sentence. Raise it if you're still being cut off mid-question; lower for snappier replies |
| `smartEndpointingPlan.provider` | `livekit` | Predicts end-of-turn from the words, not just silence — the reason a pause for breath no longer ends your turn |
| `stopSpeakingPlan.numWords` | `3` | Words needed to interrupt the assistant. At `2`, a cough cuts it off |
| `silenceTimeoutSeconds` | `300` | Hang up after this much silence |
| `backgroundSpeechDenoisingPlan` | Krisp on | Helps on speakerphone and in noisy rooms |

`responseDelaySeconds`, `llmRequestDelaySeconds` and `backgroundDenoisingEnabled`
are retired from Vapi's assistant schema — `startSpeakingPlan` and
`backgroundSpeechDenoisingPlan` replace them.

---

## Troubleshooting

**Call connects, assistant never speaks**
Vapi can't reach your server. Check the tunnel is live, that `VAPI_SERVER_URL`
matches it, and that the assistant's `model.url` is current:
```bash
python scripts/setup_vapi_assistant.py --list
curl https://<your-tunnel>/api/v1/health/ready
```

**It mishears you, or answers half a question**
Read the actual transcript before tuning anything — `endedReason` and the full
transcript are on the call record (`GET https://api.vapi.ai/call?limit=10`), and
they tell you which of the two failures you have. Garbled words are a
transcriber problem; a coherent but truncated question (`"Is"`, `"Please
check"`) is an endpointing problem. See *Voice and transcriber options* above.

**Webhook returns 401**
The secret on the assistant doesn't match `VAPI_WEBHOOK_SECRET`. Re-run
`setup_vapi_assistant.py` — it sends the current value.

**Assistant answers from general knowledge instead of your documents**
You're in tool mode and the model skipped the tool. Switch to
`--mode custom-llm`, where retrieval is unconditional.

**Answers sound like markup is being read aloud**
The custom-LLM endpoint always uses voice mode. If you see this, the reply is
probably coming from Vapi's own model (tool mode) rather than from here — that
text isn't ours to normalise. Strengthen the dashboard prompt or switch modes.

**Long pauses before each answer**
Check the split:
```bash
curl -X POST localhost:8000/api/v1/rag/query -H 'content-type: application/json' \
  -d '{"question":"..."}' | python -m json.tool
```
`timings.retrieval_ms` should be ~10 ms. If `generation_ms` is high, try
`LLM_PROVIDER=groq`, lower `LLM_MAX_TOKENS`, or reduce `RAG_TOP_K`.

**Streaming works locally but not behind a reverse proxy**
Nginx buffers SSE by default. The `X-Accel-Buffering: no` header is set for you —
make sure your proxy doesn't strip it, and set `proxy_buffering off;` for the
location.

**Assistant interrupts the caller mid-sentence**
Raise `responseDelaySeconds` and `numWordsToInterruptAssistantSpeech`.

**ngrok URL changed**
Update `VAPI_SERVER_URL` and re-run `setup_vapi_assistant.py --server-url <new>`.
A paid static domain avoids this.

---

## Reference

- Vapi docs — <https://docs.vapi.ai>
- Custom LLM — <https://docs.vapi.ai/customization/custom-llm/using-your-server>
- Server URL & events — <https://docs.vapi.ai/server-url>
- Web SDK — <https://docs.vapi.ai/sdk/web>
