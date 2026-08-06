# Deployment — Vercel (UI) + Render (API)

The React UI is static, so Vercel serves it well. The API is not a good fit for
Vercel: functions get a read-only filesystem, and Chroma needs to write its
index while `fastembed` needs ~130 MB of model weights on disk. It goes on a
container host instead. Everything here uses free tiers.

```
   Vercel (static)                Render (container)
┌──────────────────┐  HTTPS   ┌────────────────────────┐
│  React UI        │─────────▶│  FastAPI               │
│  VITE_API_BASE ──┘          │   Chroma index (disk)  │
└──────────────────┘          │   fastembed weights    │
                              └───────────▲────────────┘
                                          │  custom-llm + webhook
                                    ┌─────┴─────┐
                                    │   Vapi    │
                                    └───────────┘
```

The big win over the dev setup: Render gives you a **permanent HTTPS URL**, so
ngrok disappears and you stop re-provisioning the assistant on every restart.

---

## Before you start

Commit and push to GitHub — both platforms deploy from the repo.

Check what will be indexed. `render.yaml` builds the index from everything in
`data/documents/`, and the repo ships a `sample_knowledge_base.md` full of
fictional support-desk content. Delete it unless you want the assistant
answering questions about refund policies:

```bash
git rm data/documents/sample_knowledge_base.md
```

`.env` is gitignored and does **not** travel. Every value gets re-entered as an
environment variable on Render.

---

## Step 1 — API on Render

**New → Blueprint → pick this repo.** Render reads [`render.yaml`](../render.yaml)
and pre-fills everything except secrets, which it prompts for:

| Variable | Value |
|---|---|
| `GROQ_API_KEY` | from your `.env` |
| `VAPI_API_KEY` | from your `.env` |
| `VAPI_PUBLIC_KEY` | from your `.env` |
| `VAPI_ASSISTANT_ID` | from your `.env` |
| `VAPI_WEBHOOK_SECRET` | from your `.env` — **required**, `APP_ENV=production` refuses to start without it |
| `VAPI_TRANSCRIBER_KEYTERMS` | from your `.env` |
| `APP_CORS_ORIGINS` | leave blank for now — you don't know the Vercel URL yet |
| `VAPI_SERVER_URL` | leave blank for now |

The first build takes ~5 minutes: it installs dependencies, downloads the
embedding model, and runs `scripts/ingest.py --reset` to build the index.
That last step matters — `data/vector_store/` is gitignored, so without it the
service would start with an empty knowledge base and answer nothing.

When it goes live, note the URL: `https://voice-rag-api.onrender.com`.

Verify before moving on:

```bash
curl https://<your-service>.onrender.com/api/v1/health/ready
```

`knowledge_base_ready` must be `true` and `vector_store.vectors` non-zero. If
vectors is `0`, the build's ingest step failed — check the build log.

## Step 2 — UI on Vercel

**Add New → Project → import the repo**, then:

| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Framework Preset | Vite (auto-detected) |
| Environment Variable | `VITE_API_BASE` = `https://<your-service>.onrender.com` |

`VITE_API_BASE` is read at **build** time, not run time — changing it later
requires a redeploy, not just a settings save. Leaving it unset makes the UI
call its own origin, which has no API and fails.

[`frontend/vercel.json`](../frontend/vercel.json) handles the rest. The
catch-all rewrite exists because React Router owns `/documents`; without it,
loading that URL directly returns a 404 from Vercel's static host.

## Step 3 — let the two talk

Back on Render, set the CORS origin to your new Vercel domain and redeploy:

```
APP_CORS_ORIGINS=https://your-app.vercel.app
```

The browser blocks every API call until this matches your domain **exactly** —
scheme included, no trailing slash. Add preview domains as extra
comma-separated entries if you use them.

## Step 4 — point Vapi at Render

Set `VAPI_SERVER_URL` on Render to its own public URL, then re-provision the
assistant so it stops calling your laptop. Run this locally:

```bash
VAPI_SERVER_URL=https://<your-service>.onrender.com \
  python scripts/setup_vapi_assistant.py \
    --server-url https://<your-service>.onrender.com \
    --assistant-id <your-assistant-id>
```

This rewrites the assistant's webhook URL, its custom-LLM URL, and the
`x-vapi-secret` header on both. The secret you send must match
`VAPI_WEBHOOK_SECRET` on Render, or every turn 401s and calls die with
`pipeline-error-custom-llm-401-unauthorized`.

## Step 5 — verify

```bash
# custom-LLM endpoint reachable and authenticated
curl -N -X POST https://<your-service>.onrender.com/api/v1/vapi/chat/completions \
  -H 'Content-Type: application/json' \
  -H "x-vapi-secret: $VAPI_WEBHOOK_SECRET" \
  -d '{"model":"llama-3.1-8b-instant","stream":true,
       "messages":[{"role":"user","content":"what skills are listed?"}]}'
```

Tokens streaming back means the whole chain works. Then open the Vercel URL and
click **Start Real-Time Voice Call**. Microphone access needs HTTPS, which both
platforms give you.

---

## What the free tier costs you

**The service sleeps after ~15 minutes idle and takes ~50 s to wake.** For voice
this is the thing to know: Vapi will not wait 50 s for a first response, so the
first call after an idle period drops. Text chat just hangs, then works.

Two ways around it:

- Warm it before demoing — load the health endpoint and wait for a reply
- Upgrade to Render Starter (~$7/mo), which never sleeps

An uptime pinger every 10 minutes also works, but Render's terms discourage it
and it burns your monthly free hours (750/month — one always-on service is just
under the cap).

**Uploads don't survive.** The index is rebuilt at each deploy from
`data/documents/`, and the free tier's disk resets when the instance sleeps.
Documents added through the Documents tab disappear. To make one permanent,
commit it to `data/documents/` and redeploy.

---

## Troubleshooting

**UI loads, every API call fails** — open the browser console. A CORS error
means `APP_CORS_ORIGINS` doesn't match your Vercel domain exactly. A request to
`your-app.vercel.app/api/...` instead of the Render URL means `VITE_API_BASE`
was missing at build time; set it and redeploy.

**Assistant answers "I don't have that information"** — the index is empty.
Check `/api/v1/health/ready`; if `vectors` is `0`, the build's ingest step
failed.

**Voice call connects, then ends immediately** — check the call's `endedReason`
via `GET https://api.vapi.ai/call?limit=5`. A `custom-llm-401-unauthorized`
means the assistant's secret and Render's `VAPI_WEBHOOK_SECRET` disagree; re-run
step 4.

**First request after idle times out** — the free tier waking up. See above.
