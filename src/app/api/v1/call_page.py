"""Standalone browser-call page.

Why this exists rather than living in the Streamlit UI: Streamlit renders custom
HTML inside a sandboxed ``<iframe>`` that carries no ``allow="microphone"``
Permissions-Policy, so ``getUserMedia`` is refused and the Vapi Web SDK fails
before a call can start. Serving the page as a top-level document from our own
origin gives the browser a normal permission prompt.

Only the *public* key reaches the browser -- that key is designed to be exposed.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.api.deps import ContainerDep

router = APIRouter(tags=["vapi"])

_MISSING = """<!doctype html>
<meta charset="utf-8"><title>Voice call unavailable</title>
<style>body{{font:16px/1.6 system-ui,-apple-system,sans-serif;max-width:44rem;
margin:4rem auto;padding:0 1.5rem;color:#0f172a}}code{{background:#0f172a12;
padding:.15em .4em;border-radius:.3em}}</style>
<h1>Not configured yet</h1>
<p>{reason}</p>
<p>Provision an assistant, then restart the server:</p>
<pre><code>python scripts/setup_vapi_assistant.py --server-url https://your-tunnel</code></pre>
"""

_PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Talk to {name}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font:16px/1.6 system-ui,-apple-system,sans-serif; display:grid;
          place-items:center; min-height:100vh; margin:0; }}
  main {{ width:min(38rem,92vw); text-align:center; }}
  button {{ font:600 17px system-ui; padding:16px 34px; border:0;
            border-radius:999px; background:#16a34a; color:#fff; cursor:pointer; }}
  button:disabled {{ opacity:.55; cursor:progress; }}
  #status {{ margin-top:1rem; min-height:1.6em; color:#64748b; }}
  #log {{ margin-top:1.5rem; padding:1rem; border-radius:.75rem;
          background:#64748b1a; text-align:left; max-height:15rem;
          overflow-y:auto; }}
  #log:empty {{ display:none; }}
  .who {{ font-weight:600; }}
</style>
<main>
  <h1>Talk to {name}</h1>
  <p style="color:#64748b">assistant <code>{assistant_id}</code></p>
  <button id="btn">Start call</button>
  <div id="status">Click to start. Your browser will ask for microphone access.</div>
  <div id="log"></div>
</main>
<script type="module">
  const btn = document.getElementById("btn");
  const statusEl = document.getElementById("status");
  const log = document.getElementById("log");
  let vapi = null, active = false;

  const say = (text, color) => {{
    statusEl.textContent = text;
    statusEl.style.color = color || "#64748b";
  }};

  // Vapi surfaces plain objects and DOM exceptions as well as Errors; blindly
  // concatenating gives "[object Object]" and hides the actual cause.
  const describe = (err) => {{
    if (!err) return "unknown error";
    if (typeof err === "string") return err;
    for (const key of ["errorMsg", "message", "error", "reason", "statusText"]) {{
      const value = err[key];
      if (typeof value === "string" && value) return value;
      if (value && typeof value === "object") {{
        const nested = describe(value);
        if (nested && nested !== "unknown error") return nested;
      }}
    }}
    try {{
      const json = JSON.stringify(err, Object.getOwnPropertyNames(err));
      if (json && json !== "{{}}") return json;
    }} catch {{}}
    return String(err);
  }};

  const fail = (err) => {{
    console.error("vapi error:", err);
    let hint = describe(err);
    if (/permission|notallowed|denied/i.test(hint)) {{
      hint += " — allow microphone access for this page, then reload.";
    }}
    say("error: " + hint, "#dc2626");
    btn.disabled = false;
    btn.textContent = "Start call";
    btn.style.background = "#16a34a";
    active = false;
  }};

  btn.onclick = async () => {{
    if (active) {{ vapi?.stop(); return; }}
    btn.disabled = true;
    say("requesting microphone…", "#d97706");
    try {{
      // Ask for the mic before the SDK does, so a denial is reported clearly
      // instead of surfacing as an opaque SDK failure.
      const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
      stream.getTracks().forEach((t) => t.stop());
    }} catch (err) {{
      return fail(err);
    }}

    say("connecting…", "#d97706");
    try {{
      if (!vapi) {{
        const {{ default: Vapi }} = await import("https://esm.sh/@vapi-ai/web@2");
        vapi = new Vapi("{public_key}");
        vapi.on("call-start", () => {{
          active = true;
          btn.disabled = false;
          btn.textContent = "End call";
          btn.style.background = "#dc2626";
          say("connected — speak now", "#16a34a");
        }});
        vapi.on("call-end", () => {{
          active = false;
          btn.disabled = false;
          btn.textContent = "Start call";
          btn.style.background = "#16a34a";
          say("call ended");
        }});
        vapi.on("speech-start", () => say("{name} is speaking", "#2563eb"));
        vapi.on("speech-end", () => say("listening", "#16a34a"));
        vapi.on("message", (m) => {{
          if (m.type === "transcript" && m.transcriptType === "final") {{
            const row = document.createElement("div");
            row.innerHTML =
              '<span class="who">' + (m.role === "user" ? "You" : "{name}") + ":</span> ";
            row.appendChild(document.createTextNode(m.transcript));
            log.appendChild(row);
            log.scrollTop = log.scrollHeight;
          }}
        }});
        vapi.on("error", fail);
      }}
      await vapi.start("{assistant_id}");
    }} catch (err) {{
      fail(err);
    }}
  }};
</script>
"""


@router.get("/call", response_class=HTMLResponse, summary="Browser call page (needs a mic)")
async def call_page(container: ContainerDep) -> HTMLResponse:
    cfg = container.settings.vapi
    if not cfg.public_key:
        return HTMLResponse(
            _MISSING.format(reason="<code>VAPI_PUBLIC_KEY</code> is not set."), status_code=503
        )
    if not cfg.assistant_id:
        return HTMLResponse(
            _MISSING.format(reason="<code>VAPI_ASSISTANT_ID</code> is not set."), status_code=503
        )
    return HTMLResponse(
        _PAGE.format(
            public_key=cfg.public_key,
            assistant_id=cfg.assistant_id,
            name="Aria",
        )
    )
