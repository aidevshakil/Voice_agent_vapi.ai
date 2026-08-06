#!/usr/bin/env bash

set -uo pipefail
cd "$(dirname "$0")"

VOICE=0
[ "${1:-}" = "--voice" ] && VOICE=1

VENV=.venv
PY="$VENV/bin/python"
LOGS=.run-logs
PIDS=()

c_ok()   { printf "\033[32m%s\033[0m\n" "$*"; }
c_warn() { printf "\033[33m%s\033[0m\n" "$*"; }
c_err()  { printf "\033[31m%s\033[0m\n" "$*"; }
step()   { printf "\n\033[1m==> %s\033[0m\n" "$*"; }

cleanup() {
  echo
  step "stopping"
  for pid in "${PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null && echo "  stopped pid $pid"
  done
  exit 0
}
trap cleanup INT TERM

envval() {
  [ -f .env ] || return 0
  sed -n "s/^$1=//p" .env | head -1 | sed 's/[[:space:]]*#.*$//' | tr -d '"'\''' | xargs 2>/dev/null
}

wait_http() {
  local url=$1 tries=${2:-45}
  for _ in $(seq 1 "$tries"); do
    curl -sf -m 2 -o /dev/null "$url" && return 0
    sleep 1
  done
  return 1
}

free_port() {
  local port=$1
  local pids
  pids=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null)
  if [ -n "$pids" ]; then
    c_warn "  port $port was busy (pid $(echo "$pids" | tr '\n' ' ')) - freeing it"
    echo "$pids" | xargs kill 2>/dev/null
    sleep 2
  fi
}

mkdir -p "$LOGS"

step "checking environment"
if [ ! -x "$PY" ]; then
  c_warn "  no $VENV - creating it (one time, ~2 min)"
  python3 -m venv "$VENV" || { c_err "  could not create venv"; exit 1; }
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements-dev.txt || { c_err "  dependency install failed"; exit 1; }
  "$PY" -m pip install -q -e .
fi
"$PY" -c "import fastapi, uvicorn, chromadb, fastembed" 2>/dev/null || {
  c_warn "  dependencies incomplete - installing"
  "$PY" -m pip install -q -r requirements-dev.txt && "$PY" -m pip install -q -e .
}
c_ok "  python ready: $($PY --version)"

if [ -z "$(envval GROQ_API_KEY)" ]; then
  c_err "  GROQ_API_KEY is not set in .env - the app cannot start without it."
  c_err "  Get a free key at https://console.groq.com/keys"
  exit 1
fi
c_ok "  GROQ_API_KEY present"

step "starting API on :8000"
free_port 8000
"$PY" -m uvicorn app.main:app --app-dir src --port 8000 --no-access-log > "$LOGS/app.log" 2>&1 &
PIDS+=($!)
if wait_http http://127.0.0.1:8000/api/v1/health/live 60; then
  c_ok "  API up"
else
  c_err "  API failed to start. Last lines of $LOGS/app.log:"
  tail -20 "$LOGS/app.log"
  cleanup
fi

VECTORS=$(curl -s -m 5 http://127.0.0.1:8000/api/v1/health/ready \
  | "$PY" -c "import json,sys;print(json.load(sys.stdin)['components']['vector_store']['vectors'])" 2>/dev/null || echo 0)
if [ "${VECTORS:-0}" = "0" ]; then
  c_warn "  knowledge base is EMPTY - put files in data/documents and run: make reindex"
else
  c_ok "  knowledge base: $VECTORS chunks indexed"
fi

NGROK_URL=""
if [ "$VOICE" = "1" ]; then
  step "opening ngrok tunnel"
  if ! command -v ngrok >/dev/null 2>&1; then
    c_err "  ngrok is not installed - skipping voice setup"
  elif [ -z "$(envval VAPI_API_KEY)" ]; then
    c_err "  VAPI_API_KEY is not set in .env - skipping voice setup"
  else
    free_port 4040
    ngrok http 8000 --log stdout > "$LOGS/ngrok.log" 2>&1 &
    PIDS+=($!)
    wait_http http://127.0.0.1:4040/api/tunnels 30
    NGROK_URL=$(curl -s -m 5 http://127.0.0.1:4040/api/tunnels \
      | "$PY" -c "
import json,sys
t=[x for x in (json.load(sys.stdin).get('tunnels') or []) if x['public_url'].startswith('https')]
print(t[0]['public_url'] if t else '')
" 2>/dev/null)

    if [ -z "$NGROK_URL" ]; then
      c_err "  tunnel did not come up. Check $LOGS/ngrok.log (auth token set?)"
    else
      c_ok "  tunnel: $NGROK_URL"
      if grep -q '^VAPI_SERVER_URL=' .env; then
        sed -i '' "s|^VAPI_SERVER_URL=.*|VAPI_SERVER_URL=$NGROK_URL|" .env
      else
        printf '\nVAPI_SERVER_URL=%s\n' "$NGROK_URL" >> .env
      fi

      step "pointing the Vapi assistant at this tunnel"
      ASSISTANT=$(envval VAPI_ASSISTANT_ID)
      ARGS=(--server-url "$NGROK_URL")
      [ -n "$ASSISTANT" ] && ARGS+=(--assistant-id "$ASSISTANT")
      if "$PY" scripts/setup_vapi_assistant.py "${ARGS[@]}" 2>&1 | tee "$LOGS/vapi.log" | grep -qE "assistant (updated|created)"; then
        c_ok "  assistant provisioned"
        grep -E "^  VAPI_ASSISTANT_ID=" "$LOGS/vapi.log" || true
        [ -z "$(envval VAPI_WEBHOOK_SECRET)" ] && \
          c_warn "  VAPI_WEBHOOK_SECRET is unset - your webhook is unauthenticated"
      else
        c_err "  provisioning failed - see $LOGS/vapi.log"
      fi

      step "restarting API so it sees the new VAPI_SERVER_URL"
      free_port 8000
      "$PY" -m uvicorn app.main:app --app-dir src --port 8000 --no-access-log > "$LOGS/app.log" 2>&1 &
      PIDS+=($!)
      wait_http http://127.0.0.1:8000/api/v1/health/live 60 && c_ok "  API back up"
    fi
  fi
fi

WEB_UI_URL=""
if command -v npm >/dev/null 2>&1; then
  step "starting React Web UI on :5173"
  free_port 5173
  npm run dev --prefix frontend > "$LOGS/web_ui.log" 2>&1 &
  PIDS+=($!)
  if wait_http http://127.0.0.1:5173 15; then
    c_ok "  React Web UI up"
    WEB_UI_URL="http://localhost:5173"
  fi
fi

cat <<EOF

$(printf '\033[1m%s\033[0m' "─────────────────────────────────────────────────────────")
  Web UI    ${WEB_UI_URL:-http://localhost:5173}
  API docs  http://localhost:8000/docs
  health    http://localhost:8000/api/v1/health/ready
EOF

[ -n "$NGROK_URL" ] && echo "  tunnel    $NGROK_URL"
if [ "$VOICE" = "1" ] && [ -n "$NGROK_URL" ]; then
  echo "  voice     Web UI -> Start Call"
else
  echo "  voice     not started (re-run with: ./start.sh --voice)"
fi
cat <<EOF
  logs      $LOGS/
$(printf '\033[1m%s\033[0m' "─────────────────────────────────────────────────────────")

Press Ctrl+C to stop everything.
EOF

while true; do sleep 3600; done
