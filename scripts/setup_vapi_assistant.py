#!/usr/bin/env python
"""Create or update the Vapi assistant so its configuration lives in this repo.

Prerequisites: VAPI_API_KEY set, and your backend reachable at a public HTTPS URL
(``ngrok http 8000`` during development -- Vapi cannot call localhost).

    python scripts/setup_vapi_assistant.py --server-url https://abc123.ngrok-free.app
    python scripts/setup_vapi_assistant.py --mode tool          # tool-calling instead
    python scripts/setup_vapi_assistant.py --list               # show existing assistants
    python scripts/setup_vapi_assistant.py --print-only         # dump the payload, send nothing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.vapi_client import VapiClient

logger = get_logger("setup-vapi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-url", default=None, help="Public HTTPS base URL of this backend.")
    parser.add_argument("--name", default="RAG Voice Assistant")
    parser.add_argument("--first-message", default=None)
    parser.add_argument(
        "--mode",
        choices=("custom-llm", "tool"),
        default="custom-llm",
        help="custom-llm grounds every turn through this backend (recommended).",
    )
    parser.add_argument("--assistant-id", default=None, help="Update instead of create.")
    parser.add_argument("--list", action="store_true", help="List assistants and exit.")
    parser.add_argument("--print-only", action="store_true", help="Print the payload without calling Vapi.")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    configure_logging(level="INFO", json_output=False)
    settings = get_settings()
    client = VapiClient(settings)

    try:
        if args.list:
            if not client.configured:
                print("VAPI_API_KEY is not set.")
                return 1
            assistants = await client.list_assistants()
            if not assistants:
                print("No assistants found on this account.")
                return 0
            print(f"\n{len(assistants)} assistant(s):")
            for item in assistants:
                model = (item.get("model") or {}).get("provider", "?")
                print(f"  {item.get('id')}  {item.get('name')!r}  model={model}")
            return 0

        payload = client.build_assistant_payload(
            name=args.name,
            server_url=args.server_url,
            first_message=args.first_message,
            use_custom_llm=args.mode == "custom-llm",
        )
        if args.print_only:
            # build_assistant_payload embeds the shared secret in every place Vapi
            # needs it (assistant server, tool server, custom-llm headers).
            redacted = json.loads(json.dumps(payload))
            if "secret" in redacted.get("server", {}):
                redacted["server"]["secret"] = "***"
            if "x-vapi-secret" in redacted.get("model", {}).get("headers", {}):
                redacted["model"]["headers"]["x-vapi-secret"] = "***"
            print(json.dumps(redacted, indent=2))
            return 0

        if not client.configured:
            print("VAPI_API_KEY is not set. Add it to .env and try again.")
            return 1

        target = args.assistant_id or settings.vapi.assistant_id
        if target:
            result = await client.update_assistant(target, payload)
            action, assistant_id = "updated", target
        else:
            result = await client.create_assistant(payload)
            action, assistant_id = "created", str(result.get("id", ""))

        base = (args.server_url or settings.vapi.server_url or "").rstrip("/")
        print("\n" + "=" * 62)
        print(f"assistant {action}: {assistant_id}")
        print("=" * 62)
        print(f"mode        : {args.mode}")
        print(f"webhook     : {base}/api/v1/vapi/webhook")
        if args.mode == "custom-llm":
            print(f"custom llm  : {base}/api/v1/vapi/chat/completions")
        print(f"voice       : {settings.vapi.voice_provider}/{settings.vapi.voice_id}")
        print(f"transcriber : {settings.vapi.transcriber_provider}/{settings.vapi.transcriber_model}")
        if not settings.vapi.webhook_secret:
            print("\nWARNING: VAPI_WEBHOOK_SECRET is unset -- your webhook is unauthenticated.")
        print(f"\nAdd this to .env:\n  VAPI_ASSISTANT_ID={assistant_id}")
        print("=" * 62)
        return 0
    finally:
        await client.aclose()


def main() -> int:
    """Turn expected failures into one readable line instead of a traceback.

    A rejected field or a bad key is a configuration mistake, not a crash, and a
    60-line httpx stack buries the one sentence that says what to fix.
    """
    from app.core.exceptions import AppError

    try:
        return asyncio.run(run(parse_args()))
    except AppError as exc:
        print(f"\nERROR: {exc.message}", file=sys.stderr)
        if body := exc.details.get("body"):
            print(f"vapi said: {body}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
