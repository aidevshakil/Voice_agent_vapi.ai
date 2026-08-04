"""Provider payload tests.

The rest of the suite runs against ``FakeLLM``, which implements ``stream()``
directly and therefore never builds a real request payload. That blind spot let a
groq-incompatible ``stream_options`` kwarg ship and break *every* streamed turn
(i.e. every Vapi custom-llm call) while all tests stayed green.

These tests close it by checking the payload the concrete providers actually
construct against the signature the installed SDK actually accepts. No network:
the SDK clients are inspected, never called.
"""

from __future__ import annotations

import inspect

import pytest

from app.rag.llm.base import ChatMessage
from app.rag.llm.groq_provider import build_groq_provider
from app.rag.llm.openai_provider import build_openai_provider

MESSAGES = [
    ChatMessage(role="system", content="You are a voice assistant."),
    ChatMessage(role="user", content="What are your support hours?"),
]

BUILD_KWARGS = {
    "api_key": "test-key-not-used",
    "model": "test-model",
    "temperature": 0.3,
    "max_tokens": 400,
    "timeout": 30.0,
    "max_retries": 2,
}


def _accepted_params(client) -> set[str]:
    """Keyword names the SDK's chat.completions.create actually accepts.

    Resolved off the *instance*: ``chat`` is a ``cached_property`` on the client
    class, so going through the type yields the descriptor, not the resource.
    """
    signature = inspect.signature(client.chat.completions.create)
    return set(signature.parameters)


@pytest.fixture
def groq_provider():
    return build_groq_provider(**BUILD_KWARGS)


@pytest.fixture
def openai_provider():
    return build_openai_provider(**BUILD_KWARGS)


@pytest.mark.parametrize("stream", [False, True])
def test_groq_payload_is_accepted_by_the_installed_sdk(groq_provider, stream):
    """Regression: groq's SDK rejects unknown kwargs, so every key must be real."""
    payload = groq_provider._payload(MESSAGES, None, None, stream=stream)
    unsupported = set(payload) - _accepted_params(groq_provider._client)
    assert not unsupported, f"groq SDK does not accept: {sorted(unsupported)}"


@pytest.mark.parametrize("stream", [False, True])
def test_openai_payload_is_accepted_by_the_installed_sdk(openai_provider, stream):
    payload = openai_provider._payload(MESSAGES, None, None, stream=stream)
    unsupported = set(payload) - _accepted_params(openai_provider._client)
    assert not unsupported, f"openai SDK does not accept: {sorted(unsupported)}"


def test_groq_omits_stream_options(groq_provider):
    """The exact kwarg that broke voice: groq must never receive it."""
    assert "stream_options" not in groq_provider._payload(MESSAGES, None, None, stream=True)


def test_openai_requests_stream_options(openai_provider):
    """OpenAI supports it, so the usage-reporting intent is preserved there."""
    payload = openai_provider._payload(MESSAGES, None, None, stream=True)
    assert payload["stream_options"] == {"include_usage": True}


def test_stream_options_never_sent_on_buffered_calls(openai_provider):
    assert "stream_options" not in openai_provider._payload(MESSAGES, None, None, stream=False)


def test_payload_carries_messages_and_overrides(groq_provider):
    payload = groq_provider._payload(MESSAGES, 0.9, 128, stream=False)
    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0.9
    assert payload["max_tokens"] == 128
    assert payload["messages"] == [
        {"role": "system", "content": "You are a voice assistant."},
        {"role": "user", "content": "What are your support hours?"},
    ]


def test_payload_falls_back_to_configured_defaults(groq_provider):
    payload = groq_provider._payload(MESSAGES, None, None, stream=False)
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 400
