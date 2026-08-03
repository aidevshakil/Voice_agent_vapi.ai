"""Text normalisation, sentence buffering, caching, and config parsing."""

from __future__ import annotations

import asyncio

import pytest

from app.services.cache import TTLCache
from app.utils.text import (
    SentenceBuffer,
    content_hash,
    split_sentences,
    strip_markdown,
    to_speech_friendly,
    truncate_at_sentence,
)


# ---------------------------------------------------------------- speech prep
@pytest.mark.parametrize(
    ("raw", "must_not_contain"),
    [
        ("**bold** and *italic*", ["*"]),
        ("# Heading\n\nBody text.", ["#"]),
        ("- one\n- two", ["-"]),
        ("Use `pip install x`", ["`"]),
        ("See [the docs](https://example.com)", ["[", "]", "http"]),
        ("Coffee & tea", ["&"]),
        ("Growth is 20%", ["%"]),
        ("```python\nprint(1)\n```", ["```"]),
    ],
)
def test_markdown_never_reaches_tts(raw: str, must_not_contain: list[str]):
    spoken = to_speech_friendly(raw)
    for token in must_not_contain:
        assert token not in spoken, f"{token!r} survived in {spoken!r}"


def test_symbols_become_words():
    spoken = to_speech_friendly("Revenue & profit rose 15% @ HQ")
    assert "and" in spoken
    assert "percent" in spoken
    assert "at" in spoken


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$29", "29 dollars"),
        ("$1.2M", "1.2 million dollars"),
        ("€500", "500 euros"),
        ("£15k", "15 thousand pounds"),
    ],
)
def test_currency_is_spoken(raw: str, expected: str):
    assert expected in to_speech_friendly(f"It costs {raw} per month.")


def test_abbreviations_are_expanded():
    spoken = to_speech_friendly("Use a proxy, e.g. nginx, etc.")
    assert "for example" in spoken
    assert "and so on" in spoken
    assert "e.g." not in spoken


def test_citation_markers_are_removed():
    assert "source" not in to_speech_friendly("Hours are nine to six [source: handbook.pdf].").lower()


def test_strip_markdown_keeps_prose_and_paragraphs():
    assert strip_markdown("## Title\n\nHello **world**.") == "Title\n\nHello world."


def test_strip_markdown_unwraps_nested_emphasis():
    assert strip_markdown("This is ***very*** important") == "This is very important"


def test_strip_markdown_handles_tables_and_lists():
    cleaned = strip_markdown(
        "| Plan | Price |\n| --- | --- |\n| Starter | $29 |\n\n1. first\n2. second\n> quoted"
    )
    assert not cleaned.startswith("|")
    assert "---" not in cleaned
    assert all(token in cleaned for token in ("Plan", "Starter", "first", "quoted"))


def test_table_cells_become_spoken_pauses():
    spoken = to_speech_friendly("| Starter | 29 |")
    assert "|" not in spoken
    assert "Starter" in spoken


def test_empty_input_is_safe():
    assert to_speech_friendly("") == ""


# ------------------------------------------------------------ sentence buffer
def test_sentence_buffer_releases_on_boundaries():
    buffer = SentenceBuffer()
    released: list[str] = []
    for delta in ["Support ", "is open ", "nine to six. ", "Call us ", "any time. "]:
        released.extend(buffer.push(delta))

    assert len(released) == 2
    assert released[0].startswith("Support is open")
    assert buffer.drain() == ""


def test_sentence_buffer_reassembles_split_markdown():
    """The whole point: `*` + `*` arriving separately must not leak."""
    buffer = SentenceBuffer()
    released: list[str] = []
    for delta in ["*", "*Support", "*", "*", " is open nine to six. "]:
        released.extend(buffer.push(delta))

    assert released
    assert "*" not in "".join(released)


def test_sentence_buffer_does_not_split_abbreviations():
    buffer = SentenceBuffer()
    released = buffer.push("Contact Dr. Chen for details. ")
    assert len(released) == 1


def test_sentence_buffer_does_not_split_decimals():
    buffer = SentenceBuffer()
    assert buffer.push("Version 3.5 is current") == []


def test_sentence_buffer_force_flushes_long_runs():
    buffer = SentenceBuffer(flush_at=50)
    released = buffer.push("word " * 30)
    assert released  # doesn't wait forever for punctuation


def test_sentence_buffer_drain_returns_the_tail():
    buffer = SentenceBuffer()
    buffer.push("No trailing punctuation here")
    assert "No trailing punctuation here" in buffer.drain()


# ------------------------------------------------------------------- helpers
def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_truncate_uses_a_sentence_boundary_when_it_keeps_most_of_the_budget():
    text = "First sentence is here. Second sentence follows on. Third one too."
    result = truncate_at_sentence(text, 55)

    assert len(result) <= 55
    assert result == "First sentence is here. Second sentence follows on."


def test_truncate_never_exceeds_the_budget():
    text = "First sentence is here. Second sentence follows on. Third one too."
    for budget in range(10, 70, 7):
        assert len(truncate_at_sentence(text, budget)) <= budget


def test_truncate_leaves_short_text_alone():
    assert truncate_at_sentence("Short.", 100) == "Short."


def test_content_hash_is_stable_and_distinguishing():
    assert content_hash("a", "b") == content_hash("a", "b")
    assert content_hash("a", "b") != content_hash("ab", "")


# --------------------------------------------------------------------- cache
async def test_cache_hit_and_miss():
    cache: TTLCache[str] = TTLCache(max_size=4, ttl_seconds=10)

    assert await cache.get("k") is None
    await cache.set("k", "v")
    assert await cache.get("k") == "v"
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == 0.5


async def test_cache_evicts_least_recently_used():
    cache: TTLCache[int] = TTLCache(max_size=2, ttl_seconds=10)
    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.get("a")  # refresh a
    await cache.set("c", 3)  # evicts b

    assert await cache.get("a") == 1
    assert await cache.get("b") is None
    assert cache.stats.evictions == 1


async def test_cache_expires_entries():
    cache: TTLCache[str] = TTLCache(max_size=4, ttl_seconds=0.05)
    await cache.set("k", "v")
    await asyncio.sleep(0.08)
    assert await cache.get("k") is None


async def test_cache_can_be_disabled():
    cache: TTLCache[str] = TTLCache(max_size=0, ttl_seconds=10)
    await cache.set("k", "v")
    assert await cache.get("k") is None
    assert cache.enabled is False


# -------------------------------------------------------------------- config
def test_csv_env_values_parse_into_lists():
    from app.core.config import AppSettings

    assert AppSettings(cors_origins="http://a.io, http://b.io").cors_origins == [  # type: ignore[arg-type]
        "http://a.io",
        "http://b.io",
    ]


def test_rag_settings_reject_bad_invariants():
    from app.core.config import RAGSettings

    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        RAGSettings(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError, match="FETCH_K"):
        RAGSettings(top_k=10, fetch_k=2)


def test_missing_llm_key_is_a_clear_error():
    from app.core.config import Settings

    settings = Settings(llm={"provider": "openai"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.require_llm_api_key()
