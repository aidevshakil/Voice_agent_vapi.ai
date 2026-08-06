"""Text helpers, most of them specific to making LLM output *speakable*.

TTS engines read literally: "~$1.2M (Q3'24)" becomes noise. Normalising before
the text reaches Vapi is cheaper and more reliable than prompting for it.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable

# Horizontal whitespace only -- newlines are preserved so paragraph structure
# survives. \S covers Unicode spaces (NBSP, figure space) without embedding
# ambiguous characters in this source file.
_WHITESPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_CODE_FENCE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
_LINE_MARKUP = re.compile(
    r"""
    ^\s*[-*_]{3,}\s*$        # horizontal rules (before bullets: --- is not a list)
  | ^\#{1,6}\s*              # heading markers
  | ^\s*[-*+]\s+             # bullet markers
  | ^\s*\d+[.)]\s+           # ordered-list markers
  | ^\s*>\s?                 # blockquotes
  | ^\s*\|                   # leading table pipe
  | \|\s*$                   # trailing table pipe
    """,
    re.VERBOSE | re.MULTILINE,
)
# A row of --- | :-: separators carries no spoken content at all.
_TABLE_DIVIDER = re.compile(r"^[\s|:-]+$", re.MULTILINE)
# Paired emphasis, longest delimiter first so ** is consumed before *.
_EMPHASIS = re.compile(r"(\*\*\*|\*\*|___|__|\*|_)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_STRAY_ASTERISK = re.compile(r"\*+")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_CITATION = re.compile(r"\[(?:source|doc|ref)[^\]]*\]", re.IGNORECASE)
_URL = re.compile(r"https?://\S+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_ABBREVIATIONS = {
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "and so on",
    "vs.": "versus",
    "approx.": "approximately",
}
_SYMBOLS = {
    "&": " and ",
    "%": " percent ",
    "+": " plus ",
    "=": " equals ",
    "@": " at ",
    "#": " number ",
    "/": " or ",
    "~": " about ",
    "→": " leads to ",
    "•": " ",
    "|": ", ",  # remaining table cell separators become spoken pauses
}
_CURRENCY = {"$": "dollars", "€": "euros", "£": "pounds", "₹": "rupees"}
_MAGNITUDES = {"k": "thousand", "m": "million", "b": "billion", "t": "trillion"}


def normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


def strip_markdown(text: str) -> str:
    """Remove markdown scaffolding while keeping the prose intact.

    Order matters: fences first (their contents must not be parsed as markup),
    then links and inline code, then line-anchored markers, then emphasis. A final
    pass removes any unpaired ``*`` — an asterisk has no spoken form, so leaving
    one behind is always worse than dropping it. ``_`` is left alone because it
    appears inside real identifiers.
    """
    text = _CODE_FENCE.sub(" code block omitted ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _TABLE_DIVIDER.sub("", text)
    text = _LINE_MARKUP.sub("", text)
    # Twice, so nested emphasis (***both***) fully unwraps.
    for _ in range(2):
        text = _EMPHASIS.sub(r"\2", text)
    text = _STRAY_ASTERISK.sub("", text)
    return normalize_whitespace(text)


def _expand_currency(text: str) -> str:
    pattern = "|".join(re.escape(sym) for sym in _CURRENCY)

    def replace(match: re.Match[str]) -> str:
        amount, magnitude = match.group("amount"), (match.group("mag") or "").lower()
        unit = _CURRENCY[match.group("sym")]
        suffix = f" {_MAGNITUDES[magnitude]}" if magnitude in _MAGNITUDES else ""
        return f"{amount}{suffix} {unit}"

    return re.sub(
        rf"(?P<sym>{pattern})\s?(?P<amount>[\d,]+(?:\.\d+)?)(?P<mag>[kKmMbBtT])?\b",
        replace,
        text,
    )


def to_speech_friendly(text: str) -> str:
    """Make model output safe to hand to a TTS engine."""
    if not text:
        return ""
    text = strip_markdown(text)
    text = _CITATION.sub("", text)
    text = _URL.sub("the link on screen", text)
    text = _expand_currency(text)
    for abbrev, spoken in _ABBREVIATIONS.items():
        text = re.sub(rf"(?<!\w){re.escape(abbrev)}", spoken, text, flags=re.IGNORECASE)
    for symbol, spoken in _SYMBOLS.items():
        text = text.replace(symbol, spoken)
    # Bare newlines become sentence pauses rather than run-on speech.
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"\.\s*\.(\s*\.)*", ".", text)
    return _WHITESPACE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """Trim to at most ``max_chars``, preferring a sentence boundary.

    A boundary is only used if it keeps more than 60% of the budget; otherwise
    truncating there would waste most of the allowance, so we cut at a word
    instead. The result never exceeds ``max_chars``, ellipsis included.
    """
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    for boundary in (". ", "! ", "? ", "\n"):
        cut = window.rfind(boundary)
        if cut > max_chars * 0.6:
            return window[: cut + 1].strip()
    # Reserve room for the ellipsis so the contract holds.
    clipped = text[: max(0, max_chars - 3)]
    return clipped.rsplit(" ", 1)[0].strip() + "..."


_NON_TERMINAL = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e", "approx", "no", "inc", "ltd"}
)


def _is_false_stop(buffer: str, index: int) -> bool:
    """True when the period at ``index`` is an abbreviation or initial, not a stop."""
    start = index
    while start > 0 and (buffer[start - 1].isalnum() or buffer[start - 1] == "."):
        start -= 1
    token = buffer[start:index].lower().rstrip(".")
    if not token:
        return False
    if token in _NON_TERMINAL:
        return True
    return len(token) == 1 and token.isalpha()  # "J. Smith"


class SentenceBuffer:
    """Accumulates streaming deltas and releases speech-ready sentences.

    Normalising each raw delta in isolation doesn't work: a ``**`` bold marker can
    arrive as two separate ``*`` tokens, and a sentence-ending period can arrive
    before the space that terminates it. Buffering to the next boundary and
    normalising whole sentences is both correct and still fast — TTS engines can't
    start synthesising a partial sentence anyway, so nothing is lost.

    ``flush_at`` bounds the wait for models that emit long unpunctuated runs.
    """

    __slots__ = ("_buffer", "_flush_at", "_transform")

    def __init__(self, transform: Callable[[str], str] | None = None, flush_at: int = 60) -> None:
        self._buffer = ""
        self._flush_at = flush_at
        self._transform = transform or to_speech_friendly

    def push(self, delta: str) -> list[str]:
        self._buffer += delta
        released: list[str] = []

        while True:
            cut = self._boundary()
            if cut is None:
                break
            candidate, self._buffer = self._buffer[:cut], self._buffer[cut:].lstrip()
            if spoken := self._transform(candidate):
                released.append(spoken)
        return released

    def _boundary(self) -> int | None:
        buffer = self._buffer
        for index, char in enumerate(buffer):
            if char == "\n":
                return index + 1
            if char not in ".!?,;:" or index + 1 >= len(buffer):
                continue
            if not buffer[index + 1].isspace():
                continue
            if char == "." and _is_false_stop(buffer, index):
                continue
            return index + 1

        if len(buffer) >= self._flush_at:
            space = buffer.rfind(" ", 0, self._flush_at)
            return space + 1 if space > 0 else self._flush_at
        return None

    def drain(self) -> str:
        """Return whatever is left, normalised. Call once the stream ends."""
        remaining, self._buffer = self._buffer, ""
        return self._transform(remaining)


def content_hash(*parts: str) -> str:
    """Stable id for chunk deduplication and cache keys."""
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(part.encode("utf-8", errors="ignore"))
        digest.update(b"\x00")
    return digest.hexdigest()


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token) — good enough for budgeting."""
    return max(1, len(text) // 4)
