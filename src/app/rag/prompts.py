"""Prompt templates.

These are tuned for *spoken* answers, which differ from chat answers in three
ways: no markdown, 1-3 sentences, and an explicit "say you don't know" rule so
the model never invents facts the knowledge base doesn't contain.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.rag.vectorstores.base import ScoredDocument
from app.utils.text import truncate_at_sentence

ASSISTANT_NAME = "Aria"

VOICE_SYSTEM_PROMPT = f"""\
You are {ASSISTANT_NAME}, a helpful voice assistant. Your replies are converted \
to speech and spoken aloud, so they must sound like natural conversation.

GROUNDING RULES
- Answer using ONLY the information in the KNOWLEDGE section below.
- If the KNOWLEDGE section does not contain the answer, say so plainly in one \
sentence and offer to help with something else. Never guess, never invent \
numbers, names, dates, prices, or policies.
- If the knowledge partially covers the question, answer the covered part and \
say what you don't know.

SPEAKING RULES
- Keep answers to 1-3 short sentences (under 60 words) unless asked for detail.
- Plain spoken sentences only: no markdown, no bullet points, no headings, no \
emoji, no code, no URLs.
- Write numbers and units the way a person says them ("about twenty percent", \
"nine a.m. to five p.m.").
- Never read out citation markers, file names, or document ids.
- If the question is ambiguous, ask one short clarifying question instead of \
guessing.
"""

TEXT_SYSTEM_PROMPT = f"""\
You are {ASSISTANT_NAME}, a knowledge assistant answering from a private \
document collection.

- Answer using ONLY the KNOWLEDGE section below. If it is insufficient, say so.
- Be concise and concrete. Cite the source name in brackets after each claim, \
e.g. [handbook.pdf].
- Do not speculate beyond the provided context.
"""

NO_CONTEXT_NOTICE = (
    "(No relevant documents were found in the knowledge base for this question. "
    "Tell the user you don't have that information rather than answering from "
    "general knowledge.)"
)

_CONTEXT_HEADER = "KNOWLEDGE"


def build_context_block(
    documents: Sequence[ScoredDocument],
    *,
    max_chars: int = 6000,
    include_scores: bool = False,
) -> str:
    """Render retrieved chunks into a numbered, source-labelled block.

    Chunks are added highest-score-first and the block stops at ``max_chars`` so
    a large ``top_k`` can never blow the model's context window.
    """
    if not documents:
        return f"{_CONTEXT_HEADER}:\n{NO_CONTEXT_NOTICE}"

    parts: list[str] = []
    budget = max_chars
    for index, scored in enumerate(documents, start=1):
        label = scored.source
        if page := scored.metadata.get("page"):
            label = f"{label} p.{page}"
        score = f" relevance={scored.score:.2f}" if include_scores else ""
        header = f"[{index}] source={label}{score}\n"

        remaining = budget - len(header)
        if remaining < 200:  # not enough room for a useful chunk
            break
        body = truncate_at_sentence(scored.text.strip(), remaining)
        parts.append(f"{header}{body}")
        budget -= len(header) + len(body) + 2

    return f"{_CONTEXT_HEADER}:\n" + "\n\n".join(parts)


def build_user_turn(question: str, context_block: str) -> str:
    return f"{context_block}\n\nUSER QUESTION: {question}"


CONDENSE_QUESTION_PROMPT = """\
Rewrite the user's latest message into a single standalone search query that \
makes sense without the conversation history. Resolve pronouns and references \
using the history. Output ONLY the rewritten query, nothing else.

CONVERSATION:
{history}

LATEST MESSAGE: {question}

STANDALONE QUERY:"""


FALLBACK_ANSWERS = {
    "no_context": (
        "I don't have that in my knowledge base yet. Is there something else "
        "I can help you with?"
    ),
    "provider_error": (
        "Sorry, I'm having trouble reaching my knowledge system right now. "
        "Could you try again in a moment?"
    ),
    "empty_question": "Sorry, I didn't catch that. Could you say it again?",
}
