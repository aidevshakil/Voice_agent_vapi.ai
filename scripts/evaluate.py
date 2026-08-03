#!/usr/bin/env python
"""Retrieval + answer smoke evaluation.

Runs a small question set through the pipeline and reports latency and grounding.
Point it at your own questions with ``--file questions.json``:

    [{"question": "...", "expect_source": "handbook.pdf", "expect_terms": ["30 days"]}]

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --file my_questions.json --retrieval-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.core.config import get_settings
from app.core.container import Container
from app.core.logging import configure_logging

DEFAULT_QUESTIONS = [
    {"question": "What are your support hours?"},
    {"question": "How do I reset my password?"},
    {"question": "What is the refund policy?"},
    {"question": "Which plans include priority support?"},
    {"question": "Who is the CEO of Mars?"},  # deliberately unanswerable
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=None, help="JSON file of questions.")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip generation (no LLM cost).")
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    configure_logging(level="WARNING", json_output=False)
    settings = get_settings()
    container = await Container.create(settings)

    questions = json.loads(args.file.read_text(encoding="utf-8")) if args.file else DEFAULT_QUESTIONS
    latencies: list[float] = []
    grounded_count = 0

    try:
        print(f"\nevaluating {len(questions)} question(s) against "
              f"{container.vector_store.count()} vectors\n" + "-" * 78)

        for index, item in enumerate(questions, start=1):
            question = item["question"]
            if args.retrieval_only:
                outcome = await container.pipeline.retriever.retrieve(
                    question, top_k=args.top_k, use_cache=False
                )
                total = outcome.timings.get("total_ms", 0.0)
                grounded = bool(outcome.documents)
                summary = ", ".join(f"{d.source}({d.score:.2f})" for d in outcome.documents) or "-"
                sources = outcome.sources
            else:
                answer = await container.pipeline.answer(
                    question, mode="voice", top_k=args.top_k, use_cache=False
                )
                total = answer.timings.get("total_ms", 0.0)
                grounded = answer.grounded
                summary = answer.answer
                sources = [c.source for c in answer.citations]

            latencies.append(total)
            grounded_count += int(grounded)

            print(f"[{index}] {question}")
            print(f"    grounded={grounded} {total:.0f}ms sources={sources}")
            print(f"    {summary[:300]}")

            if expected := item.get("expect_source"):
                mark = "PASS" if expected in sources else "FAIL"
                print(f"    expect_source={expected} -> {mark}")
            if terms := item.get("expect_terms"):
                missing = [t for t in terms if t.lower() not in summary.lower()]
                print(f"    expect_terms -> {'PASS' if not missing else f'FAIL missing={missing}'}")
            print()

        print("-" * 78)
        print(f"grounded    : {grounded_count}/{len(questions)}")
        print(f"latency p50 : {statistics.median(latencies):.0f}ms")
        print(f"latency max : {max(latencies):.0f}ms")
        print(f"cache       : {container.pipeline.retriever.cache_stats}")
        return 0
    finally:
        await container.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
