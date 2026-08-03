"""Latency instrumentation.

Voice UX lives or dies on time-to-first-token, so every pipeline stage is timed
and the breakdown travels with the response.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Stopwatch:
    """Accumulates named stage durations in milliseconds."""

    started_at: float = field(default_factory=time.perf_counter)
    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round((time.perf_counter() - start) * 1000, 2)

    def mark(self, name: str) -> float:
        """Record elapsed-since-start under ``name`` and return it."""
        elapsed = round((time.perf_counter() - self.started_at) * 1000, 2)
        self.stages[name] = elapsed
        return elapsed

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 2)

    def summary(self) -> dict[str, float]:
        return {**self.stages, "total_ms": self.total_ms}

    def __str__(self) -> str:
        return " ".join(f"{k}={v}ms" for k, v in self.summary().items())
