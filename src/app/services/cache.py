"""Async-safe TTL + LRU cache.

Retrieval is the slowest deterministic step in the pipeline and callers repeat
themselves constantly ("what are your hours?" phrased five ways still normalises
to the same key), so caching query -> chunks is the single cheapest latency win.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }


class TTLCache(Generic[T]):
    """Bounded cache with per-entry expiry. ``max_size=0`` disables caching."""

    def __init__(self, max_size: int = 512, ttl_seconds: float = 300.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._data: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = asyncio.Lock()
        self.stats = CacheStats()

    @property
    def enabled(self) -> bool:
        return self._max_size > 0 and self._ttl > 0

    async def get(self, key: str) -> T | None:
        if not self.enabled:
            return None
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            expires_at, value = entry
            if expires_at < time.monotonic():
                del self._data[key]
                self.stats.misses += 1
                return None
            self._data.move_to_end(key)
            self.stats.hits += 1
            return value

    async def set(self, key: str, value: T) -> None:
        if not self.enabled:
            return
        async with self._lock:
            self._data[key] = (time.monotonic() + self._ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)
                self.stats.evictions += 1

    async def clear(self) -> None:
        async with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
