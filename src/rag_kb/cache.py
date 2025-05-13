"""In-memory TTL cache for repeated queries under concurrent load."""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Optional

from cachetools import TTLCache

from rag_kb.config import get_settings


class QueryCache:
    """Thread-safe TTL cache keyed by normalized query + mode flags."""

    def __init__(self, maxsize: int = 512, ttl: Optional[int] = None) -> None:
        settings = get_settings()
        self._cache: TTLCache = TTLCache(
            maxsize=maxsize,
            ttl=ttl if ttl is not None else settings.cache_ttl_seconds,
        )
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(query: str, **flags: Any) -> str:
        raw = query.strip().lower() + "|" + "|".join(f"{k}={v}" for k, v in sorted(flags.items()))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            value = self._cache.get(key)
            if value is None:
                self.misses += 1
            else:
                self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "size": len(self._cache),
            }


# Process-wide singleton used by the API layer
query_cache = QueryCache()
