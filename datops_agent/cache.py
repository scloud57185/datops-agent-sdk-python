"""
DatOps Agent SDK — Thread-safe TTL cache

Replaces Redis caching for external developers.
Uses pure Python stdlib (threading.Lock + time.monotonic).
"""

import threading
import time
from typing import Any, Optional


class TrustCache:
    """Thread-safe in-memory cache with TTL expiration."""

    def __init__(self, default_ttl: int = 60):
        self._store: dict = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get cached value. Returns None if expired or missing."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with TTL in seconds."""
        ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            self._store[key] = (value, time.monotonic() + ttl)

    def delete(self, key: str) -> None:
        """Delete a cached entry."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._store.clear()

    def _cleanup_expired(self) -> int:
        """Remove expired entries. Returns number removed."""
        now = time.monotonic()
        removed = 0
        with self._lock:
            expired_keys = [
                k for k, (_, expires_at) in self._store.items()
                if now > expires_at
            ]
            for k in expired_keys:
                del self._store[k]
                removed += 1
        return removed
