"""Completed task cache for delegation dedup.

Prevents re-delegation of recently completed tasks by caching results
keyed on ``(room_id, agent_id, task_hash)``.  Entries expire after a
configurable TTL.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _CacheKey:
    room_id: str
    agent_id: str
    task_hash: str


@dataclass
class _CacheEntry:
    result: dict[str, Any]
    created_at: float
    task_text: str


class CompletedTaskCache:
    """In-memory TTL cache for completed delegation results.

    Args:
        ttl_seconds: Seconds before a cached result expires.  Defaults to 300 (5 min).
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._entries: dict[_CacheKey, _CacheEntry] = {}

    @staticmethod
    def _hash_task(task: str) -> str:
        return hashlib.sha256(task.strip().lower().encode()).hexdigest()[:16]

    def put(
        self,
        room_id: str,
        agent_id: str,
        task: str,
        result: dict[str, Any],
    ) -> None:
        """Store a completed task result."""
        key = _CacheKey(room_id, agent_id, self._hash_task(task))
        self._entries[key] = _CacheEntry(
            result=result,
            created_at=time.monotonic(),
            task_text=task,
        )

    def get(
        self,
        room_id: str,
        agent_id: str,
        task: str,
    ) -> dict[str, Any] | None:
        """Return cached result if present and not expired, else None."""
        key = _CacheKey(room_id, agent_id, self._hash_task(task))
        entry = self._entries.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry.created_at > self._ttl:
            del self._entries[key]
            return None
        return entry.result

    def recent_context(self, room_id: str, limit: int = 5) -> list[str]:
        """Return summaries of recent tasks for a room (for context injection).

        Returns task descriptions from newest to oldest, up to *limit*.
        Expired entries are skipped.
        """
        now = time.monotonic()
        relevant: list[tuple[float, str]] = []
        expired_keys: list[_CacheKey] = []

        for key, entry in self._entries.items():
            if key.room_id != room_id:
                continue
            if now - entry.created_at > self._ttl:
                expired_keys.append(key)
                continue
            relevant.append((entry.created_at, entry.task_text))

        for k in expired_keys:
            del self._entries[k]

        relevant.sort(key=lambda t: t[0], reverse=True)
        return [text for _, text in relevant[:limit]]

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
