"""Completed task cache for delegation dedup.

Prevents re-delegation of recently completed tasks by caching results
keyed on ``(room_id, agent_id, task_hash)``, the hash reading the task and
what the call passed with it: another context is other work (RFC §23.4). An
entry also keeps the agent whose call ran it, whose repeat alone it answers.
Entries expire after a configurable TTL.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _CacheKey:
    room_id: str
    agent_id: str
    task_hash: str
    answers: bool = True
    """Whether the entry answers a repeat."""
    caller: str | None = None
    """The agent whose call it ran for: what it returned may hold what that
    agent passed or may read. An entry the host wrote has none."""


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
    def _hash_task(task: str, context: Mapping[str, Any] | None = None) -> str:
        """The key of *task*, whatever its case and outer spaces, and of what
        was passed with it, whatever the order of its keys."""
        key = json.dumps([task.strip().lower(), dict(context or {})], sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def put(
        self,
        room_id: str,
        agent_id: str,
        task: str,
        result: dict[str, Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Store a completed task result, under *task* and the *context* it
        was run with."""
        self._keep(room_id, agent_id, task, result, context, None, answers=True)

    def _keep(
        self,
        room_id: str,
        agent_id: str,
        task: str,
        result: dict[str, Any],
        context: Mapping[str, Any] | None,
        caller: str | None,
        *,
        answers: bool,
    ) -> None:
        """Keep a completed task, for the agent that called (*caller*): *answers*
        says whether what it returned answers a repeat, which one depending on
        when it ran does not."""
        key = _CacheKey(
            room_id, agent_id, self._hash_task(task, context), answers=answers, caller=caller
        )
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
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Return cached result of *task* run with *context*, if present and
        not expired, else None: the host's read, whichever agent's call ran it,
        the latest when several did."""
        task_hash = self._hash_task(task, context)
        kept = [
            key
            for key in self._entries
            if key.answers
            and (key.room_id, key.agent_id, key.task_hash) == (room_id, agent_id, task_hash)
        ]
        fresh = [entry for key in kept if (entry := self._fresh(key)) is not None]
        return max(fresh, key=lambda entry: entry.created_at).result if fresh else None

    def _answer_for(
        self,
        room_id: str,
        agent_id: str,
        task: str,
        context: Mapping[str, Any] | None,
        caller: str,
    ) -> dict[str, Any] | None:
        """What answers *caller*'s repeat of *task*: its own call's result or
        one the host wrote, the latest of the two, never another agent's, which
        may hold what that agent passed or may read (RFC §23.4)."""
        task_hash = self._hash_task(task, context)
        keys = (_CacheKey(room_id, agent_id, task_hash, caller=who) for who in (caller, None))
        fresh = [entry for key in keys if (entry := self._fresh(key)) is not None]
        return max(fresh, key=lambda entry: entry.created_at).result if fresh else None

    def _fresh(self, key: _CacheKey) -> _CacheEntry | None:
        """The entry of *key*, dropped once expired."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry.created_at > self._ttl:
            del self._entries[key]
            return None
        return entry

    def recent_context(self, room_id: str, limit: int = 5) -> list[str]:
        """Return summaries of recent tasks for a room (for context injection).

        Returns task descriptions from newest to oldest, up to *limit*, each
        once, whatever its case and outer spaces, however many times it ran.
        Expired entries are skipped.
        """
        return self._recent(room_id, limit)

    def _recent_of(self, room_id: str, caller: str, limit: int) -> list[str]:
        """:meth:`recent_context` for *caller*: its own tasks and those the host
        wrote, never another agent's, which may hold what *caller* may not read
        (RFC §23.4)."""
        return self._recent(room_id, limit, lambda key: key.caller in (None, caller))

    def _recent(
        self,
        room_id: str,
        limit: int,
        read: Callable[[_CacheKey], bool] = lambda _: True,
    ) -> list[str]:
        now = time.monotonic()
        relevant: list[tuple[float, str]] = []
        expired_keys: list[_CacheKey] = []

        for key, entry in self._entries.items():
            if key.room_id != room_id:
                continue
            if now - entry.created_at > self._ttl:
                expired_keys.append(key)
                continue
            if read(key):
                relevant.append((entry.created_at, entry.task_text))

        for k in expired_keys:
            del self._entries[k]

        relevant.sort(key=lambda t: t[0], reverse=True)
        seen: set[str] = set()
        once: list[str] = []
        for _, text in relevant:
            if (folded := text.strip().lower()) not in seen:
                seen.add(folded)
                once.append(text)
        return once[:limit]

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
