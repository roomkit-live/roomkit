"""Shared status bus for multi-agent coordination.

All agents post status updates, all agents can subscribe to be notified.
The bus uses a pluggable backend (in-memory, Redis, NATS, etc.).

Usage::

    # In-memory (default)
    bus = StatusBus()

    # Redis backend
    bus = StatusBus(backend=RedisStatusBackend("redis://localhost:6379"))

    # Post from any agent
    bus.post("exec", "search_google", "ok", detail="Found 7 results")

    # Subscribe for real-time notifications
    bus.subscribe(my_callback)

    # Read recent entries
    entries = bus.recent(5, agent_id="exec")
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("roomkit.orchestration.status_bus")

StatusCallback = Callable[["StatusEntry"], Awaitable[None]]


@dataclass
class StatusEntry:
    """A single status update from an agent."""

    ts: str
    agent_id: str
    action: str
    status: str  # ok | failed | pending | info | completed
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "agent_id": self.agent_id,
            "action": self.action,
            "status": self.status,
            "detail": self.detail,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StatusEntry:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------


class StatusBackend(ABC):
    """Abstract backend for status bus persistence and pub/sub.

    Implementations:
    - InMemoryStatusBackend (default)
    - Could be extended with: RedisStatusBackend, NATSStatusBackend, etc.
    """

    @abstractmethod
    async def publish(self, entry: StatusEntry) -> None:
        """Publish a status entry to the backend."""
        ...

    @abstractmethod
    async def recent(
        self, n: int, *, agent_id: str | None = None, status: str | None = None
    ) -> list[StatusEntry]:
        """Retrieve recent entries from the backend."""
        ...

    @abstractmethod
    async def subscribe(self, callback: StatusCallback) -> None:
        """Subscribe to new entries."""
        ...

    @abstractmethod
    async def unsubscribe(self, callback: StatusCallback) -> None:
        """Unsubscribe from entries."""
        ...

    async def close(self) -> None:
        """Close the backend and release resources."""


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class InMemoryStatusBackend(StatusBackend):
    """In-memory backend — entries stored in a list, pub/sub via callbacks.

    Suitable for single-process deployments. For multi-process or
    distributed setups, use a Redis or NATS backend.

    Args:
        max_entries: Maximum entries to keep in memory.
        persist_path: Optional JSONL file for persistence across restarts.
    """

    def __init__(
        self,
        *,
        max_entries: int = 500,
        persist_path: Path | str | None = None,
    ) -> None:
        self._entries: list[StatusEntry] = []
        self._subscribers: list[StatusCallback] = []
        self._max_entries = max_entries
        self._persist_path: Path | None = None
        if persist_path is not None:
            self._persist_path = Path(persist_path)
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

    async def publish(self, entry: StatusEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        # Persist
        if self._persist_path is not None:
            try:
                with open(self._persist_path, "a") as f:
                    f.write(json.dumps(entry.to_dict(), default=str) + "\n")
            except Exception:
                logger.exception("Failed to persist status entry")

        # Notify subscribers
        for cb in self._subscribers:
            try:
                await cb(entry)
            except Exception:
                logger.exception("StatusBus subscriber failed")

    async def recent(
        self, n: int, *, agent_id: str | None = None, status: str | None = None
    ) -> list[StatusEntry]:
        entries = self._entries
        if agent_id is not None:
            entries = [e for e in entries if e.agent_id == agent_id]
        if status is not None:
            entries = [e for e in entries if e.status == status]
        return entries[-n:]

    async def subscribe(self, callback: StatusCallback) -> None:
        self._subscribers.append(callback)

    async def unsubscribe(self, callback: StatusCallback) -> None:
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# StatusBus — main API
# ---------------------------------------------------------------------------


class StatusBus:
    """Shared event log for multi-agent coordination.

    Uses a pluggable backend for persistence and pub/sub.
    The default InMemoryStatusBackend works for single-process setups.
    Swap in a Redis or NATS backend for distributed deployments.

    Args:
        backend: StatusBackend implementation. Defaults to InMemoryStatusBackend.
        persist_path: Shortcut — if set and no backend provided, creates an
            InMemoryStatusBackend with JSONL persistence at this path.
    """

    def __init__(
        self,
        *,
        backend: StatusBackend | None = None,
        persist_path: Path | str | None = None,
    ) -> None:
        if backend is not None:
            self._backend = backend
        else:
            self._backend = InMemoryStatusBackend(
                persist_path=persist_path,
            )

    @staticmethod
    def _log_task_exception(task: asyncio.Task[None]) -> None:
        """Log unhandled exceptions from fire-and-forget publish tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("StatusBus publish task failed: %s", exc, exc_info=exc)

    def post(
        self,
        agent_id: str,
        action: str,
        status: str,
        *,
        detail: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> StatusEntry:
        """Post a status update. Notifies all subscribers.

        Synchronous API — internally schedules the async publish.
        Safe to call from any context (sync tool handlers, hooks, etc.).
        """
        entry = StatusEntry(
            ts=datetime.now(UTC).isoformat(),
            agent_id=agent_id,
            action=action,
            status=status,
            detail=detail,
            metadata=metadata or {},
        )

        logger.info(
            "[%s] %s → %s | %s",
            agent_id, action, status, detail[:80] if detail else "",
        )

        # Schedule async publish
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._backend.publish(entry),
                name=f"status_bus_publish:{agent_id}",
            )
            task.add_done_callback(self._log_task_exception)
        except RuntimeError:
            # No event loop — can happen during shutdown
            pass

        return entry

    async def post_async(
        self,
        agent_id: str,
        action: str,
        status: str,
        *,
        detail: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> StatusEntry:
        """Post a status update (async version). Awaits subscriber notification."""
        entry = StatusEntry(
            ts=datetime.now(UTC).isoformat(),
            agent_id=agent_id,
            action=action,
            status=status,
            detail=detail,
            metadata=metadata or {},
        )

        logger.info(
            "[%s] %s → %s | %s",
            agent_id, action, status, detail[:80] if detail else "",
        )

        await self._backend.publish(entry)
        return entry

    async def subscribe(self, callback: StatusCallback) -> None:
        """Subscribe to status updates."""
        await self._backend.subscribe(callback)

    async def unsubscribe(self, callback: StatusCallback) -> None:
        """Remove a subscriber."""
        await self._backend.unsubscribe(callback)

    async def recent(
        self,
        n: int = 10,
        *,
        agent_id: str | None = None,
        status: str | None = None,
    ) -> list[StatusEntry]:
        """Get the most recent entries."""
        return await self._backend.recent(n, agent_id=agent_id, status=status)

    async def recent_text(self, n: int = 10) -> str:
        """Format recent entries as text for agent context injection."""
        entries = await self.recent(n)
        if not entries:
            return "No activity yet."
        lines = []
        for e in entries:
            lines.append(
                f"[{e.ts[11:19]}] {e.agent_id}: {e.action} → {e.status}"
                + (f" | {e.detail}" if e.detail else "")
            )
        return "\n".join(lines)

    async def has_completed(self, agent_id: str | None = None) -> bool:
        """Check if any task has completed."""
        entries = await self._backend.recent(50, status="completed")
        if agent_id is not None:
            entries = [e for e in entries if e.agent_id == agent_id]
        return len(entries) > 0

    async def close(self) -> None:
        """Close the backend."""
        await self._backend.close()

    def print_summary(self) -> None:
        """Print a formatted summary (sync, reads from in-memory backend)."""
        if isinstance(self._backend, InMemoryStatusBackend):
            entries = self._backend._entries
        else:
            logger.warning("print_summary only works with InMemoryStatusBackend")
            return

        if not entries:
            print("\nNo status entries.")
            return
        print("\nStatus Log")
        print("=" * 60)
        for i, e in enumerate(entries, 1):
            icon = {"ok": "+", "failed": "x", "info": "~", "completed": "*"}.get(e.status, "?")
            print(f"  {i:2d}. [{icon}] {e.agent_id:6s} {e.action}")
            if e.detail:
                print(f"      {e.detail[:100]}")
        ok = sum(1 for e in entries if e.status == "ok")
        fail = sum(1 for e in entries if e.status == "failed")
        done = sum(1 for e in entries if e.status == "completed")
        print(f"\n  Total: {len(entries)} entries ({ok} ok, {fail} failed, {done} completed)")
