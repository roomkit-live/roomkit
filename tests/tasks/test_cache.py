"""Tests for CompletedTaskCache and delegation state tracking."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from roomkit.tasks.cache import CompletedTaskCache
from roomkit.tasks.delegate import DelegateHandler
from roomkit.tasks.models import DelegatedTask


class TestCompletedTaskCache:
    def test_put_and_get(self) -> None:
        cache = CompletedTaskCache(ttl_seconds=60.0)
        cache.put("room-1", "agent-a", "search for cats", {"status": "done"})

        result = cache.get("room-1", "agent-a", "search for cats")
        assert result == {"status": "done"}

    def test_get_returns_none_when_missing(self) -> None:
        cache = CompletedTaskCache()
        assert cache.get("room-1", "agent-a", "unknown task") is None

    def test_get_returns_none_when_expired(self) -> None:
        cache = CompletedTaskCache(ttl_seconds=0.01)
        cache.put("room-1", "agent-a", "task", {"status": "done"})
        time.sleep(0.02)
        assert cache.get("room-1", "agent-a", "task") is None

    def test_case_insensitive_hash(self) -> None:
        """Task hash should be case-insensitive."""
        cache = CompletedTaskCache()
        cache.put("room-1", "agent-a", "Search For Cats", {"found": True})
        assert cache.get("room-1", "agent-a", "search for cats") is not None

    def test_different_rooms_isolated(self) -> None:
        cache = CompletedTaskCache()
        cache.put("room-1", "agent-a", "task", {"r1": True})
        assert cache.get("room-2", "agent-a", "task") is None

    def test_recent_context(self) -> None:
        cache = CompletedTaskCache()
        cache.put("room-1", "a", "first task", {})
        cache.put("room-1", "b", "second task", {})
        cache.put("room-2", "a", "other room", {})

        recent = cache.recent_context("room-1", limit=5)
        assert len(recent) == 2
        assert "second task" in recent[0]  # newest first

    def test_recent_context_respects_limit(self) -> None:
        cache = CompletedTaskCache()
        for i in range(5):
            cache.put("room-1", "a", f"task {i}", {})
        assert len(cache.recent_context("room-1", limit=2)) == 2

    def test_clear(self) -> None:
        cache = CompletedTaskCache()
        cache.put("room-1", "a", "task", {})
        cache.clear()
        assert cache.get("room-1", "a", "task") is None


class TestDelegateHandlerWithCache:
    async def test_returns_cached_result(self) -> None:
        """Should return cached result instead of re-delegating."""
        kit = MagicMock()
        cache = CompletedTaskCache()
        cache.put("room-1", "exec-agent", "search cats", {
            "status": "delegated",
            "task_id": "t-old",
            "child_room_id": "child-old",
            "agent_id": "exec-agent",
        })

        handler = DelegateHandler(kit, cache=cache)
        result = await handler.handle(
            room_id="room-1",
            calling_agent_id="voice-agent",
            arguments={"agent": "exec-agent", "task": "search cats"},
        )

        assert result["from_cache"] is True
        assert result["task_id"] == "t-old"
        kit.delegate.assert_not_called()

    async def test_delegates_when_not_cached(self) -> None:
        """Should delegate normally when no cache hit."""
        kit = MagicMock()
        task_handle = DelegatedTask(
            id="t1",
            child_room_id="child-1",
            parent_room_id="room-1",
            agent_id="exec-agent",
            task="new task",
        )
        task_handle.wait = AsyncMock()
        kit.delegate = AsyncMock(return_value=task_handle)

        cache = CompletedTaskCache()
        handler = DelegateHandler(kit, cache=cache)

        result = await handler.handle(
            room_id="room-1",
            calling_agent_id="voice-agent",
            arguments={"agent": "exec-agent", "task": "new task"},
        )

        assert result["status"] == "delegated"
        assert "from_cache" not in result
        kit.delegate.assert_called_once()

    async def test_injects_previous_task_context(self) -> None:
        """Should inject previous task summaries into context."""
        kit = MagicMock()
        task_handle = DelegatedTask(
            id="t1",
            child_room_id="child-1",
            parent_room_id="room-1",
            agent_id="exec-agent",
            task="do more work",
        )
        task_handle.wait = AsyncMock()
        kit.delegate = AsyncMock(return_value=task_handle)

        cache = CompletedTaskCache()
        cache.put("room-1", "other-agent", "previous task", {"done": True})

        handler = DelegateHandler(kit, cache=cache)
        await handler.handle(
            room_id="room-1",
            calling_agent_id="voice-agent",
            arguments={"agent": "exec-agent", "task": "do more work"},
        )

        # Check that previous_tasks was injected into context
        _, kwargs = kit.delegate.call_args
        assert "previous_tasks" in kwargs["context"]


class TestDelegateHandlerSerialization:
    async def test_serializes_per_room(self) -> None:
        """With serialize_per_room=True, delegations should be serialized."""
        execution_order: list[str] = []

        async def slow_delegate(**kwargs):
            agent_id = kwargs["agent_id"]
            execution_order.append(f"start:{agent_id}")
            await asyncio.sleep(0.05)
            execution_order.append(f"end:{agent_id}")
            task = DelegatedTask(
                id=f"t-{agent_id}",
                child_room_id=f"child-{agent_id}",
                parent_room_id=kwargs["room_id"],
                agent_id=agent_id,
                task=kwargs["task"],
            )
            task.wait = AsyncMock()
            return task

        kit = MagicMock()
        kit.delegate = AsyncMock(side_effect=slow_delegate)

        handler = DelegateHandler(kit, serialize_per_room=True)

        # Launch two concurrent delegations for the same room
        t1 = asyncio.create_task(handler.handle(
            room_id="room-1",
            calling_agent_id="voice",
            arguments={"agent": "agent-a", "task": "task a"},
        ))
        t2 = asyncio.create_task(handler.handle(
            room_id="room-1",
            calling_agent_id="voice",
            arguments={"agent": "agent-b", "task": "task b"},
        ))

        await asyncio.gather(t1, t2)

        # They should be serialized: start:a, end:a, start:b, end:b
        assert execution_order[0] == "start:agent-a"
        assert execution_order[1] == "end:agent-a"
        assert execution_order[2] == "start:agent-b"
        assert execution_order[3] == "end:agent-b"
