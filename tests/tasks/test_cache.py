"""Tests for CompletedTaskCache and delegation state tracking."""

from __future__ import annotations

import asyncio
import time
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

from roomkit import RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.cache import CompletedTaskCache
from roomkit.tasks.delegate import DelegateHandler
from roomkit.tasks.models import DelegatedTask
from tests.conference.test_conference_realtime import until
from tests.test_framework import SimpleChannel


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

    def test_the_context_is_part_of_the_key(self) -> None:
        cache = CompletedTaskCache()
        cache.put(
            "room-1", "agent-a", "task", {"n": 1}, context=MappingProxyType({"x": 1, "y": 2})
        )

        assert cache.get("room-1", "agent-a", "task") is None
        assert cache.get("room-1", "agent-a", "task", context={"x": 2, "y": 2}) is None
        assert cache.get("room-1", "agent-a", "task", context={"y": 2, "x": 1}) == {"n": 1}

    def test_a_context_never_reads_as_part_of_the_task(self) -> None:
        cache = CompletedTaskCache()
        cache.put("room-1", "agent-a", "a", {"n": 1}, context={"x": 1})

        assert cache.get("room-1", "agent-a", 'a\n{"x": 1}') is None

    def test_the_host_reads_what_an_agent_s_call_kept(self) -> None:
        cache = CompletedTaskCache()
        cache._keep("room-1", "agent-a", "task", {"n": 1}, None, "boss", answers=True)

        assert cache.get("room-1", "agent-a", "task") == {"n": 1}
        assert cache._answer_for("room-1", "agent-a", "task", None, "boss") == {"n": 1}
        assert cache._answer_for("room-1", "agent-a", "task", None, "other") is None

    def test_a_task_the_host_wrote_answers_any_agent(self) -> None:
        cache = CompletedTaskCache()
        cache.put("room-1", "agent-a", "task", {"n": 1})

        assert cache._answer_for("room-1", "agent-a", "task", None, "boss") == {"n": 1}
        assert cache._recent_of("room-1", "boss", limit=3) == ["task"]

    def test_a_kept_task_that_answers_no_repeat_is_read_by_no_get(self) -> None:
        cache = CompletedTaskCache()
        cache._keep("room-1", "agent-a", "task", {}, None, "boss", answers=False)

        assert cache.get("room-1", "agent-a", "task") is None
        assert cache._answer_for("room-1", "agent-a", "task", None, "boss") is None
        assert cache.recent_context("room-1") == ["task"]

    def test_the_latest_of_what_is_kept_answers(self) -> None:
        cache = CompletedTaskCache()
        cache._keep("room-1", "agent-a", "task", {"n": "first"}, None, "boss", answers=True)
        cache._keep("room-1", "agent-a", "task", {"n": "second"}, None, "other", answers=True)
        cache.put("room-1", "agent-a", "task", {"n": "the host's"})

        assert cache.get("room-1", "agent-a", "task") == {"n": "the host's"}
        assert cache._answer_for("room-1", "agent-a", "task", None, "boss") == {"n": "the host's"}

        cache._keep("room-1", "agent-a", "task", {"n": "third"}, None, "boss", answers=True)
        assert cache._answer_for("room-1", "agent-a", "task", None, "boss") == {"n": "third"}

    def test_recent_context_lists_a_task_once(self) -> None:
        cache = CompletedTaskCache()
        cache.put("room-1", "agent-a", "Other.", {})
        cache.put("room-1", "agent-a", "Fix.", {}, context={"x": 1})
        cache.put("room-1", "agent-a", " fix. ", {}, context={"x": 2})

        assert cache.recent_context("room-1", limit=2) == [" fix. ", "Other."]

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
        cache.put(
            "room-1",
            "exec-agent",
            "search cats",
            {
                "status": "delegated",
                "task_id": "t-old",
                "child_room_id": "child-old",
                "agent_id": "exec-agent",
            },
        )

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

    async def test_the_worker_s_model_reads_the_previous_tasks(self) -> None:
        """What a cached task asked reaches the worker's model, quoted, and the
        task's own text stays as written."""
        worker = MockAIProvider(responses=["Done."])
        kit = RoomKit()
        kit.register_channel(SimpleChannel("sms"))
        kit.register_channel(AIChannel("worker", provider=worker))
        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "sms")

        cache = CompletedTaskCache()
        cache.put("room-1", "other-agent", "previous task", {"done": True})
        handler = DelegateHandler(kit, cache=cache)
        await handler.handle(
            room_id="room-1",
            calling_agent_id="voice-agent",
            arguments={"agent": "worker", "task": "do more work"},
        )
        await until(lambda: bool(worker.calls))
        await kit.close()

        read = worker.calls[0].messages[-1].content
        assert read.startswith("do more work\n\n")
        assert '<previous_tasks>\n"previous task"\n</previous_tasks>' in read


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
        t1 = asyncio.create_task(
            handler.handle(
                room_id="room-1",
                calling_agent_id="voice",
                arguments={"agent": "agent-a", "task": "task a"},
            )
        )
        t2 = asyncio.create_task(
            handler.handle(
                room_id="room-1",
                calling_agent_id="voice",
                arguments={"agent": "agent-b", "task": "task b"},
            )
        )

        await asyncio.gather(t1, t2)

        # They should be serialized: start:a, end:a, start:b, end:b
        assert execution_order[0] == "start:agent-a"
        assert execution_order[1] == "end:agent-a"
        assert execution_order[2] == "start:agent-b"
        assert execution_order[3] == "end:agent-b"
        # The last delegation out drops the room's lock.
        assert handler._room_locks == {}

    async def test_a_room_that_stopped_delegating_keeps_no_lock(self) -> None:
        async def delegate(**kwargs):
            task = DelegatedTask(
                id=f"t-{kwargs['room_id']}",
                child_room_id="child",
                parent_room_id=kwargs["room_id"],
                agent_id=kwargs["agent_id"],
                task=kwargs["task"],
            )
            task.wait = AsyncMock()
            return task

        kit = MagicMock()
        kit.delegate = AsyncMock(side_effect=delegate)
        handler = DelegateHandler(kit, serialize_per_room=True)

        for room_id in ("room-1", "room-2", "room-3"):
            await handler.handle(
                room_id=room_id,
                calling_agent_id="voice",
                arguments={"agent": "agent-a", "task": "task a"},
            )

        assert handler._room_locks == {}
