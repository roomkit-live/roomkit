"""Tests for Supervisor strategy parameter (sequential / parallel).

Exercises framework-controlled execution via delegate_workers tool
using MockAIProvider — no real API calls.

The deterministic runners and the supervised hub-&-spoke flow are tested
directly in tests/test_supervisor.py; this file covers the tool wiring:
injection, dedup, hook firing, error propagation, and the parallel path's
review-brief contract.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from roomkit import Agent, RoomKit, Supervisor, WebSocketChannel
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.models.event import RoomEvent
from roomkit.orchestration.strategies.supervisor import WorkerStrategy
from roomkit.providers.ai.mock import MockAIProvider
from tests.tool_room import owner_tool_call_in, owner_tool_definitions, room_tool_names

# -- Helpers ------------------------------------------------------------------


def _agent(channel_id: str, response: str) -> Agent:
    return Agent(
        channel_id,
        provider=MockAIProvider(responses=[response]),
        role=channel_id,
        memory=SlidingWindowMemory(max_events=50),
    )


async def _setup(strategy: str, workers: list[Agent]) -> tuple[RoomKit, Agent]:
    """Create a kit with Supervisor strategy and return (kit, supervisor)."""
    supervisor = _agent("boss", "ok")
    kit = RoomKit(
        orchestration=Supervisor(
            supervisor=supervisor,
            workers=workers,
            strategy=strategy,
        ),
    )
    ws = WebSocketChannel("ws")
    ws.register_connection("u", lambda _c, _e: None, room_id="room")
    kit.register_channel(ws)
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "ws")
    return kit, supervisor


# -- Strategy tool injection --------------------------------------------------


class TestStrategyToolInjection:
    async def test_strategy_injects_delegate_workers_tool(self) -> None:
        kit, supervisor = await _setup("sequential", [_agent("w1", "r1")])
        # Declared in the room it was installed in (RFC §19.7), not channel-wide.
        assert room_tool_names(kit, "boss", "room") == ["delegate_workers"]
        assert "delegate_workers" not in [t.name for t in owner_tool_definitions(kit, "boss")]
        # No per-worker tools
        assert "delegate_to_w1" not in [t.name for t in owner_tool_definitions(kit, "boss")]
        await kit.close()

    async def test_no_strategy_injects_per_worker_tools(self) -> None:
        supervisor = _agent("boss", "ok")
        kit = RoomKit(
            orchestration=Supervisor(
                supervisor=supervisor,
                workers=[_agent("w1", "r1"), _agent("w2", "r2")],
            ),
        )
        ws = WebSocketChannel("ws")
        ws.register_connection("u", lambda _c, _e: None, room_id="room")
        kit.register_channel(ws)
        await kit.create_room(room_id="room")
        await kit.attach_channel("room", "ws")

        tool_names = room_tool_names(kit, "boss", "room")
        assert tool_names == ["delegate_to_w1", "delegate_to_w2"]
        assert "delegate_workers" not in tool_names
        await kit.close()

    async def test_worker_strategy_enum_values(self) -> None:
        assert WorkerStrategy.SEQUENTIAL == "sequential"
        assert WorkerStrategy.PARALLEL == "parallel"


# -- Parallel strategy --------------------------------------------------------


class TestParallelStrategy:
    async def test_runs_all_workers_concurrently(self) -> None:
        """All workers run; the tool returns a review brief surfacing each
        worker's output for the supervisor's final summary."""
        kit, supervisor = await _setup(
            "parallel",
            [_agent("tech", "Tech analysis."), _agent("biz", "Biz analysis.")],
        )

        with owner_tool_call_in(kit, "boss", "room"):
            review = await supervisor._channel_tool_handler(
                "delegate_workers", {"task": "Analyze X"}
            )

        # Both workers ran and their outputs are surfaced in the review brief.
        assert "tech" in review
        assert "biz" in review
        assert "Tech analysis." in review
        assert "Biz analysis." in review
        await kit.close()

    async def test_parallel_all_receive_same_task(self) -> None:
        """All workers receive the same task description."""
        received_tasks: list[str] = []

        kit, supervisor = await _setup(
            "parallel",
            [_agent("w1", "r1"), _agent("w2", "r2")],
        )

        original_delegate = kit.delegate

        async def tracking_delegate(*args, **kwargs):  # type: ignore[no-untyped-def]
            received_tasks.append(args[2])
            return await original_delegate(*args, **kwargs)

        kit.delegate = tracking_delegate  # type: ignore[assignment]

        with owner_tool_call_in(kit, "boss", "room"):
            await supervisor._channel_tool_handler("delegate_workers", {"task": "Same task"})

        # Both workers should get the exact same task
        assert all(t == "Same task" for t in received_tasks)
        await kit.close()

    async def test_parallel_single_worker(self) -> None:
        kit, supervisor = await _setup("parallel", [_agent("solo", "Result.")])

        with owner_tool_call_in(kit, "boss", "room"):
            review = await supervisor._channel_tool_handler("delegate_workers", {"task": "Do it"})

        assert "Result." in review
        await kit.close()


# -- Hooks fire for strategy mode ---------------------------------------------


class TestStrategyHooks:
    async def test_hooks_fire_for_each_worker(self) -> None:
        """ON_TASK_DELEGATED and ON_TASK_COMPLETED fire per worker."""
        kit, supervisor = await _setup(
            "parallel",
            [_agent("w1", "r1"), _agent("w2", "r2")],
        )

        events: list[tuple[str, str]] = []

        @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC)
        async def on_del(event: RoomEvent, _ctx: object) -> None:
            events.append(("delegated", event.metadata.get("agent_id", "")))

        @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
        async def on_done(event: RoomEvent, _ctx: object) -> None:
            events.append(("completed", event.metadata.get("agent_id", "")))

        with owner_tool_call_in(kit, "boss", "room"):
            await supervisor._channel_tool_handler("delegate_workers", {"task": "test"})

        delegated = [e for e in events if e[0] == "delegated"]
        completed = [e for e in events if e[0] == "completed"]

        assert len(delegated) == 2
        assert len(completed) == 2
        assert {d[1] for d in delegated} == {"w1", "w2"}
        assert {c[1] for c in completed} == {"w1", "w2"}
        await kit.close()


# -- Dedup guard --------------------------------------------------------------


class TestStrategyDedup:
    async def test_second_call_in_same_turn_returns_cached(self) -> None:
        """Calling delegate_workers twice within the dedup window returns cached."""
        kit, supervisor = await _setup("parallel", [_agent("w1", "r1")])

        call_count = 0
        original_delegate = kit.delegate

        async def counting_delegate(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return await original_delegate(*args, **kwargs)

        kit.delegate = counting_delegate  # type: ignore[assignment]

        with owner_tool_call_in(kit, "boss", "room"):
            r1 = await supervisor._channel_tool_handler("delegate_workers", {"task": "test"})
        # Second call — even with different task, blocked within dedup window
        with owner_tool_call_in(kit, "boss", "room"):
            r2 = await supervisor._channel_tool_handler(
                "delegate_workers", {"task": "different task"}
            )

        assert r1 == r2
        assert call_count == 1
        await kit.close()


# -- Recursion guard ----------------------------------------------------------


class TestSubTaskRecursionGuard:
    async def test_delegate_workers_refuses_inside_subtask_room(self) -> None:
        """The supervisor owns delegate_workers, and the supervised flow re-invokes
        it for dispatch/review inside ``::task-`` child rooms. Calling the tool from
        such a room must refuse (no re-delegation) — otherwise delegate_workers
        recurses within delegate_workers."""
        kit, supervisor = await _setup("sequential", [_agent("w1", "r1")])

        call_count = 0
        original_delegate = kit.delegate

        async def counting_delegate(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return await original_delegate(*args, **kwargs)

        kit.delegate = counting_delegate  # type: ignore[assignment]

        # A call while the supervisor runs in a dispatch/review child room of the
        # supervised flow: the tool is the installed room's, not served there.
        with (
            owner_tool_call_in(kit, "boss", "room::task-abc123"),
            pytest.raises(UnservedToolCallError),
        ):
            await supervisor._channel_tool_handler("delegate_workers", {"task": "go"})

        # The pipeline never ran — no delegation happened, recursion prevented.
        assert call_count == 0
        await kit.close()


# -- Error handling -----------------------------------------------------------


class TestSequentialWorkerFailure:
    async def test_sequential_propagates_error_in_result(self) -> None:
        """If delegation raises, the handler raises on: the channel reads it as
        any failed call, its message never reaching the model (RMK-295)."""
        kit, supervisor = await _setup("sequential", [_agent("w1", "ok")])

        # Make delegate raise
        async def failing_delegate(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        kit.delegate = failing_delegate  # type: ignore[assignment]

        with owner_tool_call_in(kit, "boss", "room"), pytest.raises(RuntimeError, match="boom"):
            await supervisor._channel_tool_handler("delegate_workers", {"task": "fail"})
        await kit.close()


class TestParallelWorkerFailure:
    async def test_parallel_propagates_error(self) -> None:
        """If delegation raises during parallel, the handler raises on (RMK-295)."""
        kit, supervisor = await _setup("parallel", [_agent("w1", "ok")])

        async def failing_delegate(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("parallel boom")

        kit.delegate = failing_delegate  # type: ignore[assignment]

        with (
            owner_tool_call_in(kit, "boss", "room"),
            pytest.raises(RuntimeError, match="parallel boom"),
        ):
            await supervisor._channel_tool_handler("delegate_workers", {"task": "fail"})
        await kit.close()

    async def test_the_first_error_cuts_the_other_workers(self) -> None:
        """RMK-1279, RFC §19.7.3: the first delegation that raises answers the
        call at once and cuts the others, awaited to their end, never left
        running unread."""
        kit, supervisor = await _setup("parallel", [_agent("w1", "ok"), _agent("w2", "ok")])
        cut = asyncio.Event()

        async def delegate(room_id: str, agent_id: str, task: str, **kwargs: object) -> None:
            if agent_id == "w1":
                raise RuntimeError("parallel boom")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cut.set()
                raise

        kit.delegate = delegate  # type: ignore[assignment]

        with (
            owner_tool_call_in(kit, "boss", "room"),
            pytest.raises(RuntimeError, match="parallel boom"),
        ):
            await supervisor._channel_tool_handler("delegate_workers", {"task": "fail"})
        assert cut.is_set()
        await kit.close()


# -- Background dedup (pending set) ------------------------------------------


class TestBackgroundDedup:
    async def test_pending_guard_blocks_re_delegation(self) -> None:
        """In manual mode with wait_for_result=False, re-calling the same
        worker returns 'already_running' instead of delegating again."""
        supervisor = _agent("boss", "ok")
        worker = _agent("w1", "result")

        kit = RoomKit(
            orchestration=Supervisor(
                supervisor=supervisor,
                workers=[worker],
                wait_for_result=False,
            ),
        )
        ws = WebSocketChannel("ws")
        ws.register_connection("u", lambda _c, _e: None, room_id="room")
        kit.register_channel(ws)
        await kit.create_room(room_id="room")
        await kit.attach_channel("room", "ws")

        # First call — delegates
        with owner_tool_call_in(kit, "boss", "room"):
            r1 = json.loads(
                await supervisor._channel_tool_handler("delegate_to_w1", {"task": "first"})
            )
        assert r1["status"] == "delegated"

        # Second call — blocked by pending guard
        with owner_tool_call_in(kit, "boss", "room"):
            r2 = json.loads(
                await supervisor._channel_tool_handler("delegate_to_w1", {"task": "second"})
            )
        assert r2["status"] == "already_running"

        await kit.close()

    async def test_pending_clears_after_completion(self) -> None:
        """After a background task completes, the worker can be delegated again."""
        supervisor = _agent("boss", "ok")
        worker = _agent("w1", "result")

        kit = RoomKit(
            orchestration=Supervisor(
                supervisor=supervisor,
                workers=[worker],
                wait_for_result=False,
            ),
        )
        ws = WebSocketChannel("ws")
        ws.register_connection("u", lambda _c, _e: None, room_id="room")
        kit.register_channel(ws)
        await kit.create_room(room_id="room")
        await kit.attach_channel("room", "ws")

        # First delegation
        with owner_tool_call_in(kit, "boss", "room"):
            r1 = json.loads(
                await supervisor._channel_tool_handler("delegate_to_w1", {"task": "first"})
            )
        assert r1["status"] == "delegated"

        # The worker is freed before its outcome is handed back to the supervisor.
        async with asyncio.timeout(5.0):
            while not supervisor.provider.calls:
                await asyncio.sleep(0.01)

        # Now should be able to delegate again
        with owner_tool_call_in(kit, "boss", "room"):
            r2 = json.loads(
                await supervisor._channel_tool_handler("delegate_to_w1", {"task": "second"})
            )
        assert r2["status"] == "delegated"

        await kit.close()
