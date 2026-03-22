"""Tests for Supervisor strategy parameter (sequential / parallel).

Exercises framework-controlled execution via delegate_workers tool
using MockAIProvider — no real API calls.
"""

from __future__ import annotations

import json

from roomkit import Agent, RoomKit, Supervisor, WebSocketChannel
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.enums import HookExecution, HookTrigger
from roomkit.models.event import RoomEvent
from roomkit.orchestration.strategies.supervisor import WorkerStrategy
from roomkit.providers.ai.mock import MockAIProvider

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
    ws.register_connection("u", lambda _c, _e: None)
    kit.register_channel(ws)
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "ws")
    return kit, supervisor


# -- Strategy tool injection --------------------------------------------------


class TestStrategyToolInjection:
    async def test_strategy_injects_delegate_workers_tool(self) -> None:
        kit, supervisor = await _setup("sequential", [_agent("w1", "r1")])
        tool_names = [t.name for t in supervisor._injected_tools]
        assert "delegate_workers" in tool_names
        # No per-worker tools
        assert "delegate_to_w1" not in tool_names
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
        ws.register_connection("u", lambda _c, _e: None)
        kit.register_channel(ws)
        await kit.create_room(room_id="room")
        await kit.attach_channel("room", "ws")

        tool_names = [t.name for t in supervisor._injected_tools]
        assert "delegate_to_w1" in tool_names
        assert "delegate_to_w2" in tool_names
        assert "delegate_workers" not in tool_names
        await kit.close()

    async def test_worker_strategy_enum_values(self) -> None:
        assert WorkerStrategy.SEQUENTIAL == "sequential"
        assert WorkerStrategy.PARALLEL == "parallel"


# -- Sequential strategy ------------------------------------------------------


class TestSequentialStrategy:
    async def test_runs_workers_in_order(self) -> None:
        """Workers execute sequentially, each receiving previous output."""
        kit, supervisor = await _setup(
            "sequential",
            [_agent("researcher", "Research findings."), _agent("writer", "Final article.")],
        )

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        result = json.loads(
            await supervisor.tool_handler("delegate_workers", {"task": "Write about AI"})
        )

        assert result["status"] == "completed"
        assert len(result["results"]) == 2
        assert result["results"][0]["worker"] == "researcher"
        assert "Research findings." in result["results"][0]["output"]
        assert result["results"][1]["worker"] == "writer"
        assert "Final article." in result["results"][1]["output"]
        await kit.close()

    async def test_sequential_chains_output(self) -> None:
        """Second worker receives first worker's output as its task."""
        received_tasks: list[str] = []
        original_delegate = None

        kit, supervisor = await _setup(
            "sequential",
            [_agent("step1", "Step 1 output."), _agent("step2", "Step 2 output.")],
        )

        # Wrap delegate to capture task descriptions
        original_delegate = kit.delegate

        async def tracking_delegate(*args, **kwargs):  # type: ignore[no-untyped-def]
            received_tasks.append(args[2])  # task is 3rd positional arg
            return await original_delegate(*args, **kwargs)

        kit.delegate = tracking_delegate  # type: ignore[assignment]

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        await supervisor.tool_handler("delegate_workers", {"task": "Initial task"})

        # First worker gets the user's task
        assert received_tasks[0] == "Initial task"
        # Second worker gets first worker's output
        assert received_tasks[1] == "Step 1 output."
        await kit.close()

    async def test_sequential_single_worker(self) -> None:
        kit, supervisor = await _setup("sequential", [_agent("solo", "Done.")])

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        result = json.loads(await supervisor.tool_handler("delegate_workers", {"task": "Do it"}))

        assert result["status"] == "completed"
        assert len(result["results"]) == 1
        assert "Done." in result["results"][0]["output"]
        await kit.close()


# -- Parallel strategy --------------------------------------------------------


class TestParallelStrategy:
    async def test_runs_all_workers_concurrently(self) -> None:
        """All workers run and results are collected."""
        kit, supervisor = await _setup(
            "parallel",
            [_agent("tech", "Tech analysis."), _agent("biz", "Biz analysis.")],
        )

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        result = json.loads(
            await supervisor.tool_handler("delegate_workers", {"task": "Analyze X"})
        )

        assert result["status"] == "completed"
        assert len(result["results"]) == 2
        workers = {r["worker"] for r in result["results"]}
        assert workers == {"tech", "biz"}
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

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        await supervisor.tool_handler("delegate_workers", {"task": "Same task"})

        # Both workers should get the exact same task
        assert all(t == "Same task" for t in received_tasks)
        await kit.close()

    async def test_parallel_single_worker(self) -> None:
        kit, supervisor = await _setup("parallel", [_agent("solo", "Result.")])

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        result = json.loads(await supervisor.tool_handler("delegate_workers", {"task": "Do it"}))

        assert result["status"] == "completed"
        assert len(result["results"]) == 1
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

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        await supervisor.tool_handler("delegate_workers", {"task": "test"})

        delegated = [e for e in events if e[0] == "delegated"]
        completed = [e for e in events if e[0] == "completed"]

        assert len(delegated) == 2
        assert len(completed) == 2
        assert {d[1] for d in delegated} == {"w1", "w2"}
        assert {c[1] for c in completed} == {"w1", "w2"}
        await kit.close()


# -- Dedup guard --------------------------------------------------------------


class TestStrategyDedup:
    async def test_same_task_returns_cached_result(self) -> None:
        """Calling delegate_workers twice with the same task dedupes."""
        kit, supervisor = await _setup("parallel", [_agent("w1", "r1")])

        call_count = 0
        original_delegate = kit.delegate

        async def counting_delegate(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return await original_delegate(*args, **kwargs)

        kit.delegate = counting_delegate  # type: ignore[assignment]

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")

        r1 = await supervisor.tool_handler("delegate_workers", {"task": "test"})
        r2 = await supervisor.tool_handler("delegate_workers", {"task": "test"})

        # Same task — second call returns cached
        assert r1 == r2
        assert call_count == 1
        await kit.close()

    async def test_different_task_runs_fresh(self) -> None:
        """Calling delegate_workers with a different task runs workers again."""
        kit, supervisor = await _setup("parallel", [_agent("w1", "r1")])

        call_count = 0
        original_delegate = kit.delegate

        async def counting_delegate(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return await original_delegate(*args, **kwargs)

        kit.delegate = counting_delegate  # type: ignore[assignment]

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")

        r1 = await supervisor.tool_handler("delegate_workers", {"task": "analyze anthropic"})
        r2 = await supervisor.tool_handler("delegate_workers", {"task": "analyze openai"})

        # Different tasks — both run fresh
        assert r1 != r2 or call_count == 2  # results may match (mock), but both ran
        assert call_count == 2
        await kit.close()


# -- Error handling -----------------------------------------------------------


class TestSequentialWorkerFailure:
    async def test_sequential_continues_on_empty_output(self) -> None:
        """If a worker returns empty, the chain continues with empty input."""
        kit, supervisor = await _setup(
            "sequential",
            # First worker returns empty string — second still runs
            [_agent("step1", ""), _agent("step2", "Step 2 done.")],
        )

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        result = json.loads(await supervisor.tool_handler("delegate_workers", {"task": "Go"}))

        assert result["status"] == "completed"
        assert len(result["results"]) == 2
        # Second worker still ran
        assert result["results"][1]["worker"] == "step2"
        await kit.close()

    async def test_sequential_propagates_error_in_result(self) -> None:
        """If delegation raises, the strategy handler returns an error JSON."""
        kit, supervisor = await _setup("sequential", [_agent("w1", "ok")])

        # Make delegate raise
        async def failing_delegate(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        kit.delegate = failing_delegate  # type: ignore[assignment]

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        result = json.loads(await supervisor.tool_handler("delegate_workers", {"task": "fail"}))

        assert "error" in result
        assert "boom" in result["error"]
        await kit.close()


class TestParallelWorkerFailure:
    async def test_parallel_propagates_error(self) -> None:
        """If delegation raises during parallel, error is returned."""
        kit, supervisor = await _setup("parallel", [_agent("w1", "ok")])

        async def failing_delegate(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("parallel boom")

        kit.delegate = failing_delegate  # type: ignore[assignment]

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")
        result = json.loads(await supervisor.tool_handler("delegate_workers", {"task": "fail"}))

        assert "error" in result
        assert "parallel boom" in result["error"]
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
        ws.register_connection("u", lambda _c, _e: None)
        kit.register_channel(ws)
        await kit.create_room(room_id="room")
        await kit.attach_channel("room", "ws")

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")

        # First call — delegates
        r1 = json.loads(await supervisor.tool_handler("delegate_to_w1", {"task": "first"}))
        assert r1["status"] == "delegated"

        # Second call — blocked by pending guard
        r2 = json.loads(await supervisor.tool_handler("delegate_to_w1", {"task": "second"}))
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
        ws.register_connection("u", lambda _c, _e: None)
        kit.register_channel(ws)
        await kit.create_room(room_id="room")
        await kit.attach_channel("room", "ws")

        from roomkit.orchestration.handoff import _room_id_var

        _room_id_var.set("room")

        # First delegation
        r1 = json.loads(await supervisor.tool_handler("delegate_to_w1", {"task": "first"}))
        assert r1["status"] == "delegated"

        # Wait for background task to complete
        import asyncio

        await asyncio.sleep(0.1)

        # Now should be able to delegate again
        r2 = json.loads(await supervisor.tool_handler("delegate_to_w1", {"task": "second"}))
        assert r2["status"] == "delegated"

        await kit.close()
