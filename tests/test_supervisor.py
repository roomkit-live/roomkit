"""Tests for Supervisor orchestration (orchestration/strategies/supervisor.py).

Covers: install wiring, _run_workers, _two_pass_delegate, _one_pass_delegate,
_format_worker_results, _extract_output_text, strategy tool handler,
per-worker delegation (wait/no-wait), and async delivery helpers.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.orchestration.state import get_conversation_state
from roomkit.orchestration.strategies.supervisor import (
    Supervisor,
    WorkerStrategy,
    _async_run_and_deliver,
    _extract_output_text,
    _format_worker_results,
    _one_pass_delegate,
    _run_parallel,
    _run_sequential,
    _run_workers,
    _two_pass_delegate,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.models import DelegatedTaskResult

# -- Helpers ------------------------------------------------------------------


class _NoopLock:
    async def __aenter__(self) -> None:
        pass

    async def __aexit__(self, *args: object) -> None:
        pass


def _make_agent(
    channel_id: str,
    responses: list[str] | None = None,
    role: str | None = None,
    description: str | None = None,
) -> Agent:
    return Agent(
        channel_id=channel_id,
        provider=MockAIProvider(responses=responses or ["ok"]),
        role=role,
        description=description,
    )


def _make_mock_kit(room: Room) -> MagicMock:
    kit = MagicMock()
    kit.get_room = AsyncMock(return_value=room)
    kit.store.update_room = AsyncMock()
    kit.hook_engine = MagicMock()
    kit.hook_engine.add_room_hook = MagicMock()
    kit.lock_manager = MagicMock()
    kit.lock_manager.locked = MagicMock(return_value=_NoopLock())
    kit.channels = {}
    kit.register_channel = MagicMock()
    kit.deliver = AsyncMock()
    return kit


def _delegated_task_with_output(output: str) -> MagicMock:
    task = MagicMock()
    task.result = DelegatedTaskResult(
        task_id="t1",
        child_room_id="child-1",
        parent_room_id="r1",
        agent_id="agent-1",
        output=output,
    )
    return task


def _delegated_task_with_error(error: str) -> MagicMock:
    task = MagicMock()
    task.result = DelegatedTaskResult(
        task_id="t1",
        child_room_id="child-1",
        parent_room_id="r1",
        agent_id="agent-1",
        output=None,
        error=error,
    )
    return task


def _delegated_task_empty() -> MagicMock:
    task = MagicMock()
    task.result = None
    return task


def _make_event(
    body: str = "Hello",
    channel_id: str = "user-ch",
    channel_type: ChannelType = ChannelType.SMS,
    room_id: str = "r1",
) -> RoomEvent:
    return RoomEvent(
        room_id=room_id,
        source=EventSource(channel_id=channel_id, channel_type=channel_type),
        content=TextContent(body=body),
    )


def _make_binding(room_id: str = "r1") -> ChannelBinding:
    return ChannelBinding(
        channel_id="user-ch",
        room_id=room_id,
        channel_type=ChannelType.SMS,
    )


def _make_context(room_id: str = "r1") -> RoomContext:
    return RoomContext(room=Room(id=room_id))


# -- Tests: Constructor -------------------------------------------------------


class TestSupervisorConstructor:
    def test_constructor_defaults(self) -> None:
        supervisor = _make_agent("sup", role="supervisor")
        workers = [_make_agent("w1"), _make_agent("w2")]
        s = Supervisor(supervisor=supervisor, workers=workers)
        assert s._supervisor is supervisor
        assert len(s._workers) == 2
        assert s._strategy is None
        assert s._auto_delegate is False

    def test_constructor_with_strategy(self) -> None:
        supervisor = _make_agent("sup")
        workers = [_make_agent("w1")]
        s = Supervisor(supervisor=supervisor, workers=workers, strategy="sequential")
        assert s._strategy == WorkerStrategy.SEQUENTIAL

    def test_constructor_parallel_strategy(self) -> None:
        supervisor = _make_agent("sup")
        workers = [_make_agent("w1")]
        s = Supervisor(supervisor=supervisor, workers=workers, strategy="parallel")
        assert s._strategy == WorkerStrategy.PARALLEL

    def test_auto_delegate_without_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="auto_delegate"):
            Supervisor(
                supervisor=_make_agent("sup"),
                workers=[_make_agent("w1")],
                auto_delegate=True,
            )

    def test_agents_returns_supervisor(self) -> None:
        supervisor = _make_agent("sup")
        s = Supervisor(supervisor=supervisor, workers=[_make_agent("w1")])
        result = s.agents()
        assert len(result) == 1
        assert result[0] is supervisor

    def test_agents_empty_when_async_delivery(self) -> None:
        s = Supervisor(
            supervisor=_make_agent("sup"),
            workers=[_make_agent("w1")],
            strategy="parallel",
            async_delivery=True,
        )
        assert s.agents() == []

    def test_worker_strategy_enum_values(self) -> None:
        assert WorkerStrategy.SEQUENTIAL == "sequential"
        assert WorkerStrategy.PARALLEL == "parallel"


# -- Tests: Install -----------------------------------------------------------


class TestSupervisorInstall:
    async def test_installs_router_hook(self) -> None:
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=[_make_agent("w1")])
        await s.install(kit, "r1")

        kit.hook_engine.add_room_hook.assert_called_once()

    async def test_registers_workers_on_kit(self) -> None:
        boss = _make_agent("boss")
        workers = [_make_agent("w1"), _make_agent("w2")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")
        assert kit.register_channel.call_count == 2

    async def test_skips_already_registered_workers(self) -> None:
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))
        kit.channels = {"w1": w1}

        s = Supervisor(supervisor=boss, workers=[w1])
        await s.install(kit, "r1")
        kit.register_channel.assert_not_called()

    async def test_sets_initial_state(self) -> None:
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=[_make_agent("w1")])
        await s.install(kit, "r1")

        updated_room = kit.store.update_room.call_args[0][0]
        state = get_conversation_state(updated_room)
        assert state.active_agent_id == "boss"
        assert state.phase == "supervisor"

    async def test_install_auto_delegate_wraps_on_event(self) -> None:
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))

        original = boss.on_event
        s = Supervisor(
            supervisor=boss,
            workers=[_make_agent("w1")],
            strategy="sequential",
            auto_delegate=True,
        )
        await s.install(kit, "r1")

        assert boss.on_event is not original

    async def test_install_strategy_tool_injects_delegate_workers(self) -> None:
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(
            supervisor=boss,
            workers=[_make_agent("w1")],
            strategy="sequential",
        )
        await s.install(kit, "r1")

        tool_names = [t.name for t in boss._injected_tools]
        assert "delegate_workers" in tool_names

    async def test_install_per_worker_tools(self) -> None:
        boss = _make_agent("boss")
        workers = [_make_agent("w1", description="Worker 1"), _make_agent("w2")]
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")

        tool_names = [t.name for t in boss._injected_tools]
        assert "delegate_to_w1" in tool_names
        assert "delegate_to_w2" in tool_names

    async def test_double_install_no_duplicate_tools(self) -> None:
        boss = _make_agent("boss")
        workers = [_make_agent("w1")]
        kit = _make_mock_kit(Room(id="r1"))
        mock_task = MagicMock()
        mock_task.task_id = "t1"
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(supervisor=boss, workers=workers)
        await s.install(kit, "r1")
        kit2 = _make_mock_kit(Room(id="r2"))
        kit2.delegate = AsyncMock(return_value=mock_task)
        await s.install(kit2, "r2")

        tool_count = sum(1 for t in boss._injected_tools if t.name == "delegate_to_w1")
        assert tool_count == 1

    async def test_double_install_strategy_tool_no_duplicate(self) -> None:
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(
            supervisor=boss,
            workers=[_make_agent("w1")],
            strategy="sequential",
        )
        await s.install(kit, "r1")
        kit2 = _make_mock_kit(Room(id="r2"))
        await s.install(kit2, "r2")

        tool_count = sum(1 for t in boss._injected_tools if t.name == "delegate_workers")
        assert tool_count == 1

    async def test_no_router_hook_when_async_delivery(self) -> None:
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(
            supervisor=boss,
            workers=[_make_agent("w1")],
            strategy="parallel",
            async_delivery=True,
        )
        await s.install(kit, "r1")

        kit.hook_engine.add_room_hook.assert_not_called()


# -- Tests: Per-worker delegation tool handler --------------------------------


class TestPerWorkerDelegation:
    async def test_wait_for_result_returns_output(self) -> None:
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("Worker result"))

        s = Supervisor(supervisor=boss, workers=[w1], wait_for_result=True)
        await s.install(kit, "r1")

        result = await boss.tool_handler("delegate_to_w1", {"task": "Do work"})
        parsed = json.loads(result)
        assert parsed["result"] == "Worker result"
        assert parsed["worker"] == "w1"

    async def test_wait_for_result_with_error(self) -> None:
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(return_value=_delegated_task_with_error("Something went wrong"))

        s = Supervisor(supervisor=boss, workers=[w1], wait_for_result=True)
        await s.install(kit, "r1")

        result = await boss.tool_handler("delegate_to_w1", {"task": "Do work"})
        parsed = json.loads(result)
        assert parsed["result"] == "Something went wrong"

    async def test_wait_for_result_none(self) -> None:
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(return_value=_delegated_task_empty())

        s = Supervisor(supervisor=boss, workers=[w1], wait_for_result=True)
        await s.install(kit, "r1")

        result = await boss.tool_handler("delegate_to_w1", {"task": "Do work"})
        parsed = json.loads(result)
        assert parsed["status"] == "failed"

    async def test_no_wait_delegates_background(self) -> None:
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))
        mock_task = MagicMock()
        mock_task.id = "task-123"
        mock_task._set_result = MagicMock()
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(supervisor=boss, workers=[w1], wait_for_result=False)
        await s.install(kit, "r1")

        result = await boss.tool_handler("delegate_to_w1", {"task": "Do work"})
        parsed = json.loads(result)
        assert parsed["status"] == "delegated"
        assert parsed["task_id"] == "task-123"

    async def test_no_wait_already_running(self) -> None:
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))
        mock_task = MagicMock()
        mock_task.id = "task-123"
        mock_task._set_result = MagicMock()
        kit.delegate = AsyncMock(return_value=mock_task)

        s = Supervisor(supervisor=boss, workers=[w1], wait_for_result=False)
        await s.install(kit, "r1")

        # First call succeeds
        await boss.tool_handler("delegate_to_w1", {"task": "Do work"})

        # Second call should detect already running
        result = await boss.tool_handler("delegate_to_w1", {"task": "Do more"})
        parsed = json.loads(result)
        assert parsed["status"] == "already_running"

    async def test_delegation_exception_returns_error(self) -> None:
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(side_effect=RuntimeError("Connection lost"))

        s = Supervisor(supervisor=boss, workers=[w1], wait_for_result=True)
        await s.install(kit, "r1")

        result = await boss.tool_handler("delegate_to_w1", {"task": "Do work"})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Connection lost" in parsed["error"]

    async def test_unknown_tool_falls_through(self) -> None:
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(supervisor=boss, workers=[_make_agent("w1")])
        await s.install(kit, "r1")

        result = await boss.tool_handler("unknown_tool", {})
        parsed = json.loads(result)
        assert "error" in parsed


# -- Tests: Strategy tool handler ---------------------------------------------


class TestStrategyToolHandler:
    async def test_sequential_strategy_tool(self) -> None:
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("result"))

        s = Supervisor(
            supervisor=boss,
            workers=[w1],
            strategy="sequential",
        )
        await s.install(kit, "r1")

        result = await boss.tool_handler("delegate_workers", {"task": "analyze"})
        parsed = json.loads(result)
        assert parsed["status"] == "completed"

    async def test_parallel_strategy_tool(self) -> None:
        boss = _make_agent("boss")
        w1 = _make_agent("w1")
        w2 = _make_agent("w2")
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(
            side_effect=[
                _delegated_task_with_output("result 1"),
                _delegated_task_with_output("result 2"),
            ]
        )

        s = Supervisor(
            supervisor=boss,
            workers=[w1, w2],
            strategy="parallel",
        )
        await s.install(kit, "r1")

        result = await boss.tool_handler("delegate_workers", {"task": "analyze"})
        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        assert len(parsed["results"]) == 2

    async def test_strategy_tool_unknown_falls_through(self) -> None:
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(
            supervisor=boss,
            workers=[_make_agent("w1")],
            strategy="sequential",
        )
        await s.install(kit, "r1")

        result = await boss.tool_handler("unknown_tool", {})
        parsed = json.loads(result)
        assert "error" in parsed

    async def test_strategy_tool_exception_returns_error(self) -> None:
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(side_effect=RuntimeError("Boom"))

        s = Supervisor(
            supervisor=boss,
            workers=[_make_agent("w1")],
            strategy="sequential",
        )
        await s.install(kit, "r1")

        result = await boss.tool_handler("delegate_workers", {"task": "x"})
        parsed = json.loads(result)
        assert "error" in parsed

    async def test_strategy_tool_dedup_cache(self) -> None:
        """Second call within dedup window returns cached result."""
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("result"))

        s = Supervisor(
            supervisor=boss,
            workers=[_make_agent("w1")],
            strategy="sequential",
        )
        await s.install(kit, "r1")

        result1 = await boss.tool_handler("delegate_workers", {"task": "analyze"})
        result2 = await boss.tool_handler("delegate_workers", {"task": "analyze again"})

        # Both should return the same cached result
        assert result1 == result2
        # delegate should only be called once
        assert kit.delegate.call_count == 1


# -- Tests: _run_workers ------------------------------------------------------


class TestRunWorkers:
    async def test_sequential(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        w1 = _make_agent("w1")
        w2 = _make_agent("w2")
        kit.delegate = AsyncMock(
            side_effect=[
                _delegated_task_with_output("output 1"),
                _delegated_task_with_output("output 2"),
            ]
        )

        results = await _run_workers(kit, "r1", WorkerStrategy.SEQUENTIAL, [w1, w2], "task")
        assert len(results) == 2
        assert results[0]["output"] == "output 1"
        assert results[1]["output"] == "output 2"

    async def test_parallel(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        w1 = _make_agent("w1")
        w2 = _make_agent("w2")
        kit.delegate = AsyncMock(
            side_effect=[
                _delegated_task_with_output("output 1"),
                _delegated_task_with_output("output 2"),
            ]
        )

        results = await _run_workers(kit, "r1", WorkerStrategy.PARALLEL, [w1, w2], "task")
        assert len(results) == 2

    async def test_none_strategy_uses_parallel(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        w1 = _make_agent("w1")
        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("output"))

        results = await _run_workers(kit, "r1", None, [w1], "task")
        assert len(results) == 1


# -- Tests: _run_sequential and _run_parallel ----------------------------------


class TestRunSequential:
    async def test_chains_output(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        w1 = _make_agent("w1")
        w2 = _make_agent("w2")
        kit.delegate = AsyncMock(
            side_effect=[
                _delegated_task_with_output("first output"),
                _delegated_task_with_output("second output"),
            ]
        )

        result = await _run_sequential(kit, "r1", [w1, w2], "initial task")
        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        assert len(parsed["results"]) == 2

        # Second worker should receive first worker's output as input
        second_call = kit.delegate.call_args_list[1]
        assert second_call[0][2] == "first output"

    async def test_empty_result(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        w1 = _make_agent("w1")
        kit.delegate = AsyncMock(return_value=_delegated_task_empty())

        result = await _run_sequential(kit, "r1", [w1], "task")
        parsed = json.loads(result)
        assert parsed["results"][0]["output"] == ""

    async def test_error_result(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        w1 = _make_agent("w1")
        kit.delegate = AsyncMock(return_value=_delegated_task_with_error("Worker error"))

        result = await _run_sequential(kit, "r1", [w1], "task")
        parsed = json.loads(result)
        assert parsed["results"][0]["output"] == "Worker error"


class TestRunParallel:
    async def test_concurrent_workers(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        w1 = _make_agent("w1")
        w2 = _make_agent("w2")
        kit.delegate = AsyncMock(
            side_effect=[
                _delegated_task_with_output("output A"),
                _delegated_task_with_output("output B"),
            ]
        )

        result = await _run_parallel(kit, "r1", [w1, w2], "task")
        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        assert len(parsed["results"]) == 2


# -- Tests: _extract_output_text ---------------------------------------------


class TestExtractOutputText:
    async def test_from_response_events(self) -> None:
        event = _make_event(body="Response text")
        output = ChannelOutput(responded=True, response_events=[event])

        text = await _extract_output_text(output)
        assert text == "Response text"

    async def test_from_stream(self) -> None:
        async def _stream() -> Any:
            yield "chunk1"
            yield "chunk2"

        output = ChannelOutput(responded=True, response_stream=_stream())

        text = await _extract_output_text(output)
        assert text == "chunk1chunk2"

    async def test_empty_output(self) -> None:
        output = ChannelOutput()
        text = await _extract_output_text(output)
        assert text == ""

    async def test_non_text_events_skipped(self) -> None:
        from roomkit.models.event import MediaContent

        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="ch", channel_type=ChannelType.AI),
            content=MediaContent(url="https://example.com/img.png", mime_type="image/png"),
        )
        output = ChannelOutput(responded=True, response_events=[event])

        text = await _extract_output_text(output)
        assert text == ""


# -- Tests: _format_worker_results -------------------------------------------


class TestFormatWorkerResults:
    def test_formats_results(self) -> None:
        results = [
            {"worker": "w1", "output": "Analysis complete"},
            {"worker": "w2", "output": "Report ready"},
        ]
        text = _format_worker_results(results)
        assert "--- w1 ---" in text
        assert "Analysis complete" in text
        assert "--- w2 ---" in text
        assert "Report ready" in text

    def test_empty_results(self) -> None:
        text = _format_worker_results([])
        assert text == ""

    def test_missing_keys_use_defaults(self) -> None:
        results = [{}]
        text = _format_worker_results(results)
        assert "--- unknown ---" in text


# -- Tests: _two_pass_delegate -----------------------------------------------


class TestTwoPassDelegate:
    async def test_two_pass_runs_workers(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        supervisor = _make_agent("boss", responses=["Anthropic"])
        w1 = _make_agent("w1")

        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("Worker analysis"))

        event = _make_event(body="Analyse anthropic")
        binding = _make_binding()
        context = _make_context()

        result = await _two_pass_delegate(
            kit,
            "r1",
            supervisor,
            supervisor.on_event,
            event,
            binding,
            context,
            WorkerStrategy.SEQUENTIAL,
            [w1],
        )

        # Should have called delegate for the worker
        assert kit.delegate.called
        assert isinstance(result, ChannelOutput)

    async def test_two_pass_custom_instruction(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        supervisor = _make_agent("boss", responses=["custom topic"])
        w1 = _make_agent("w1")

        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("Result"))

        event = _make_event(body="Do the thing")
        result = await _two_pass_delegate(
            kit,
            "r1",
            supervisor,
            supervisor.on_event,
            event,
            _make_binding(),
            _make_context(),
            WorkerStrategy.SEQUENTIAL,
            [w1],
            instruction="Extract the main topic only.",
        )

        assert isinstance(result, ChannelOutput)

    async def test_two_pass_restores_prompt_on_error(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        supervisor = _make_agent("boss", responses=["topic"])
        original_prompt = supervisor._system_prompt

        async def failing_on_event(*args: Any, **kwargs: Any) -> ChannelOutput:
            raise RuntimeError("AI failed")

        event = _make_event(body="test")

        with pytest.raises(RuntimeError):
            await _two_pass_delegate(
                kit,
                "r1",
                supervisor,
                failing_on_event,
                event,
                _make_binding(),
                _make_context(),
                WorkerStrategy.SEQUENTIAL,
                [_make_agent("w1")],
            )

        # Prompt should be restored even after error
        assert supervisor._system_prompt == original_prompt


# -- Tests: _one_pass_delegate -----------------------------------------------


class TestOnePassDelegate:
    async def test_one_pass_runs_workers_with_raw_message(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        supervisor = _make_agent("boss", responses=["Presentation"])
        w1 = _make_agent("w1")

        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("Worker output"))

        event = _make_event(body="Research quantum computing")
        result = await _one_pass_delegate(
            kit,
            "r1",
            supervisor,
            supervisor.on_event,
            event,
            _make_binding(),
            _make_context(),
            WorkerStrategy.SEQUENTIAL,
            [w1],
        )

        # Delegate should have been called with the raw user message
        call_args = kit.delegate.call_args_list[0]
        assert "Research quantum computing" in call_args[0][2]
        assert isinstance(result, ChannelOutput)

    async def test_one_pass_empty_message_uses_original(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        supervisor = _make_agent("boss", responses=["ok"])

        from roomkit.models.event import MediaContent

        event = RoomEvent(
            room_id="r1",
            source=EventSource(channel_id="user", channel_type=ChannelType.SMS),
            content=MediaContent(url="https://example.com/img.png", mime_type="image/png"),
        )

        result = await _one_pass_delegate(
            kit,
            "r1",
            supervisor,
            supervisor.on_event,
            event,
            _make_binding(),
            _make_context(),
            WorkerStrategy.SEQUENTIAL,
            [_make_agent("w1")],
        )

        # No delegation should happen — falls through to original
        kit.delegate.assert_not_called()
        assert isinstance(result, ChannelOutput)


# -- Tests: _async_run_and_deliver -------------------------------------------


class TestAsyncRunAndDeliver:
    async def test_delivers_results(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        w1 = _make_agent("w1")
        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("Analysis done"))
        on_done = MagicMock()

        await _async_run_and_deliver(
            kit=kit,
            room_id="r1",
            strategy=WorkerStrategy.SEQUENTIAL,
            workers=[w1],
            task_desc="Analyze this",
            on_done=on_done,
        )

        kit.deliver.assert_called_once()
        delivered_text = kit.deliver.call_args[0][1]
        assert "Analysis done" in delivered_text
        on_done.assert_called_once()

    async def test_calls_on_done_even_on_failure(self) -> None:
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(side_effect=RuntimeError("Boom"))
        on_done = MagicMock()

        await _async_run_and_deliver(
            kit=kit,
            room_id="r1",
            strategy=WorkerStrategy.SEQUENTIAL,
            workers=[_make_agent("w1")],
            task_desc="task",
            on_done=on_done,
        )

        # on_done should still be called in finally block
        on_done.assert_called_once()
        # deliver should not have been called
        kit.deliver.assert_not_called()


# -- Tests: Auto-delegate wrapped on_event -----------------------------------


class TestAutoDelegate:
    async def test_sync_auto_delegate_skips_supervisor_events(self) -> None:
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("result"))

        s = Supervisor(
            supervisor=boss,
            workers=[_make_agent("w1")],
            strategy="sequential",
            auto_delegate=True,
        )
        await s.install(kit, "r1")

        # Events from the supervisor itself should return empty
        event = _make_event(channel_id="boss", channel_type=ChannelType.AI)
        result = await boss.on_event(event, _make_binding(), _make_context())
        assert not result.responded

    async def test_sync_auto_delegate_skips_ai_events(self) -> None:
        boss = _make_agent("boss")
        kit = _make_mock_kit(Room(id="r1"))

        s = Supervisor(
            supervisor=boss,
            workers=[_make_agent("w1")],
            strategy="sequential",
            auto_delegate=True,
        )
        await s.install(kit, "r1")

        event = _make_event(channel_id="other-ai", channel_type=ChannelType.AI)
        result = await boss.on_event(event, _make_binding(), _make_context())
        assert not result.responded

    async def test_sync_auto_delegate_with_refine(self) -> None:
        boss = _make_agent("boss", responses=["TopicExtracted", "Final presentation"])
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("Worker output"))

        s = Supervisor(
            supervisor=boss,
            workers=[_make_agent("w1")],
            strategy="sequential",
            auto_delegate=True,
            refine_task=True,
        )
        await s.install(kit, "r1")

        event = _make_event(body="Analyse quantum computing")
        result = await boss.on_event(event, _make_binding(), _make_context())
        assert isinstance(result, ChannelOutput)

    async def test_sync_auto_delegate_without_refine(self) -> None:
        boss = _make_agent("boss", responses=["Presentation"])
        kit = _make_mock_kit(Room(id="r1"))
        kit.delegate = AsyncMock(return_value=_delegated_task_with_output("Worker output"))

        s = Supervisor(
            supervisor=boss,
            workers=[_make_agent("w1")],
            strategy="sequential",
            auto_delegate=True,
            refine_task=False,
        )
        await s.install(kit, "r1")

        event = _make_event(body="Analyse quantum computing")
        result = await boss.on_event(event, _make_binding(), _make_context())
        assert isinstance(result, ChannelOutput)
