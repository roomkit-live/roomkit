"""The tools RoomKit sets up are served by the room owner (RFC §10.6.5).

A turn's request lists them; the channel declares them and calls them
through the turn's link, and their handler runs on the owner, in a task of
its own, under what the turn gave the call.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from roomkit import RoomKit
from roomkit.channels._tool_registry import orchestration_tool
from roomkit.channels.agent import Agent
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, HookExecution, HookTrigger
from roomkit.models.event import TextContent
from roomkit.models.tool_call import ToolCallEvent
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.orchestration.strategies.swarm import Swarm
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks.delegate import DelegateHandler, setup_delegation
from roomkit.tools.context import (
    current_response_metadata,
    current_tool_actor_id,
    current_tool_allowed_names,
    current_tool_call,
    current_tool_chain_depth,
    current_tool_requester,
    current_tool_room,
    current_tool_room_id,
)
from roomkit.turns._registry_tools import RegistryOwnerTools
from tests.bounds import CUT, expire, opened_with_cut
from tests.test_framework import SimpleChannel

PROBE = AITool(name="probe", description="Probe the owner.", parameters={"type": "object"})


def _call(name: str, arguments: dict[str, Any], call_id: str = "c1") -> AIResponse:
    return AIResponse(
        content="", tool_calls=[AIToolCall(id=call_id, name=name, arguments=arguments)]
    )


def _agent(
    name: str, *responses: AIResponse | str, streaming: bool = False, **kwargs: Any
) -> Agent:
    scripted = [r if isinstance(r, AIResponse) else AIResponse(content=r) for r in responses]
    provider = MockAIProvider(ai_responses=scripted or [AIResponse(content=f"{name} here")])
    provider._streaming = streaming
    return Agent(name, provider=provider, role=name, **kwargs)


async def _kit(*agents: Agent, attach: tuple[str, ...] = ()) -> RoomKit:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("ops"))
    for agent in agents:
        kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ops")
    for channel_id in attach:
        await kit.attach_channel("r1", channel_id, category=ChannelCategory.INTELLIGENCE)
    return kit


async def _say(kit: RoomKit, body: str) -> Any:
    return await kit.process_inbound(
        InboundMessage(channel_id="ops", sender_id="ops", content=TextContent(body=body)),
        room_id="r1",
    )


def _reports(kit: RoomKit) -> list[ToolCallEvent]:
    reports: list[ToolCallEvent] = []

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC)
    async def report(event: ToolCallEvent, ctx: Any) -> None:
        reports.append(event)

    return reports


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Each call the owner served: the tool's name, the task that ran it."""
    seen: list[tuple[str, str]] = []
    serve = RegistryOwnerTools.serve

    async def spy(self: RegistryOwnerTools, request: Any, call: Any) -> Any:
        task = asyncio.current_task()
        seen.append((call.name, task.get_name() if task else ""))
        return await serve(self, request, call)

    monkeypatch.setattr(RegistryOwnerTools, "serve", spy)
    return seen


def _on_owner(served: list[tuple[str, str]], name: str) -> bool:
    return any(n == name and task.startswith("roomkit-owner-call-") for n, task in served)


# -- The call ------------------------------------------------------------------


@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "not-streaming"])
async def test_a_handler_runs_on_the_owner_under_what_the_turn_gave_its_call(
    streaming: bool,
) -> None:
    seen: dict[str, Any] = {}

    async def serve(arguments: dict[str, Any]) -> dict[str, Any]:
        task = asyncio.current_task()
        record = current_tool_call()
        assert record is not None
        record.structured_content = {"rows": 2}
        seen.update(
            task=task.get_name() if task else "",
            arguments=arguments,
            room=current_tool_room_id(),
            room_object=getattr(current_tool_room(), "id", None),
            requester=current_tool_requester(),
            call=(record.tool_call_id, record.channel_id),
            depth=current_tool_chain_depth(),
            actor=current_tool_actor_id(),
            allowed=current_tool_allowed_names(),
            record=current_response_metadata(),
        )
        return {"ok": 1}

    agent = _agent("agent", _call("probe", {"q": "x"}), "done", streaming=streaming)
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve), room_id="r1", owner=object()
    )
    reports = _reports(kit)

    await kit.send_event(
        room_id="r1", channel_id="ops", content=TextContent(body="go"), participant_id="alice"
    )

    assert seen["task"].startswith("roomkit-owner-call-")
    assert seen["arguments"] == {"q": "x"}
    assert seen["room"] == "r1"
    assert seen["room_object"] == "r1"
    # A one-to-one conversation names no requester, locally as here.
    assert seen["requester"] is None
    assert seen["call"] == ("c1", "agent")
    assert seen["depth"] == 1
    assert seen["actor"] == "alice"
    assert "probe" in seen["allowed"]
    # The turn's record is the executor's.
    assert seen["record"] is None
    [report] = reports
    assert report.result == '{"ok": 1}'
    assert report.structured_content == {"rows": 2}
    assert not report.is_error
    # The model read the answer and the turn went on.
    assert len(agent.provider.calls) == 2
    await kit.close()


async def test_a_handler_that_raises_fails_the_call_with_its_class() -> None:
    async def serve(arguments: dict[str, Any]) -> str:
        raise ValueError("no such row")

    agent = _agent("agent", _call("probe", {}), "sorry")
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve), room_id="r1", owner=object()
    )
    reports = _reports(kit)

    await _say(kit, "go")

    [report] = reports
    assert report.is_error
    assert report.error_detail == "ValueError: no such row"
    await kit.close()


async def test_a_call_nothing_on_the_owner_serves_is_served_by_nothing() -> None:
    async def serve(arguments: dict[str, Any]) -> str:
        raise UnservedToolCallError("not mine")

    agent = _agent("agent", _call("probe", {}), "fine")
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve), room_id="r1", owner=object()
    )
    reports = _reports(kit)

    await _say(kit, "go")

    [report] = reports
    assert report.is_error
    await kit.close()


async def test_a_call_past_its_bound_is_withdrawn_from_the_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def serve(arguments: dict[str, Any]) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never"

    agent = _agent("agent", _call("probe", {}), "too slow", tool_timeout_seconds=CUT)
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve), room_id="r1", owner=object()
    )
    reports = _reports(kit)
    bounds = opened_with_cut(monkeypatch)

    turn = asyncio.ensure_future(_say(kit, "go"))
    await asyncio.wait_for(started.wait(), 2)
    expire(bounds)  # the call's bound runs out while its handler serves it
    await asyncio.wait_for(turn, 5)

    await asyncio.wait_for(cancelled.wait(), 2)
    [report] = reports
    assert report.is_error
    await kit.close()


async def test_a_turn_cut_withdraws_its_calls_from_the_owner() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def serve(arguments: dict[str, Any]) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never"

    agent = _agent("agent", _call("probe", {}), "unread")
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve, waits=True), room_id="r1", owner=object()
    )

    turn = asyncio.create_task(_say(kit, "go"))
    await asyncio.wait_for(started.wait(), 2)
    turn.cancel()
    await asyncio.gather(turn, return_exceptions=True)

    await asyncio.wait_for(cancelled.wait(), 2)
    await kit.close()


async def test_a_structured_copy_that_cannot_cross_fails_the_call() -> None:
    async def serve(arguments: dict[str, Any]) -> str:
        record = current_tool_call()
        assert record is not None
        record.structured_content = {"handle": object()}
        return "ok"

    agent = _agent("agent", _call("probe", {}), "fine")
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve), room_id="r1", owner=object()
    )
    reports = _reports(kit)

    await _say(kit, "go")

    [report] = reports
    assert report.is_error
    assert report.error_detail is not None
    assert "cannot cross the turn backend" in report.error_detail
    await kit.close()


# -- The tools RoomKit sets up -------------------------------------------------


async def test_delegate_task_is_served_on_the_owner(served: list[tuple[str, str]]) -> None:
    boss = _agent("boss", _call("delegate_task", {"agent": "worker", "task": "Look"}), "Sent.")
    worker = _agent("worker", "worker result")
    kit = await _kit(boss, worker, attach=("boss",))
    setup_delegation(boss, DelegateHandler(kit))

    await _say(kit, "go")

    assert _on_owner(served, "delegate_task")
    await kit.close()


async def test_work_in_background_is_served_on_the_owner(served: list[tuple[str, str]]) -> None:
    agent = _agent("agent", _call("work_in_background", {"task": "Long job"}), "Started.")
    kit = await _kit(agent, attach=("agent",))
    kit.setup_background(agent)

    await _say(kit, "go")

    assert _on_owner(served, "work_in_background")
    await kit.close()


async def test_a_swarm_handoff_is_served_on_the_owner(served: list[tuple[str, str]]) -> None:
    first = _agent(
        "first",
        _call("handoff_conversation", {"target": "second", "reason": "r", "summary": "s"}),
        "Handing over.",
    )
    second = _agent("second", "Here.")
    kit = await _kit(first, second)
    await kit.install_strategy("r1", Swarm(agents=[first, second], entry="first"))

    await _say(kit, "go")

    assert _on_owner(served, "handoff_conversation")
    await kit.close()


async def test_a_supervisors_strategy_tool_is_served_on_the_owner(
    served: list[tuple[str, str]],
) -> None:
    boss = _agent("boss", _call("delegate_workers", {"task": "Look"}), "Found it.")
    worker = _agent("worker", "worker result")
    kit = await _kit(boss, worker)
    await kit.install_strategy(
        "r1", Supervisor(supervisor=boss, workers=[worker], strategy="parallel")
    )

    await _say(kit, "go")

    assert _on_owner(served, "delegate_workers")
    await kit.close()


async def test_a_delegations_result_tool_is_served_on_the_owner(
    served: list[tuple[str, str]],
) -> None:
    worker = _agent(
        "worker", _call("submit_result", {"status": "completed", "summary": "Looked."}), "ok"
    )
    kit = await _kit(worker)

    task = await kit.delegate("r1", "worker", "Look", wait=True, require_structured_result=True)

    assert _on_owner(served, "submit_result")
    assert task.result is not None
    assert task.result.output is not None
    assert "Looked." in task.result.output
    await kit.close()


async def test_a_supervisors_per_worker_tool_is_served_on_the_owner(
    served: list[tuple[str, str]],
) -> None:
    boss = _agent("boss", _call("delegate_to_worker", {"task": "Look"}), "Found it.")
    worker = _agent("worker", "worker result")
    kit = await _kit(boss, worker)
    await kit.install_strategy("r1", Supervisor(supervisor=boss, workers=[worker]))

    await _say(kit, "go")

    assert _on_owner(served, "delegate_to_worker")
    await kit.close()


# -- Withdrawals, closes and failures -----------------------------------------


class QuotaExceededError(Exception):
    """A host's own error class, which the turn backend does not rebuild."""


async def test_a_handlers_error_of_a_class_of_its_own_keeps_its_name() -> None:
    async def serve(arguments: dict[str, Any]) -> str:
        raise QuotaExceededError("over quota")

    agent = _agent("agent", _call("probe", {}), "sorry")
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve), room_id="r1", owner=object()
    )
    reports = _reports(kit)

    await _say(kit, "go")

    [report] = reports
    assert report.error_detail == "QuotaExceededError: over quota"
    assert "(QuotaExceededError)" in str(report.result)
    await kit.close()


async def test_a_call_past_its_bound_is_cancelled_on_the_owner_while_its_turn_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started, cancelled = asyncio.Event(), asyncio.Event()
    committed: list[str] = []
    seen_during_turn: list[bool] = []

    async def slow(arguments: dict[str, Any]) -> str:
        started.set()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        committed.append("side effect")
        return "late"

    async def after(arguments: dict[str, Any]) -> str:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(cancelled.wait(), 2)
        seen_during_turn.append(cancelled.is_set())
        return "checked"

    later = AITool(name="later", description="Later.", parameters={"type": "object"})
    agent = _agent(
        "agent",
        _call("probe", {}),
        _call("later", {}, call_id="c2"),
        "done",
        tool_timeout_seconds=CUT,
    )
    owner = object()
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, slow), room_id="r1", owner=owner
    )
    kit._owner_tools.of("agent").register(
        orchestration_tool(later, after, waits=True), room_id="r1", owner=owner
    )
    bounds = opened_with_cut(monkeypatch)

    turn = asyncio.ensure_future(_say(kit, "go"))
    await asyncio.wait_for(started.wait(), 2)
    expire(bounds)
    await asyncio.wait_for(turn, 5)

    # Withdrawn at its bound, while the turn's next round ran: the handler
    # never committed what it was doing.
    assert seen_during_turn == [True]
    assert committed == []
    await kit.close()


async def test_a_handler_ended_by_a_cancellation_not_of_its_call_fails_the_call() -> None:
    async def serve(arguments: dict[str, Any]) -> str:
        job = asyncio.get_running_loop().create_future()
        job.cancel()
        return await job

    agent = _agent("agent", _call("probe", {}), "moved on")
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve, waits=True), room_id="r1", owner=object()
    )
    reports = _reports(kit)

    await asyncio.wait_for(_say(kit, "go"), 5)

    [report] = reports
    assert report.is_error
    assert len(agent.provider.calls) == 2
    await kit.close()


async def test_parallel_calls_of_one_round_each_run_on_the_owner_at_once() -> None:
    both = asyncio.Barrier(2)

    async def serve(arguments: dict[str, Any]) -> str:
        await asyncio.wait_for(both.wait(), 2)
        return f"served {arguments['n']}"

    two = AIResponse(
        content="",
        tool_calls=[
            AIToolCall(id="c1", name="probe", arguments={"n": 1}),
            AIToolCall(id="c2", name="probe", arguments={"n": 2}),
        ],
    )
    agent = _agent("agent", two, "both")
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve), room_id="r1", owner=object()
    )
    reports = _reports(kit)

    await _say(kit, "go")

    assert sorted(str(r.result) for r in reports) == ["served 1", "served 2"]
    await kit.close()


async def test_the_kits_close_waits_for_a_running_handler_to_unwind() -> None:
    started = asyncio.Event()
    order: list[str] = []

    async def serve(arguments: dict[str, Any]) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # A cleanup that takes its time (a store write, say).
            await asyncio.sleep(0.2)
            order.append("handler unwound")
            raise
        return "never"

    agent = _agent("agent", _call("probe", {}), "unread")
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve, waits=True), room_id="r1", owner=object()
    )
    turn = asyncio.create_task(_say(kit, "go"))
    await asyncio.wait_for(started.wait(), 2)

    await kit.close()
    order.append("kit closed")

    assert order == ["handler unwound", "kit closed"]
    await asyncio.gather(turn, return_exceptions=True)


async def test_the_kits_close_abandons_a_handler_ignoring_its_cut_once_its_unwind_ran_out(
    close_grace: float, cut_unwind: float, caplog: pytest.LogCaptureFixture
) -> None:
    """An owner-served call that ignores its cut does not hold the kit's
    close: it is abandoned once the unwind ran out (RFC §12.10.4)."""
    started, release = asyncio.Event(), asyncio.Event()

    async def serve(arguments: dict[str, Any]) -> str:
        started.set()
        while not release.is_set():
            with contextlib.suppress(asyncio.CancelledError):
                await release.wait()
        return "late"

    agent = _agent("agent", _call("probe", {}), "unread")
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve, waits=True), room_id="r1", owner=object()
    )
    turn = asyncio.create_task(_say(kit, "go"))
    await asyncio.wait_for(started.wait(), 2)

    try:
        await asyncio.wait_for(kit.close(), 1)
    finally:
        release.set()
    await asyncio.gather(turn, return_exceptions=True)


async def test_a_handler_that_closes_the_kit_sees_the_close_through() -> None:
    steps: list[str] = []
    closed = asyncio.Event()

    async def serve(arguments: dict[str, Any]) -> str:
        steps.append("close called")
        try:
            await kit.close()
        except asyncio.CancelledError:
            steps.append("close cancelled")
            raise
        steps.append("close returned")
        closed.set()
        return "closed"

    agent = _agent("agent", _call("probe", {}), "done")
    kit = await _kit(agent, attach=("agent",))
    kit._owner_tools.of("agent").register(
        orchestration_tool(PROBE, serve, waits=True), room_id="r1", owner=object()
    )

    await asyncio.wait_for(asyncio.gather(_say(kit, "go"), return_exceptions=True), 5)
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(closed.wait(), 5)

    # The close ran to its end: it spared the turn it was called from.
    assert steps == ["close called", "close returned"]
    assert kit._closed
