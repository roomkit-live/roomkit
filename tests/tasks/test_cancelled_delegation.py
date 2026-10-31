"""A delegation cancelled from outside ends as every task ends (RMK-434, RFC §23.3).

Inline (its caller's timeout) or in the background (the runner's ``cancel``
or ``close``, even before the task ran a line): the task ends ``cancelled``,
``ON_TASK_COMPLETED`` and its completion callback run, the notified agent is
told, and the delegation span ends with the task's status. A supervisor's
worker is free again once its task was cancelled.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import ChannelCategory, HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.core.mixins import delegation
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import TaskStatus
from roomkit.models.event import TextContent
from roomkit.orchestration._worker_run import task_output
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.orchestration.strategies.supervisor import Supervisor
from roomkit.orchestration.strategies.supervisor.execution import _run_sequential
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks import InMemoryTaskRunner
from roomkit.tasks.models import DelegatedTaskResult, cancelled_task_fields
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.mock import MockTelemetryProvider
from tests.bounds import CUT, expire, opened_with_cut
from tests.test_framework import SimpleChannel

LOOKUP = AITool(name="lookup", description="look up", parameters={"type": "object"})
CALL = AIResponse(
    content="Checking.",
    finish_reason="tool_calls",
    tool_calls=[AIToolCall(id="c", name="lookup", arguments={})],
)


async def _slow(name: str, arguments: dict[str, Any]) -> str:
    await asyncio.sleep(5)
    return "found"


def _working(started: asyncio.Event) -> Any:
    """The slow lookup, *started* set once the worker's turn runs it: its task
    ran a line, and a cut now lands on its work."""

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        started.set()
        return await _slow(name, arguments)

    return lookup


async def _times_out_once(
    event: asyncio.Event, deadline: asyncio.Timeout, then: asyncio.Event | None = None
) -> None:
    """Expire *deadline* once *event* is set, on a slow machine as on a fast
    one; then set *then* once the cut is on its way to the caller, for what
    the cut call still waits for."""
    await event.wait()
    deadline.reschedule(asyncio.get_running_loop().time())
    if then is None:
        return
    while not deadline.expired():
        await asyncio.sleep(0)
    then.set()


class _Ending:
    """A completion callback holding its task's end open until released: *seen*
    records that it entered, then that it finished."""

    def __init__(self) -> None:
        self.seen: list[Any] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def on_complete(self, result: DelegatedTaskResult) -> None:
        self.seen.append(result.status)
        self.entered.set()
        await self.release.wait()
        self.seen.append("done")


@pytest.fixture
def bounds(monkeypatch: pytest.MonkeyPatch) -> list[asyncio.Timeout]:
    return opened_with_cut(monkeypatch)


async def _found(name: str, arguments: dict[str, Any]) -> str:
    return "found"


async def _kit(
    responses: list[AIResponse] | None = None,
    handler: Any = _slow,
    task_runner: InMemoryTaskRunner | None = None,
) -> tuple[RoomKit, MockTelemetryProvider, list[Any]]:
    telemetry = MockTelemetryProvider()
    kit = RoomKit(telemetry=telemetry, task_runner=task_runner)
    kit.register_channel(
        Agent(
            "worker",
            provider=MockAIProvider(ai_responses=responses or [CALL] * 9, streaming=True),
            tools=[LOOKUP],
            tool_handler=handler,
            tool_search=False,
            max_tool_rounds=3,
        )
    )
    await kit.create_room(room_id="p")
    completed: list[Any] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC, name="completed")
    async def on_completed(event: Any, ctx: Any) -> None:
        completed.append((event.metadata.get("task_status"), event.metadata.get("error")))

    return kit, telemetry, completed


def _delegation_spans(telemetry: MockTelemetryProvider) -> list[tuple[str, str]]:
    assert not [s for s in telemetry.get_active_spans() if s.kind == SpanKind.DELEGATION]
    return [(s.name, s.status) for s in telemetry.completed_spans if s.kind == SpanKind.DELEGATION]


async def _settle(completed: list[Any]) -> None:
    for _ in range(100):
        if completed:
            return
        await asyncio.sleep(0.01)


async def test_an_inline_delegation_its_caller_times_out_ends_cancelled() -> None:
    """The caller's timeout lands while the worker's turn runs its tool, on a
    slow machine as on a fast one: a fixed timeout could land before the task
    was announced, when nothing reports its end (RFC §23.3)."""
    in_turn = asyncio.Event()
    kit, telemetry, completed = await _kit(handler=_working(in_turn))
    ended: list[DelegatedTaskResult] = []

    async def on_complete(result: DelegatedTaskResult) -> None:
        ended.append(result)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(None) as deadline:
            watcher = asyncio.ensure_future(_times_out_once(in_turn, deadline))
            await kit.delegate("p", "worker", "go", wait=True, on_complete=on_complete)
    await watcher
    await _settle(completed)

    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    assert [r.status for r in ended] == [TaskStatus.CANCELLED]
    assert _delegation_spans(telemetry) == [("delegation.inline", "cancelled")]
    await kit.close()


async def test_a_background_delegation_the_runner_cancels_ends_cancelled() -> None:
    started = asyncio.Event()
    kit, telemetry, completed = await _kit(handler=_working(started))
    ended: list[DelegatedTaskResult] = []

    async def on_complete(result: DelegatedTaskResult) -> None:
        ended.append(result)

    task = await kit.delegate("p", "worker", "go", on_complete=on_complete)
    await asyncio.wait_for(started.wait(), 5)

    assert await kit.task_runner.cancel(task.id) is True
    await _settle(completed)

    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    assert [r.status for r in ended] == [TaskStatus.CANCELLED]
    assert task.result is not None and task.result.error == "cancelled"
    assert _delegation_spans(telemetry) == [("delegation.background", "cancelled")]
    await kit.close()


async def test_a_task_cancelled_before_it_ran_a_line_still_ends() -> None:
    kit, telemetry, completed = await _kit()

    task = await kit.delegate("p", "worker", "go")
    await kit.task_runner.cancel(task.id)

    assert (await task.wait(timeout=1)).status == TaskStatus.CANCELLED
    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    assert _delegation_spans(telemetry) == [("delegation.background", "cancelled")]
    await kit.close()


async def test_closing_the_runner_ends_its_tasks_cancelled() -> None:
    started = asyncio.Event()
    kit, _, completed = await _kit(handler=_working(started))
    task = await kit.delegate("p", "worker", "go")
    await asyncio.wait_for(started.wait(), 5)

    await kit.task_runner.close()

    assert task.status == TaskStatus.CANCELLED
    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    await kit.close()


async def test_the_notified_agent_is_told_its_task_was_cancelled() -> None:
    started = asyncio.Event()
    kit, _, completed = await _kit(handler=_working(started))
    notified = AIChannel("assistant", provider=MockAIProvider(responses=["ok"]))
    kit.register_channel(notified)
    kit.register_channel(SimpleChannel("phone"))
    await kit.attach_channel("p", "phone")
    await kit.attach_channel("p", "assistant", category=ChannelCategory.INTELLIGENCE)
    task = await kit.delegate("p", "worker", "go", notify="assistant")
    await asyncio.wait_for(started.wait(), 5)

    await kit.task_runner.cancel(task.id)
    for _ in range(100):
        if notified._provider.calls:
            break
        await asyncio.sleep(0.01)

    [call] = notified._provider.calls
    assert "[Background task from worker cancelled." in str(call.messages[-1].content)
    await kit.close()


@pytest.mark.parametrize(
    ("responses", "span_status"),
    [([AIResponse(content="Done.")], "ok"), ([CALL] * 9, "error")],
)
async def test_a_delegation_span_ends_with_its_task_status(
    responses: list[AIResponse], span_status: str
) -> None:
    kit, telemetry, _ = await _kit(responses, _found)

    await kit.delegate("p", "worker", "go", wait=True)

    assert _delegation_spans(telemetry) == [("delegation.inline", span_status)]
    await kit.close()


def _delegating(n: int) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=f"d{n}", name="delegate_to_worker", arguments={"task": "go"})],
    )


async def test_a_supervisors_worker_is_free_once_its_task_was_cancelled() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    supervisor = Agent(
        "sup",
        provider=MockAIProvider(
            ai_responses=[
                _delegating(1),
                AIResponse(content="Started."),
                _delegating(2),
                AIResponse(content="Started again."),
            ],
            streaming=True,
        ),
        tool_search=False,
    )
    worker = Agent(
        "worker",
        provider=MockAIProvider(ai_responses=[CALL], streaming=True),
        tools=[LOOKUP],
        tool_handler=_slow,
        tool_search=False,
    )
    kit.register_channel(supervisor)
    kit.register_channel(worker)
    delegated: list[str] = []

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC, name="delegated")
    async def on_delegated(event: Any, ctx: Any) -> None:
        delegated.append(event.metadata["task_id"])

    await kit.create_room(
        room_id="r",
        orchestration=Supervisor(supervisor=supervisor, workers=[worker], wait_for_result=False),
    )
    await kit.attach_channel("r", "sms")

    async def ask(text: str, delegations: int) -> None:
        await kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body=text))
        )
        # The delegation reaches an async hook: wait for it, not a fixed delay.
        async with asyncio.timeout(5):
            while len(delegated) < delegations:
                await asyncio.sleep(0.01)

    await ask("Find it.", 1)
    await kit.task_runner.cancel(delegated[0])
    await ask("Try again.", 2)

    # The second call delegated anew, the worker no longer "already running".
    assert len(delegated) == 2

    async def told_not_completed() -> bool:
        entries = await kit.status_bus.recent(50)
        return any(
            (e.agent_id, e.action, e.status, e.detail)
            == ("orchestration", "worker", StatusLevel.FAILED, "not completed")
            for e in entries
        )

    # And the cancelled run's line ends as not completed.
    async with asyncio.timeout(5):
        while not await told_not_completed():
            await asyncio.sleep(0.01)
    await kit.close()


async def test_a_supervisors_task_timeout_ends_the_task_cancelled(
    bounds: list[asyncio.Timeout],
) -> None:
    started = asyncio.Event()
    kit, telemetry, completed = await _kit(handler=_working(started))

    async def times_out_once_working() -> None:
        await started.wait()
        expire(bounds)

    watcher = asyncio.ensure_future(times_out_once_working())
    await _run_sequential(kit, "p", [kit.channels["worker"]], "go", task_timeout=CUT)  # type: ignore[list-item]
    await watcher
    await _settle(completed)

    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    assert _delegation_spans(telemetry) == [("delegation.inline", "cancelled")]
    await kit.close()


async def test_a_cancel_cut_while_the_task_ends_still_ends_it() -> None:
    started, ending = asyncio.Event(), _Ending()
    kit, _, completed = await _kit(handler=_working(started))

    task = await kit.delegate("p", "worker", "go", on_complete=ending.on_complete)
    await asyncio.wait_for(started.wait(), 5)
    cancelling = asyncio.create_task(kit.task_runner.cancel(task.id))
    await asyncio.wait_for(ending.entered.wait(), 5)
    cancelling.cancel()
    await asyncio.gather(cancelling, return_exceptions=True)
    ending.release.set()

    assert (await task.wait(timeout=1)).status == TaskStatus.CANCELLED
    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    await kit.close()


DONE = [AIResponse(content="Done.")]


@pytest.mark.parametrize("door", ["cancel", "close"])
async def test_a_task_that_ran_ends_as_it_stands_whatever_cancels_it(door: str) -> None:
    """Its work done, a task is ending (its callback, the hand-back): a
    cancel then ends it once, as it stands, never a second time cancelled."""
    kit, telemetry, completed = await _kit(DONE)
    ending = _Ending()
    task = await kit.delegate("p", "worker", "go", on_complete=ending.on_complete)
    await asyncio.wait_for(ending.entered.wait(), 5)

    asked = asyncio.ensure_future(
        kit.task_runner.cancel(task.id) if door == "cancel" else kit.task_runner.close()
    )
    await asyncio.wait({asked}, timeout=0.05)
    assert not asked.done()  # it waits for the task's end, under way
    ending.release.set()
    answer = await asyncio.wait_for(asked, 5)
    if door == "cancel":
        assert answer is False

    assert ending.seen == [TaskStatus.COMPLETED, "done"]
    assert completed == [(TaskStatus.COMPLETED, None)]
    assert task.status == TaskStatus.COMPLETED
    room = await kit.get_room(task.child_room_id)
    assert (room.metadata["task_status"], room.metadata["task_result"]) == ("completed", "Done.")
    assert _delegation_spans(telemetry) == [("delegation.background", "ok")]
    await kit.close()


async def test_a_cancel_cut_while_the_worker_unwinds_still_ends_the_task() -> None:
    started, unwinding, unwound = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def slow_cleanup(name: str, arguments: dict[str, Any]) -> str:
        started.set()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            unwinding.set()
            await asyncio.shield(unwound.wait())
            raise
        return "found"

    kit, telemetry, completed = await _kit(handler=slow_cleanup)
    task = await kit.delegate("p", "worker", "go")
    await asyncio.wait_for(started.wait(), 5)

    cancelling = asyncio.ensure_future(kit.task_runner.cancel(task.id))
    await asyncio.wait_for(unwinding.wait(), 5)
    await asyncio.wait({cancelling}, timeout=0.05)
    assert not cancelling.done() and task.result is None  # it waits for the unwind
    cancelling.cancel()  # its caller gives up while the worker unwinds
    unwound.set()
    await asyncio.gather(cancelling, return_exceptions=True)

    assert (await task.wait(timeout=1)).status == TaskStatus.CANCELLED
    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    assert _delegation_spans(telemetry) == [("delegation.background", "cancelled")]
    await kit.close()


async def test_an_inline_caller_timing_out_while_the_task_ends_leaves_its_end_whole() -> None:
    kit, telemetry, completed = await _kit(DONE)
    ending = _Ending()

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(None) as deadline:
            ended = _times_out_once(ending.entered, deadline, ending.release)
            watcher = asyncio.ensure_future(ended)
            await kit.delegate("p", "worker", "go", wait=True, on_complete=ending.on_complete)
    await watcher
    # The end goes on past the caller: its callback, then its hook.
    async with asyncio.timeout(5):
        while not (completed and ending.seen[-1:] == ["done"]):
            await asyncio.sleep(0.01)

    assert ending.seen == [TaskStatus.COMPLETED, "done"]
    assert completed == [(TaskStatus.COMPLETED, None)]
    assert _delegation_spans(telemetry) == [("delegation.inline", "ok")]
    await kit.close()


class _SlowSubmit(InMemoryTaskRunner):
    """A runner slow to take a task, as a remote queue can be."""

    async def submit(self, kit: Any, task: Any, **kwargs: Any) -> None:
        await asyncio.sleep(1)
        await super().submit(kit, task, **kwargs)


@pytest.mark.parametrize("wait", [True, False])
async def test_a_delegation_cut_before_its_task_ran_ends_it(wait: bool) -> None:
    """The caller's deadline lands once the task is announced, before it
    ran: the inline task starting, or the background one being handed to a
    runner slow to take it. Its ON_TASK_DELEGATED hooks hold neither (RFC
    §9.3)."""
    kit, telemetry, completed = await _kit(task_runner=None if wait else _SlowSubmit())
    announcing = asyncio.Event()

    @kit.hook(HookTrigger.ON_TASK_DELEGATED, execution=HookExecution.ASYNC, name="announced")
    async def announced(event: Any, ctx: Any) -> None:
        announcing.set()

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(None) as deadline:
            watcher = asyncio.ensure_future(_times_out_once(announcing, deadline))
            await kit.delegate("p", "worker", "go", wait=wait)
    await watcher
    await _settle(completed)

    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    mode = "inline" if wait else "background"
    assert _delegation_spans(telemetry) == [(f"delegation.{mode}", "cancelled")]
    await kit.close()


async def test_the_runner_ends_a_task_whose_handle_was_cancelled() -> None:
    """A handle's own ``cancel`` only unblocks its waiters: the runner still
    ends the task when it cancels it."""
    started = asyncio.Event()
    kit, telemetry, completed = await _kit(handler=_working(started))
    task = await kit.delegate("p", "worker", "go")
    await asyncio.wait_for(started.wait(), 5)

    task.cancel()
    assert await kit.task_runner.cancel(task.id) is True

    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    assert _delegation_spans(telemetry) == [("delegation.background", "cancelled")]
    await kit.close()


async def test_closing_the_kit_hands_nothing_back() -> None:
    started = asyncio.Event()
    kit, _, completed = await _kit(handler=_working(started))
    notified = AIChannel("assistant", provider=MockAIProvider(responses=["ok"]))
    kit.register_channel(notified)
    kit.register_channel(SimpleChannel("phone"))
    await kit.attach_channel("p", "phone")
    await kit.attach_channel("p", "assistant", category=ChannelCategory.INTELLIGENCE)
    await kit.delegate("p", "worker", "go", notify="assistant")
    await asyncio.wait_for(started.wait(), 5)

    await kit.close()

    # The task still ends, and its observers hear it; no turn starts.
    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    assert notified._provider.calls == []


async def test_a_task_delegated_while_the_runner_closes_ends_at_once() -> None:
    started = asyncio.Event()
    kit, _, completed = await _kit(handler=_working(started))
    again: list[Any] = []

    async def delegate_again(result: DelegatedTaskResult) -> None:
        again.append(await kit.delegate("p", "worker", "again"))

    await kit.delegate("p", "worker", "go", on_complete=delegate_again)
    await asyncio.wait_for(started.wait(), 5)

    await kit.task_runner.close()

    [retry] = again
    assert retry.status == TaskStatus.CANCELLED
    assert [status for status, _ in completed] == [TaskStatus.CANCELLED, TaskStatus.CANCELLED]
    await kit.close()


async def test_an_inline_task_records_its_end_on_its_child_room() -> None:
    kit, _, _ = await _kit(DONE)

    task = await kit.delegate("p", "worker", "go", wait=True)

    room = await kit.get_room(task.child_room_id)
    assert (room.metadata["task_status"], room.metadata["task_result"]) == ("completed", "Done.")
    await kit.close()


async def test_a_cancelled_task_span_carries_how_long_it_ran() -> None:
    started = asyncio.Event()
    kit, telemetry, completed = await _kit(handler=_working(started))
    task = await kit.delegate("p", "worker", "go")
    await asyncio.wait_for(started.wait(), 5)
    await asyncio.sleep(0.05)  # the time it ran, which its span reads

    await kit.task_runner.cancel(task.id)

    [span] = [s for s in telemetry.completed_spans if s.kind == SpanKind.DELEGATION]
    assert span.attributes[Attr.DURATION_MS] >= 40
    await kit.close()


def test_a_supervisor_reads_a_cancelled_task_as_cancelled() -> None:
    cancelled = DelegatedTaskResult(
        task_id="t",
        child_room_id="c",
        parent_room_id="p",
        agent_id="worker",
        **cancelled_task_fields(None),
    )

    assert task_output(cancelled) == "The task was cancelled."


async def test_a_cancel_that_took_a_task_ends_it_once_though_its_worker_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that swallows the cancellation still returns its answer: the
    cancel took the task's end, and the task ends once, cancelled."""

    started = asyncio.Event()

    async def swallowing(kit: Any, child_room_id: str, task: str, **kw: Any) -> str:
        started.set()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            return "found anyway"
        return "found"

    monkeypatch.setattr(delegation, "run_agent_in_child_room", swallowing)
    kit, telemetry, completed = await _kit()
    task = await kit.delegate("p", "worker", "go")
    await asyncio.wait_for(started.wait(), 5)

    # The cancel waits for the task's work and its end: a second end would be
    # written by then, its hook as soon as the first's.
    assert await kit.task_runner.cancel(task.id) is True
    await _settle(completed)

    assert completed == [(TaskStatus.CANCELLED, "cancelled")]
    assert task.status == TaskStatus.CANCELLED
    assert _delegation_spans(telemetry) == [("delegation.background", "cancelled")]
    await kit.close()


async def test_a_task_that_ran_ends_whole_though_its_runner_task_is_cancelled() -> None:
    kit, _, completed = await _kit(DONE)
    ending = _Ending()
    task = await kit.delegate("p", "worker", "go", on_complete=ending.on_complete)
    await asyncio.wait_for(ending.entered.wait(), 5)

    kit.task_runner._tasks[task.id].cancel()  # type: ignore[attr-defined]
    ending.release.set()

    assert (await task.wait(timeout=1)).status == TaskStatus.COMPLETED
    assert ending.seen == [TaskStatus.COMPLETED, "done"]
    assert completed == [(TaskStatus.COMPLETED, None)]
    await kit.close()


async def test_an_inline_task_whose_turn_ended_before_its_caller_was_cut_ends_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller is cancelled once the worker's turn ended, before it
    resumed: the work ran, so the task ends as it stands, completed with its
    output, then the cancellation goes on (RFC §23.3)."""
    real_turn = delegation.run_agent_in_child_room
    caller: list[asyncio.Task[Any]] = []

    async def turn_then_cut_the_caller(*args: Any, **kwargs: Any) -> Any:
        answer = await real_turn(*args, **kwargs)
        asyncio.get_running_loop().call_soon(caller[0].cancel)  # before the caller resumes
        return answer

    monkeypatch.setattr(delegation, "run_agent_in_child_room", turn_then_cut_the_caller)
    kit = RoomKit()
    ends: list[tuple[str, str]] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def _ended(event: Any, ctx: Any) -> None:
        ends.append((str(event.metadata["task_status"]), event.content.body))

    kit.register_channel(Agent("w1", provider=MockAIProvider(responses=["The findings."])))
    await kit.create_room(room_id="r")

    async def delegate() -> None:
        caller.append(asyncio.current_task())  # type: ignore[arg-type]
        await kit.delegate("r", "w1", "Find it.", wait=True)

    task = asyncio.create_task(delegate())
    await asyncio.wait({task})
    # The end reaches an async hook: wait for it, not a fixed delay.
    async with asyncio.timeout(5):
        while not ends:
            await asyncio.sleep(0.01)
    await kit.close()

    assert task.cancelled()
    assert ends == [("completed", "The findings.")]
