"""An agent cancels a task of its room with ``cancel_task``, and is not told twice
(RMK-549; RFC §23.3, §23.4).

A request the person changed or dropped ("the weather in Québec… no, Montréal")
must be able to stop its work: the task ends ``cancelled`` as any task cancelled
from outside, and the agent that cancelled it is not handed the cancellation
back, which would have it say so twice. A host's ``kit.cancel_task`` still tells
the agent, which did not ask for it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import roomkit.tasks._room_tasks as room_tasks_module
from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.core.exceptions import ToolNameCollisionError, UnservedToolCallError
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType, TaskStatus
from roomkit.models.event import TextContent
from roomkit.orchestration._worker_run import WorkerStatus, follow_worker, run_worker
from roomkit.orchestration.status_bus import (
    InMemoryStatusBackend,
    StatusBus,
    StatusEntry,
    StatusLevel,
)
from roomkit.providers.ai.base import AIContext, AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tasks import CANCEL_TASK_TOOL, CancelTaskTool
from roomkit.tasks._room_tasks import ENDED_ROOM_TASKS_KEPT, RoomTaskRecord
from roomkit.tasks.handback import CALLER_HANDS_BACK
from roomkit.tasks.status import TASK_STATUS_TOOL, post_task_progress, room_task_lines
from roomkit.tools.context import ToolCallContext, tool_turn_context
from roomkit.turns._registry_tools import RegistryOwnerTools
from tests.conference.test_conference_realtime import until
from tests.test_framework import SimpleChannel


class _Endless(MockAIProvider):
    """A worker that never finishes on its own."""

    def __init__(self) -> None:
        super().__init__(responses=["never"])
        self.started = asyncio.Event()

    async def generate(self, context: AIContext) -> AIResponse:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _Room:
    """Room ``r``: a transport, the speaking agent, a worker; what is handed back
    and which tasks ended."""

    def __init__(self, worker: MockAIProvider) -> None:
        self.kit = RoomKit()
        self.worker = worker
        self.handed_back: list[dict[str, Any]] = []
        self.ended: list[dict[str, Any]] = []

    async def open(self) -> _Room:
        kit = self.kit
        kit.register_channel(SimpleChannel("sms"))
        kit.register_channel(Agent("speaker", provider=MockAIProvider(responses=["Here it is."])))
        kit.register_channel(Agent("worker", provider=self.worker))
        await kit.create_room(room_id="r")
        await kit.attach_channel("r", "sms")
        await kit.attach_channel("r", "speaker", category=ChannelCategory.INTELLIGENCE)

        @kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.INSTRUCTION})
        async def hand_back(event: Any, ctx: Any) -> HookResult:
            self.handed_back.append(dict(event.metadata))
            return HookResult.allow()

        @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
        async def completed(event: Any, ctx: Any) -> None:
            self.ended.append(dict(event.metadata))

        return self

    async def running_task(self, room_id: str = "r") -> Any:
        task = await self.kit.delegate(room_id, "worker", "Count to a thousand.", notify="speaker")
        await asyncio.wait_for(self.worker.started.wait(), 5)
        return task


def _call(channel_id: str = "speaker") -> ToolCallContext:
    return ToolCallContext(room_id="r", tool_call_id="call-1", channel_id=channel_id)


async def _cancel(tool: CancelTaskTool, task_id: str, *, channel_id: str = "speaker") -> Any:
    with tool_turn_context(room_id="r", call=_call(channel_id)):
        return json.loads(await tool.handler(CANCEL_TASK_TOOL, {"task_id": task_id}))


async def test_the_agent_cancels_a_running_task_and_is_not_told_twice() -> None:
    room = await _Room(_Endless()).open()
    task = await room.running_task()

    answer = await _cancel(CancelTaskTool(room.kit), task.id)
    # Its result is set once its end ran, a hand-back included.
    result = await task.wait(timeout=5)
    await until(lambda: bool(room.ended), timeout=5)
    entries = [e for e in await room.kit.status_bus.recent(50) if e.action == "task"]
    await room.kit.close()

    assert answer == {
        "task_id": task.id,
        "status": "cancelled",
        "message": "Cancelled: its result will not come back.",
    }
    assert result.status == TaskStatus.CANCELLED
    assert room.ended[0]["task_status"] == TaskStatus.CANCELLED
    assert entries[-1].metadata["task_status"] == "cancelled"
    assert room.handed_back == []  # the tool's answer told it


@pytest.mark.parametrize("second", ["speaker", "other"], ids=["same-agent", "another-agent"])
async def test_two_cancels_at_once_cancel_the_task_once(second: str) -> None:
    """RMK-874: two cancel_task calls on one running task, from the agent
    twice or from two agents: one cancellation, both answered cancelled, and
    the agent the result was going to is not handed it back."""
    room = await _Room(_Endless()).open()
    task = await room.running_task()
    tool = CancelTaskTool(room.kit)

    answers = await asyncio.gather(
        _cancel(tool, task.id, channel_id="speaker"), _cancel(tool, task.id, channel_id=second)
    )
    await task.wait(timeout=5)  # set once its end ran, a hand-back included
    await until(lambda: bool(room.ended), timeout=5)
    await room.kit.close()

    assert [a["status"] for a in answers] == ["cancelled", "cancelled"]
    assert [e["task_status"] for e in room.ended] == [TaskStatus.CANCELLED]
    assert room.handed_back == []
    assert room.kit._task_cancels == {}


@pytest.mark.parametrize("first", ["host", "agent"])
async def test_the_host_and_an_agent_cancelling_at_once_cancel_once(first: str) -> None:
    """RMK-874 review: the host's cancel_task and an agent's at once, in
    either order: one cancellation, both told it was cancelled, the agent the
    result was going to not handed it back."""
    room = await _Room(_Endless()).open()
    task = await room.running_task()
    tool = CancelTaskTool(room.kit)

    async def host() -> bool:
        return await room.kit.cancel_task(task.id)

    calls = [host(), _cancel(tool, task.id, channel_id="speaker")]
    if first == "agent":
        calls.reverse()
    answers = await asyncio.gather(*calls)
    await task.wait(timeout=5)  # set once its end ran, a hand-back included
    await until(lambda: bool(room.ended), timeout=5)
    await room.kit.close()

    host_said, agent_said = answers if first == "host" else answers[::-1]
    assert (host_said, agent_said["status"]) == (True, "cancelled")
    assert [e["task_status"] for e in room.ended] == [TaskStatus.CANCELLED]
    assert room.handed_back == []


async def test_a_task_the_host_cancels_is_still_handed_back_to_the_agent() -> None:
    room = await _Room(_Endless()).open()
    task = await room.running_task()

    assert await room.kit.cancel_task(task.id) is True
    await until(lambda: bool(room.handed_back), timeout=5)
    await room.kit.close()

    assert room.handed_back[0]["task_id"] == task.id
    assert room.handed_back[0]["task_status"] == "cancelled"


async def test_a_cancel_by_another_agent_still_tells_the_notified_one() -> None:
    room = await _Room(_Endless()).open()
    task = await room.running_task()

    answer = await _cancel(CancelTaskTool(room.kit), task.id, channel_id="assistant-bis")
    await until(lambda: bool(room.handed_back), timeout=5)
    await room.kit.close()

    assert answer["status"] == "cancelled"
    assert room.handed_back[0]["task_status"] == "cancelled"


async def test_another_rooms_task_is_out_of_reach() -> None:
    room = await _Room(_Endless()).open()
    await room.kit.create_room(room_id="other")
    await room.kit.attach_channel("other", "speaker", category=ChannelCategory.INTELLIGENCE)
    theirs = await room.running_task("other")

    answer = await _cancel(CancelTaskTool(room.kit), theirs.id)
    still_running = theirs.result is None
    await room.kit.close()

    assert answer == {
        "task_id": theirs.id,
        "status": "unknown",
        "message": f"No task {theirs.id} here.",
    }
    assert still_running


async def test_a_task_that_already_ended_is_left_as_it_stands() -> None:
    room = await _Room(MockAIProvider(responses=["A thousand."])).open()
    task = await room.kit.delegate("r", "worker", "Count to a thousand.", notify="speaker")
    await task.wait(timeout=5)
    await until(lambda: bool(room.ended), timeout=5)

    answer = await _cancel(CancelTaskTool(room.kit), task.id)
    await room.kit.close()

    assert answer == {
        "task_id": task.id,
        "status": "completed",
        "message": "The task had already ended.",
    }
    assert task.result is not None and task.result.status == TaskStatus.COMPLETED


async def test_a_call_without_a_task_or_outside_a_conversation_is_refused() -> None:
    kit = RoomKit()
    tool = CancelTaskTool(kit)
    with tool_turn_context(room_id="r"):
        missing = json.loads(await tool.handler(CANCEL_TASK_TOOL, {}))
    outside = json.loads(await tool.handler(CANCEL_TASK_TOOL, {"task_id": "t"}))
    await kit.close()

    assert missing == {"error": "task_id is required"}
    assert "error" in outside


async def test_cancel_task_declines_another_tools_call() -> None:
    kit = RoomKit()
    with pytest.raises(UnservedToolCallError):
        await CancelTaskTool(kit).handler("task_status", {})
    await kit.close()


def test_the_definition_asks_for_the_task_id() -> None:
    definition = CancelTaskTool(RoomKit()).definition
    assert definition["name"] == CANCEL_TASK_TOOL
    assert definition["parameters"]["required"] == ["task_id"]


async def test_a_forged_status_line_reaches_no_other_rooms_task() -> None:
    """The StatusBus is not what decides: a ``task`` line for room ``r`` naming
    another room's task, posted by any code holding the kit, cancels nothing."""
    room = await _Room(_Endless()).open()
    await room.kit.create_room(room_id="other")
    await room.kit.attach_channel("other", "speaker", category=ChannelCategory.INTELLIGENCE)
    theirs = await room.running_task("other")
    room.kit.status_bus.post(
        "worker",
        "task",
        "pending",
        metadata={"room_id": "r", "task_id": theirs.id, "task_status": "running"},
    )
    await asyncio.sleep(0)

    # A cancel ends its task before it answers: read right after, it would show.
    answer = await _cancel(CancelTaskTool(room.kit), theirs.id)
    still_running = theirs.result is None
    await room.kit.close()

    assert answer["status"] == "unknown"
    assert still_running


async def test_the_owner_serves_the_task_tools_set_up_on_an_agent() -> None:
    """``setup_task_tools`` makes them the room owner's: listed in a turn's
    request, served on the owner for a turn on any executor."""
    room = await _Room(_Endless()).open()
    speaker = room.kit.channels["speaker"]
    room.kit.setup_task_tools(speaker)  # type: ignore[arg-type]
    task = await room.running_task()

    declared = room.kit._turns._tools.declared("speaker", "r")
    names = sorted(tool.tool.name for tool in declared)
    with tool_turn_context(room_id="r", call=_call()):
        registry = room.kit._owner_tools.get("speaker")
        assert registry is not None
        entry = registry.owner_lookup(CANCEL_TASK_TOOL, "r")
        assert entry is not None and entry.serve is not None
        answer = json.loads(await entry.serve({"task_id": task.id}))
    with pytest.raises(RuntimeError, match="already"):
        room.kit.setup_task_tools(speaker)  # type: ignore[arg-type]
    await room.kit.close()

    assert names == ["cancel_task", "task_status"]
    assert answer["status"] == "cancelled"


class _Failing(MockAIProvider):
    """A worker whose model fails."""

    async def generate(self, context: AIContext) -> AIResponse:
        raise RuntimeError("the model is down")


@pytest.mark.parametrize("ended", ["completed", "failed"])
@pytest.mark.parametrize("door", ["inline", "background"])
async def test_a_task_whose_end_is_being_told_reads_ended_at_once(door: str, ended: str) -> None:
    """While its ON_TASK_COMPLETED hooks run, the room's own task reads
    ended, with the status it ended with, answered at once: its end is
    recorded before its hooks run (RFC §23.4), and its line says so as they
    start, the hooks not waited for (§23.5, §9.3)."""
    worker = MockAIProvider(responses=["A thousand."]) if ended == "completed" else _Failing()
    room = await _Room(worker).open()
    in_hook = asyncio.Event()
    release = asyncio.Event()

    @room.kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def slow(event: Any, ctx: Any) -> None:
        in_hook.set()
        await release.wait()

    run = _delegated(room.kit, door)
    await asyncio.wait_for(in_hook.wait(), 5)
    (line,) = await room_task_lines(room.kit.status_bus, "r", 50)
    try:
        answer = await asyncio.wait_for(_cancel(CancelTaskTool(room.kit), line["task_id"]), 2)
    finally:
        release.set()
    await asyncio.gather(run, return_exceptions=True)
    await room.kit.close()

    assert line["status"] == ended
    assert (answer["status"], answer["message"]) == (ended, "The task had already ended.")
    assert room.kit._room_tasks.running == []


async def test_setup_task_tools_sets_up_both_tools_or_neither() -> None:
    kit = RoomKit()
    agent = Agent("speaker", provider=MockAIProvider(responses=["x"]), tools=[CancelTaskTool(kit)])
    kit.register_channel(agent)

    with pytest.raises(ToolNameCollisionError):
        kit.setup_task_tools(agent)

    registry = kit._owner_tools.get("speaker")
    assert registry is None or registry.lookup(TASK_STATUS_TOOL, None) is None
    await kit.close()


async def test_a_strategy_s_unposted_worker_is_never_a_room_s_to_cancel() -> None:
    """A task a strategy runs without posting it (``post_status=False``) is
    the strategy's to follow, though its follower posts lines (RFC §23.4)."""
    room = await _Room(_Endless()).open()
    task = await room.kit.delegate(
        "r", "worker", "Count.", notify=CALLER_HANDS_BACK, post_status=False
    )
    follower = asyncio.ensure_future(
        follow_worker(
            room.kit,
            task,
            timeout=None,
            status=WorkerStatus({"room_id": "r", "mode": "per_worker_async"}),
        )
    )
    await asyncio.wait_for(room.worker.started.wait(), 5)

    answer = await _cancel(CancelTaskTool(room.kit), task.id)
    still_running = task.result is None
    await room.kit.cancel_task(task.id)
    await asyncio.wait_for(follower, 5)
    await room.kit.close()

    assert answer["status"] == "running"
    assert "the strategy that runs it follows it" in answer["message"]
    assert still_running


async def test_an_inline_delegation_is_said_running_not_unknown() -> None:
    """RMK-829: an inline delegation of the room is listed running by
    ``task_status``; ``cancel_task`` says so too, and why it cannot cancel
    it, never that the task does not exist."""
    room = await _Room(_Endless()).open()
    inline = asyncio.ensure_future(
        room.kit.delegate("r", "worker", "Count to a thousand.", wait=True)
    )
    await asyncio.wait_for(room.worker.started.wait(), 5)
    (line,) = await room_task_lines(room.kit.status_bus, "r", 50)

    answer = await _cancel(CancelTaskTool(room.kit), line["task_id"])
    still_running = not inline.done()
    inline.cancel()
    await asyncio.gather(inline, return_exceptions=True)
    await room.kit.close()

    assert line["status"] == "running"
    assert answer["status"] == "running"
    assert "runs inside the turn that asked for it" in answer["message"]
    assert still_running
    assert room.kit._room_tasks.running == []


@pytest.mark.parametrize("named", ["another room's ended task", "a task never run"])
async def test_a_line_for_a_task_the_room_does_not_hold_is_no_task_here(named: str) -> None:
    """RMK-829: what the room holds is the kit's record, never the bus: a
    line posted for room ``r`` naming another room's task that ended, or an
    id the kit never ran, is answered as no task here, revealing nothing."""
    room = await _Room(MockAIProvider(responses=["done"])).open()
    await room.kit.create_room(room_id="other")
    await room.kit.attach_channel("other", "speaker", category=ChannelCategory.INTELLIGENCE)
    task_id = "task-never-run"
    if named == "another room's ended task":
        theirs = await room.kit.delegate("other", "worker", "Count.", wait=True)
        task_id = theirs.id
    room.kit.status_bus.post(
        "worker", "task", "pending", metadata={"room_id": "r", "task_id": task_id}
    )
    await asyncio.sleep(0)

    answer = await _cancel(CancelTaskTool(room.kit), task_id)
    await room.kit.close()

    assert (answer["status"], answer["message"]) == ("unknown", f"No task {task_id} here.")


async def test_a_task_whose_pending_entry_left_the_window_is_still_cancelled() -> None:
    """A running task of the room whose pending entry the bus's window no
    longer holds, only its progress, is listed running, as the turn's notes
    read it, and ``cancel_task`` cancels it."""
    room = await _Room(_Endless()).open()
    task = await room.running_task()
    for n in range(5):
        room.kit.status_bus.post("worker", "task", "pending", metadata={"room_id": f"o{n}"})
    room.kit.status_bus.post(
        "worker", "task", "info", detail="12 of 30", metadata={"room_id": "r", "task_id": task.id}
    )
    await asyncio.sleep(0)

    answer = await _cancel(CancelTaskTool(room.kit, window=6), task.id)
    await room.kit.close()

    assert answer["status"] == str(TaskStatus.CANCELLED)


async def test_a_cancel_with_no_tool_call_is_held_to_the_room_too() -> None:
    room = await _Room(_Endless()).open()
    await room.kit.create_room(room_id="other")
    await room.kit.attach_channel("other", "speaker", category=ChannelCategory.INTELLIGENCE)
    theirs = await room.running_task("other")
    room.kit.status_bus.post(
        "worker", "task", "pending", metadata={"room_id": "r", "task_id": theirs.id}
    )
    await asyncio.sleep(0)

    with tool_turn_context(room_id="r"):
        answer = json.loads(
            await CancelTaskTool(room.kit).handler(CANCEL_TASK_TOOL, {"task_id": theirs.id})
        )
    still_running = theirs.result is None
    await room.kit.close()

    assert answer["status"] == "unknown" and still_running


async def test_a_turn_s_cancel_task_is_served_by_the_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: list[str] = []
    serve = RegistryOwnerTools.serve

    async def spy(self: RegistryOwnerTools, request: Any, call: Any) -> Any:
        task = asyncio.current_task()
        served.append(f"{call.name}@{task.get_name() if task else ''}")
        return await serve(self, request, call)

    monkeypatch.setattr(RegistryOwnerTools, "serve", spy)
    room = await _Room(_Endless()).open()
    speaker = room.kit.channels["speaker"]
    room.kit.setup_task_tools(speaker)  # type: ignore[arg-type]
    task = await room.running_task()
    call = AIToolCall(id="c1", name=CANCEL_TASK_TOOL, arguments={"task_id": task.id})
    speaker._provider._ai_responses = [  # type: ignore[attr-defined]
        AIResponse(content="", tool_calls=[call]),
        AIResponse(content="Cancelled."),
    ]
    speaker._provider._index = 0  # type: ignore[attr-defined]

    await room.kit.process_inbound(
        InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="Cancel it.")),
        room_id="r",
    )
    result = await task.wait(timeout=5)  # set once its end ran, a hand-back included
    await room.kit.close()

    assert result.status == TaskStatus.CANCELLED
    assert any(s.startswith("cancel_task@roomkit-owner-call-") for s in served)
    assert room.handed_back == []


async def test_a_submit_that_fails_leaves_no_task_of_the_room() -> None:
    room = await _Room(_Endless()).open()

    async def down(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("down")

    room.kit._task_runner.submit = down  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await room.kit.delegate("r", "worker", "Count.", notify="speaker")
    left = room.kit._room_tasks.running
    await room.kit.close()

    assert left == []


class _Gated(MockAIProvider):
    """A worker that answers once released."""

    def __init__(self) -> None:
        super().__init__(responses=["Done."])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, context: AIContext) -> AIResponse:
        self.started.set()
        await self.release.wait()
        return AIResponse(content="Done.")


def _delegated(kit: RoomKit, door: str) -> asyncio.Future[Any]:
    if door == "inline":
        return asyncio.ensure_future(kit.delegate("r", "worker", "Count.", wait=True))
    return asyncio.ensure_future(kit.delegate("r", "worker", "Count.", notify=CALLER_HANDS_BACK))


@pytest.mark.parametrize("door", ["inline", "background"])
async def test_the_room_s_task_ending_while_cancel_task_reads_the_bus_reads_ended(
    door: str,
) -> None:
    """RMK-829: the room's own task ends while cancel_task reads its line,
    which still says running: it had already ended, with its status, never
    no task here."""
    room = await _Room(_Gated()).open()
    run = _delegated(room.kit, door)
    await asyncio.wait_for(room.worker.started.wait(), 5)
    (task_id,) = room.kit._room_tasks.running
    recent = room.kit.status_bus.recent

    async def ending_meanwhile(*args: Any, **kwargs: Any) -> Any:
        entries = await recent(*args, **kwargs)
        room.worker.release.set()
        await until(lambda: not room.kit._room_tasks.running)
        return entries

    room.kit.status_bus.recent = ending_meanwhile  # type: ignore[method-assign]
    answer = await _cancel(CancelTaskTool(room.kit), task_id)
    await asyncio.gather(run, return_exceptions=True)
    await room.kit.close()

    assert (answer["status"], answer["message"]) == ("completed", "The task had already ended.")


class _LaggingBackend(InMemoryStatusBackend):
    """A task's end lands on the bus a while after it is posted, as on a
    remote bus."""

    async def publish(self, entry: StatusEntry) -> None:
        if entry.action == "task" and entry.status != StatusLevel.PENDING:
            await asyncio.sleep(0.3)
        await super().publish(entry)


@pytest.mark.parametrize("door", ["inline", "background"])
async def test_the_room_s_task_ended_before_the_bus_shows_it_reads_ended(door: str) -> None:
    """RMK-829: the room's task ended, its end not on the bus yet: cancel_task
    answers it ended, with the status the framework recorded."""
    room = _Room(MockAIProvider(responses=["Done."]))
    room.kit = RoomKit(status_bus=StatusBus(backend=_LaggingBackend()))
    await room.open()
    await _delegated(room.kit, door)
    await until(lambda: not room.kit._room_tasks.running)
    (line,) = await room_task_lines(room.kit.status_bus, "r", 50)

    answer = await _cancel(CancelTaskTool(room.kit), line["task_id"])
    await room.kit.close()

    assert line["status"] == "running"
    assert (answer["status"], answer["message"]) == ("completed", "The task had already ended.")


async def test_past_the_bound_a_line_still_listed_running_is_no_task_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RMK-876: the room's task ended, the bus still lists it running, and
    more tasks than the record keeps ended in another room since: it reads
    as no task of the room (RFC §23.4)."""
    monkeypatch.setattr(room_tasks_module, "ENDED_ROOM_TASKS_KEPT", 2)
    room = _Room(MockAIProvider(responses=["Done."]))
    room.kit = RoomKit(status_bus=StatusBus(backend=_LaggingBackend()))
    await room.open()
    await _delegated(room.kit, "background")
    await until(lambda: not room.kit._room_tasks.running)
    (line,) = await room_task_lines(room.kit.status_bus, "r", 50)
    for n in range(2):
        room.kit._room_tasks.start(f"elsewhere-{n}", "other", "tool")
        room.kit._room_tasks.end(f"elsewhere-{n}", "completed")

    answer = await _cancel(CancelTaskTool(room.kit), line["task_id"])
    await room.kit.close()

    assert line["status"] == "running"
    assert (answer["status"], answer["message"]) == ("unknown", f"No task {line['task_id']} here.")


async def test_a_strategy_s_inline_worker_run_is_the_strategy_s() -> None:
    """RMK-829: a strategy's inline worker run, outside any turn, whose worker
    posts its progress: still running, the strategy's to follow, never said
    to run inside a turn."""
    room = await _Room(_Endless()).open()
    run = asyncio.ensure_future(
        run_worker(
            room.kit,
            "r",
            "worker",
            "Count.",
            timeout=None,
            status=WorkerStatus({"room_id": "r", "strategy": "sequential"}),
        )
    )
    await asyncio.wait_for(room.worker.started.wait(), 5)
    child = next(r.id for r in await room.kit.store.list_rooms() if r.id.startswith("r::task-"))
    with tool_turn_context(room_id=child):
        assert await post_task_progress(room.kit, "3 of 5")
    await asyncio.sleep(0)
    (task_id,) = room.kit._room_tasks.running

    answer = await _cancel(CancelTaskTool(room.kit), task_id)
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)
    await room.kit.close()

    assert answer["status"] == "running"
    assert "the strategy that runs it follows it" in answer["message"]


async def test_another_kit_s_task_on_a_shared_bus_is_no_task_here() -> None:
    """RMK-829: a task another kit runs for the room, on a bus both read, is
    not one this kit may answer for: no task here; it is cancelled where it
    runs."""
    bus = StatusBus()
    theirs = _Room(_Endless())
    theirs.kit = RoomKit(status_bus=bus)
    await theirs.open()
    ours = _Room(MockAIProvider(responses=["x"]))
    ours.kit = RoomKit(status_bus=bus)
    await ours.open()
    task = await theirs.running_task()

    answer = await _cancel(CancelTaskTool(ours.kit), task.id)
    await theirs.kit.cancel_task(task.id)
    await ours.kit.close()
    await theirs.kit.close()

    assert (answer["status"], answer["message"]) == ("unknown", f"No task {task.id} here.")


def test_the_record_keeps_the_latest_ended_tasks_only() -> None:
    record = RoomTaskRecord()
    for n in range(ENDED_ROOM_TASKS_KEPT + 1):
        record.start(f"t{n}", "r", "tool")
        record.end(f"t{n}", "completed")

    assert record.ended_with("t0", "r") is None
    assert record.ended_with(f"t{ENDED_ROOM_TASKS_KEPT}", "r") == "completed"
    assert record.ended_with(f"t{ENDED_ROOM_TASKS_KEPT}", "other") is None
    assert record.running == []


async def test_an_inline_task_that_never_completes_is_recorded_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RMK-829: an inline task whose run raises before its completion is
    recorded ended failed, never with the status it had while running."""
    room = await _Room(MockAIProvider(responses=["x"])).open()
    tasks: list[str] = []

    def broken(handle: Any, **kwargs: Any) -> Any:
        tasks.append(handle.id)
        raise RuntimeError("broken")

    monkeypatch.setattr(room.kit, "_child_turn", broken)
    with pytest.raises(RuntimeError):
        await room.kit.delegate("r", "worker", "Count.", wait=True)
    await room.kit.close()

    assert room.kit._room_tasks.ended_with(tasks[0], "r") == "failed"


@pytest.mark.parametrize("named", ["another kit's ended task", "a forged ended line"])
async def test_a_line_the_bus_lists_ended_is_answered_as_it_lists_it(named: str) -> None:
    """RMK-829: a task the room's line lists ended is answered as the bus
    lists it, as task_status does, whoever ran it: nothing is left to act on,
    and the line is the room's own."""
    bus = StatusBus()
    ours = _Room(MockAIProvider(responses=["x"]))
    ours.kit = RoomKit(status_bus=bus)
    await ours.open()
    theirs = _Room(MockAIProvider(responses=["Done."]))
    theirs.kit = RoomKit(status_bus=bus)
    await theirs.open()
    task_id = "task-never-run"
    if named == "another kit's ended task":
        task_id = (await theirs.kit.delegate("r", "worker", "Count.", wait=True)).id
    else:
        bus.post(
            "worker",
            "task",
            "completed",
            metadata={"room_id": "r", "task_id": task_id, "task_status": "completed"},
        )
    async with asyncio.timeout(5):
        while await _listed(bus, task_id) != "completed":
            await asyncio.sleep(0.01)

    answer = await _cancel(CancelTaskTool(ours.kit), task_id)
    await ours.kit.close()
    await theirs.kit.close()

    assert (answer["status"], answer["message"]) == ("completed", "The task had already ended.")


async def _listed(bus: StatusBus, task_id: str) -> str | None:
    """The status room r's line for *task_id* reads, ``None`` for no line."""
    lines = await room_task_lines(bus, "r", 50)
    return next((t.get("status") for t in lines if t["task_id"] == task_id), None)
