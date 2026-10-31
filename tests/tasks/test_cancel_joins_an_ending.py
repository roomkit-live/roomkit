"""A ``cancel_task`` asked while another cancel of the same task ends: it is
one of that cancel's calls only while the task ended ``cancelled``, the call
could have cancelled it running, and the end has not decided whom it hands
the cancellation back to; otherwise it is answered that the task ended
(RFC §23.4; RMK-910)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from roomkit import HookResult, HookTrigger
from roomkit.console._discussion import DiscussionConsole
from roomkit.models.enums import EventType, TaskStatus
from roomkit.orchestration.status_bus import StatusLevel
from roomkit.tasks import CancelTaskTool
from roomkit.tasks.handback import CALLER_HANDS_BACK
from tests.conference.test_conference_realtime import until
from tests.tasks.test_cancel_task import _cancel, _Endless, _Gated, _Room

_CANCELLED = "Cancelled: its result will not come back."
_ENDED = "The task had already ended."


def _held_end(room: _Room) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold a task's end being completed until released, at its read of the
    room it tells ``ON_TASK_COMPLETED`` in, a store slow to answer: whether
    the end began, and its release. Its hooks would hold nothing: they are
    started and not waited for (RFC §9.3)."""
    completing, release = asyncio.Event(), asyncio.Event()
    store = room.kit.store
    read = store.load_room_context

    async def held(room_id: str) -> Any:
        # The first read of the room once a task of it ended: its end's.
        if room_id == "r" and room.kit._room_tasks._ended and not release.is_set():
            completing.set()
            await release.wait()
        return await read(room_id)

    store.load_room_context = held  # type: ignore[method-assign]
    return completing, release


async def test_an_agent_cancelling_as_the_host_s_cancel_ends_joins_it() -> None:
    """The host's cancel ended the task, its end still being completed, when
    the agent the result was going to cancels it too: the agent's call joins
    that cancel, answered as the host's is, and is not handed it back."""
    room = await _Room(_Endless()).open()
    task = await room.running_task()
    completing, release = _held_end(room)

    host = asyncio.ensure_future(room.kit.cancel_task(task.id))
    await asyncio.wait_for(completing.wait(), 5)
    agent = asyncio.ensure_future(_cancel(CancelTaskTool(room.kit), task.id))
    await until(lambda: "speaker" in room.kit._task_cancels[task.id].callers)
    release.set()
    answers = await asyncio.gather(host, agent)
    await asyncio.sleep(0.1)  # a hand-back would be under way by now
    await room.kit.close()

    assert answers == [True, {"task_id": task.id, "status": "cancelled", "message": _CANCELLED}]
    assert [e["task_status"] for e in room.ended] == [TaskStatus.CANCELLED]
    assert room.kit._task_cancels == {}
    assert room.handed_back == []


async def test_a_task_that_completed_as_the_host_cancelled_is_answered_at_once() -> None:
    """The task completed on its own while the host's cancel waited for its
    end: an agent's call is answered that it completed, at once, not after
    the end's hooks and hand-back."""
    room = await _Room(_Gated()).open()
    completing, release = _held_end(room)
    task = await room.kit.delegate("r", "worker", "Count.", notify="speaker")
    await asyncio.wait_for(room.worker.started.wait(), 5)
    room.worker.release.set()
    await asyncio.wait_for(completing.wait(), 5)
    host = asyncio.ensure_future(room.kit.cancel_task(task.id))
    await until(lambda: task.id in room.kit._task_cancels)

    answer = await asyncio.wait_for(_cancel(CancelTaskTool(room.kit), task.id), 1)
    release.set()
    await asyncio.wait_for(host, 5)
    await room.kit.close()

    assert answer == {"task_id": task.id, "status": "completed", "message": _ENDED}


async def test_a_call_after_the_hand_back_was_decided_is_answered_ended() -> None:
    """The end of the host's cancel has decided its hand-back (under way to
    the agent) while the bus, lagging, still lists the task running: the
    agent's call is answered that the task ended, and is handed it back."""
    room = await _Room(_Endless()).open()
    backend = room.kit.status_bus._backend
    publish, flush = backend.publish, asyncio.Event()

    async def lagging(entry: Any) -> None:
        if entry.status in (StatusLevel.COMPLETED, StatusLevel.FAILED):
            await flush.wait()
        await publish(entry)

    backend.publish = lagging
    handing_back, go_on = asyncio.Event(), asyncio.Event()

    @room.kit.hook(HookTrigger.BEFORE_BROADCAST, event_types={EventType.INSTRUCTION})
    async def held(event: Any, ctx: Any) -> HookResult:
        handing_back.set()
        await go_on.wait()
        return HookResult.allow()

    task = await room.running_task()
    host = asyncio.ensure_future(room.kit.cancel_task(task.id))
    await asyncio.wait_for(handing_back.wait(), 5)
    answer = await asyncio.wait_for(_cancel(CancelTaskTool(room.kit), task.id), 1)
    go_on.set()
    await asyncio.wait_for(host, 5)
    flush.set()
    await until(lambda: bool(room.handed_back))
    await room.kit.close()

    assert answer == {"task_id": task.id, "status": "cancelled", "message": _ENDED}
    assert [h["task_status"] for h in room.handed_back] == ["cancelled"]


async def test_a_strategy_s_task_the_host_cancels_is_not_the_agent_s_to_join() -> None:
    """A task a strategy runs without posting it is not the room's agents' to
    cancel: an agent's call made as the host's cancel of it ends is answered
    that it ended, never that it cancelled it."""
    room = await _Room(_Endless()).open()
    completing, release = _held_end(room)
    task = await room.kit.delegate(
        "r", "worker", "Count.", notify=CALLER_HANDS_BACK, post_status=False
    )
    await asyncio.wait_for(room.worker.started.wait(), 5)
    room.kit.status_bus.post(
        "worker",
        "task",
        "info",
        detail="1/3",
        metadata={"room_id": "r", "task_id": task.id, "child_room_id": task.child_room_id},
    )
    host = asyncio.ensure_future(room.kit.cancel_task(task.id))
    await asyncio.wait_for(completing.wait(), 5)

    answer = await asyncio.wait_for(_cancel(CancelTaskTool(room.kit), task.id), 1)
    release.set()
    await asyncio.wait_for(host, 5)
    await room.kit.close()

    assert answer == {"task_id": task.id, "status": "cancelled", "message": _ENDED}


async def test_the_console_joins_a_cancel_being_completed_as_an_agent_does() -> None:
    """The person's ``/cancel`` in the console, asked as an agent's cancel
    ends, is decided as another agent's call is: both say cancelled."""
    room = await _Room(_Endless()).open()
    completing, release = _held_end(room)
    task = await room.running_task()
    first = asyncio.ensure_future(_cancel(CancelTaskTool(room.kit), task.id))
    await asyncio.wait_for(completing.wait(), 5)
    console = SimpleNamespace(_kit=room.kit, _room_id="r")

    said = asyncio.ensure_future(DiscussionConsole._cancelled(console, task.id))  # type: ignore[arg-type]
    other = asyncio.ensure_future(_cancel(CancelTaskTool(room.kit), task.id, channel_id="other"))
    await until(lambda: "other" in room.kit._task_cancels[task.id].callers)
    release.set()
    answers = await asyncio.gather(said, other, first)
    await room.kit.close()

    assert answers[0].endswith("cancelled")
    assert answers[1]["message"] == _CANCELLED
