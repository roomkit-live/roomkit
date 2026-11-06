"""A regeneration runs its re-broadcast as an inbound event's runs, in the
room's lane, off the room lock and unbounded (RMK-525, RFC §13.5, §13.6).

``process_timeout`` bounds only the wait for the room lock and the choice of
the event to replay: a strategy turn slower than it goes to its end, the room
takes messages while the agent regenerates, and a lock held too long refuses
the regeneration with a result rather than a bare ``TimeoutError``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory
from roomkit.models.event import TextContent
from roomkit.models.framework_event import FrameworkEvent
from roomkit.orchestration.strategies.loop import Loop
from roomkit.providers.ai.base import AIContext, AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from tests.bounds import CUT, expire, opened_with_cut
from tests.conference.test_conference_realtime import until
from tests.test_framework import SimpleChannel

PROCESS_TIMEOUT = 0.2


class _Slow(MockAIProvider):
    """Answers after *pause*, longer than ``process_timeout``, or once *gate*
    is set when one is given; counts the generations it started."""

    def __init__(self, text: str = "draft", pause: float = 0.5) -> None:
        super().__init__(responses=[text], streaming=True)
        self.pause = pause
        self.started = 0
        self.gate: asyncio.Event | None = None

    async def generate(self, context: AIContext) -> AIResponse:
        self.started += 1
        if self.gate is not None:
            await self.gate.wait()
        else:
            await asyncio.sleep(self.pause)
        return await super().generate(context)


def _message(body: str = "Write.") -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body=body))


async def _kit(kind: str, pause: float = 0.5) -> tuple[RoomKit, list[str]]:
    kit, tasks, _ = await _kit_with(kind, pause)
    return kit, tasks


async def _kit_with(
    kind: str, pause: float = 0.5, process_timeout: float = PROCESS_TIMEOUT
) -> tuple[RoomKit, list[str], _Slow]:
    """A room whose agent answers after *pause*: a plain AI room, or a
    synchronous Loop whose producer is delegated."""
    kit = RoomKit(process_timeout=process_timeout)
    kit.register_channel(SimpleChannel("sms"))
    slow = _Slow(pause=pause)
    producer = Agent("producer", provider=slow)
    kit.register_channel(producer)
    tasks: list[str] = []

    @kit.hook(HookTrigger.ON_TASK_COMPLETED, execution=HookExecution.ASYNC)
    async def on_task(event: Any, context: Any) -> None:
        tasks.append(f"{event.metadata['agent_id']}:{event.metadata['task_status']}")

    if kind == "ai":
        await kit.create_room(room_id="r")
        await kit.attach_channel("r", "producer", category=ChannelCategory.INTELLIGENCE)
    else:
        reviewer = Agent("reviewer", provider=MockAIProvider(responses=["APPROVED"]))
        await kit.create_room(
            room_id="r", orchestration=Loop(agent=producer, reviewer=reviewer, max_iterations=2)
        )
    await kit.attach_channel("r", "sms")
    return kit, tasks, slow


@pytest.mark.parametrize("door", ["inbound", "regenerate"])
@pytest.mark.parametrize("kind", ["ai", "loop"])
async def test_a_turn_slower_than_process_timeout_goes_to_its_end(kind: str, door: str) -> None:
    # The turn under test takes longer than process_timeout (0.25 > 0.2 s);
    # the one a regeneration replays answers at once.
    kit, tasks, slow = await _kit_with(kind, pause=0.0 if door == "regenerate" else 0.25)
    result = await kit.process_inbound(_message())
    if door == "regenerate":
        tasks.clear()
        slow.pause = 0.25
        result = await kit.regenerate_response("r")
    await kit.close()  # the ASYNC task hook it fired runs to its end first

    assert result is not None and not result.blocked and result.error is None
    assert [e.content.body for e in result.response_events] == ["draft"]
    if kind == "loop":
        assert tasks == ["producer:completed", "reviewer:completed"]


async def test_the_room_takes_a_message_while_the_agent_regenerates() -> None:
    # A Loop runs its turn inside on_event, so the broadcast itself lasts as
    # long as the generation (a streamed AI reply is read after it).
    kit, _, slow = await _kit_with("loop", pause=0.0)
    await kit.process_inbound(_message("first"))
    slow.gate = asyncio.Event()

    regeneration = asyncio.create_task(kit.regenerate_response("r"))
    await until(lambda: slow.started == 2)  # the regenerated turn is generating
    # With the room lock held for the whole generation, this message would
    # wait for it and be refused at process_timeout.
    second = await kit.process_inbound(_message("second"), defer_delivery=True)
    slow.gate.set()
    regenerated = await regeneration
    assert second.delivery is not None
    await second.delivery.wait()
    await kit.close()

    assert not second.blocked and second.event is not None
    assert [e.content.body for e in regenerated.response_events] == ["draft"]


async def test_a_room_lock_held_past_process_timeout_refuses_the_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit, _, _ = await _kit_with("ai", pause=0.0, process_timeout=CUT)
    await kit.process_inbound(_message())
    timeouts: list[FrameworkEvent] = []

    @kit.on("process_timeout")
    async def on_timeout(event: FrameworkEvent) -> None:
        timeouts.append(event)

    held = asyncio.Event()
    release = asyncio.Event()

    async def hold_the_lock() -> None:
        async with kit._lock_manager.locked("r"):
            held.set()
            await release.wait()

    holder = asyncio.create_task(hold_the_lock())
    await held.wait()
    # Bounded here too: an unbounded wait for the lock would hang the test.
    async with asyncio.timeout(5.0):
        bounds = opened_with_cut(monkeypatch)
        regeneration = asyncio.ensure_future(kit.regenerate_response("r"))
        await until(lambda: bool(bounds))  # waiting for the lock, under process_timeout
        expire(bounds)
        result = await regeneration
    release.set()
    await holder
    await kit.close()

    assert result is not None
    assert (result.blocked, result.reason) == (True, "process_timeout")
    assert [(e.room_id, e.data.get("operation")) for e in timeouts] == [("r", "regenerate")]


async def test_a_regeneration_repeats_nothing_of_its_triggers_first_pass() -> None:
    kit, _ = await _kit("ai", pause=0.0)
    processed: list[str | None] = []
    after_broadcast: list[str] = []

    @kit.on("event_processed")
    async def on_processed(event: FrameworkEvent) -> None:
        processed.append(event.event_id)

    @kit.hook(HookTrigger.AFTER_BROADCAST, execution=HookExecution.ASYNC)
    async def observe(event: Any, context: Any) -> None:
        after_broadcast.append(event.content.body)

    first = await kit.process_inbound(_message())
    regenerated = await kit.regenerate_response("r")
    await kit.close()  # the ASYNC AFTER_BROADCAST hooks run to their end first

    assert first.event is not None and regenerated is not None
    assert processed.count(first.event.id) == 1
    # The trigger's own AFTER_BROADCAST ran once; each answer ran its own.
    assert after_broadcast == ["Write.", "draft", "draft"]


async def test_a_room_closed_while_the_regeneration_waits_refuses_it_before_the_agent_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit, _, slow = await _kit_with("loop", pause=0.0)
    queued = asyncio.Event()
    enqueue_regeneration = kit._enqueue_regeneration

    def enqueued(*args: Any, **kwargs: Any) -> Any:
        plan = enqueue_regeneration(*args, **kwargs)
        queued.set()
        return plan

    monkeypatch.setattr(kit, "_enqueue_regeneration", enqueued)
    refused: list[FrameworkEvent] = []

    @kit.on("room_refused_event")
    async def on_refused(event: FrameworkEvent) -> None:
        refused.append(event)

    await kit.process_inbound(_message("first"))
    slow.gate = asyncio.Event()
    # A second message keeps the lane busy: the regeneration waits behind it.
    second = await kit.process_inbound(_message("second"), defer_delivery=True)
    await until(lambda: slow.started == 2)  # the second message's turn generates
    regeneration = asyncio.create_task(kit.regenerate_response("r"))
    await asyncio.wait_for(queued.wait(), 1.0)  # the regeneration takes its place behind it
    await kit.close_room("r")
    started_at_close = slow.started
    slow.gate.set()
    result = await asyncio.wait_for(regeneration, timeout=5.0)
    assert second.delivery is not None
    await second.delivery.wait()
    await kit.close()

    assert result is not None
    assert (result.blocked, result.reason) == (True, "room_closed")
    assert slow.started == started_at_close
    assert "regenerate" in [e.data.get("operation") for e in refused]


@pytest.mark.parametrize("door", ["regenerate", "send_event"])
async def test_a_caller_cancelled_cuts_the_turn_it_was_waiting_for(door: str) -> None:
    kit, tasks, slow = await _kit_with("loop", pause=0.0)
    await kit.process_inbound(_message())
    await kit.hook_engine.drain_observers()  # the first turn's task hooks are in
    tasks.clear()
    slow.pause = 0.5  # the call's turn answers after this, had it run on

    if door == "regenerate":
        call = asyncio.create_task(kit.regenerate_response("r"))
    else:
        call = asyncio.create_task(kit.send_event("r", "sms", TextContent(body="Again.")))
    await until(lambda: slow.started == 2)  # the producer is generating
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    await until(lambda: bool(tasks))  # how the producer's task ended: cut, or answered
    await kit.close()

    assert tasks == ["producer:cancelled"]


async def test_a_room_read_past_process_timeout_refuses_the_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``process_timeout`` bounds the regeneration's choice too, its read of
    the room under the lock (RFC §13.6): a read that outlasts it refuses the
    call ``process_timeout`` with nothing run, and the lock is released."""
    kit, _, slow = await _kit_with("ai", pause=0.0, process_timeout=CUT)
    await kit.process_inbound(_message())
    timeouts: list[FrameworkEvent] = []

    @kit.on("process_timeout")
    async def on_timeout(event: FrameworkEvent) -> None:
        timeouts.append(event)

    reading = asyncio.Event()
    build_context = kit._build_context

    async def slow_read(room_id: str, **kwargs: Any) -> Any:
        if kwargs.get("reads_history"):
            reading.set()
            await asyncio.Event().wait()  # never answers; the bound cuts it
        return await build_context(room_id, **kwargs)

    bounds = opened_with_cut(monkeypatch)
    monkeypatch.setattr(kit, "_build_context", slow_read)
    async with asyncio.timeout(5.0):
        regeneration = asyncio.ensure_future(kit.regenerate_response("r"))
        await reading.wait()  # the room is read under the lock, within the bound
        expire(bounds)
        result = await regeneration
    monkeypatch.setattr(kit, "_build_context", build_context)
    # The lock is free again: the next message goes through.
    after = await kit.process_inbound(_message("again"))
    await kit.close()
    assert result is not None
    assert (result.blocked, result.reason) == (True, "process_timeout")
    assert [(e.room_id, e.data.get("operation")) for e in timeouts] == [("r", "regenerate")]
    assert slow.started == 2  # the first message's and "again"'s turns, none regenerated
    assert not after.blocked
