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
from tests.test_framework import SimpleChannel

PROCESS_TIMEOUT = 0.2


class _Slow(MockAIProvider):
    """Answers after *pause*, longer than ``process_timeout``."""

    def __init__(self, text: str = "draft", pause: float = 0.5) -> None:
        super().__init__(responses=[text], streaming=True)
        self.pause = pause

    async def generate(self, context: AIContext) -> AIResponse:
        await asyncio.sleep(self.pause)
        return await super().generate(context)


def _message(body: str = "Write.") -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body=body))


async def _kit(kind: str, pause: float = 0.5) -> tuple[RoomKit, list[str]]:
    """A room whose agent answers after *pause*: a plain AI room, or a
    synchronous Loop whose producer is delegated."""
    kit = RoomKit(process_timeout=PROCESS_TIMEOUT)
    kit.register_channel(SimpleChannel("sms"))
    producer = Agent("producer", provider=_Slow(pause=pause))
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
    return kit, tasks


@pytest.mark.parametrize("door", ["inbound", "regenerate"])
@pytest.mark.parametrize("kind", ["ai", "loop"])
async def test_a_turn_slower_than_process_timeout_goes_to_its_end(kind: str, door: str) -> None:
    kit, tasks = await _kit(kind)
    result = await kit.process_inbound(_message())
    if door == "regenerate":
        tasks.clear()
        result = await kit.regenerate_response("r")
    await asyncio.sleep(0.1)
    await kit.close()

    assert result is not None and not result.blocked and result.error is None
    assert [e.content.body for e in result.response_events] == ["draft"]
    if kind == "loop":
        assert tasks == ["producer:completed", "reviewer:completed"]


async def test_the_room_takes_a_message_while_the_agent_regenerates() -> None:
    # A Loop runs its turn inside on_event, so the broadcast itself lasts as
    # long as the generation (a streamed AI reply is read after it).
    kit, _ = await _kit("loop", pause=0.3)
    await kit.process_inbound(_message("first"))

    regeneration = asyncio.create_task(kit.regenerate_response("r"))
    await asyncio.sleep(0.05)  # the regenerated turn is generating
    # With the room lock held for the whole generation, this message would
    # wait for it and be refused at process_timeout.
    second = await kit.process_inbound(_message("second"), defer_delivery=True)
    regenerated = await regeneration
    assert second.delivery is not None
    await second.delivery.wait()
    await kit.close()

    assert not second.blocked and second.event is not None
    assert [e.content.body for e in regenerated.response_events] == ["draft"]


async def test_a_room_lock_held_past_process_timeout_refuses_the_regeneration() -> None:
    kit, _ = await _kit("ai", pause=0.0)
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
    result = await kit.regenerate_response("r")
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
    await asyncio.sleep(0.05)
    await kit.close()

    assert first.event is not None and regenerated is not None
    assert processed.count(first.event.id) == 1
    # The trigger's own AFTER_BROADCAST ran once; each answer ran its own.
    assert after_broadcast == ["Write.", "draft", "draft"]
