"""A status naming a room fires that room's ON_STATUS_POSTED hooks on a kit used
without ``async with`` (RFC §19.8, RMK-541).

The framework listened to its StatusBus only once opened as a context manager, or
after a ``send_event`` or a WebSocket registration: a kit driven by
``create_room`` / ``process_inbound`` / ``delegate`` posted its entries to the bus
and no room hook ever heard of them.
"""

from __future__ import annotations

import asyncio

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.enums import ChannelCategory
from roomkit.orchestration.status_bus import StatusEntry, StatusLevel
from roomkit.providers.ai.mock import MockAIProvider
from tests.conference.test_conference_realtime import until
from tests.test_framework import SimpleChannel


def _listen(kit: RoomKit) -> list[StatusEntry]:
    heard: list[StatusEntry] = []

    @kit.hook(HookTrigger.ON_STATUS_POSTED, HookExecution.ASYNC)
    async def on_status(entry: StatusEntry, context: object) -> None:
        heard.append(entry)

    return heard


async def test_a_delegation_on_a_plain_kit_fires_the_rooms_status_hooks() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    kit.register_channel(Agent("speaker", provider=MockAIProvider(responses=["ok"])))
    kit.register_channel(Agent("worker", provider=MockAIProvider(responses=["8°C."])))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "speaker", category=ChannelCategory.INTELLIGENCE)
    heard = _listen(kit)

    task = await kit.delegate("r", "worker", "La météo ?", notify="speaker")
    await task.wait()
    await until(lambda: len(heard) == 2, timeout=5)
    await kit.close()

    assert [e.status for e in heard] == [StatusLevel.PENDING, StatusLevel.COMPLETED]


async def test_a_status_posted_after_room_activity_reaches_the_rooms_hooks() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "sms")
    heard = _listen(kit)

    await kit.status_bus.post_async(
        "agent-1", "working", StatusLevel.PENDING, metadata={"room_id": "r"}
    )
    await asyncio.sleep(0.05)
    await kit.close()

    assert [e.agent_id for e in heard] == ["agent-1"]


async def test_first_activities_racing_subscribe_once() -> None:
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms"))
    await kit.create_room(room_id="r")
    await asyncio.gather(*(kit._build_context("r") for _ in range(5)))  # noqa: SLF001
    heard = _listen(kit)

    await kit.status_bus.post_async(
        "agent-1", "working", StatusLevel.PENDING, metadata={"room_id": "r"}
    )
    await asyncio.sleep(0.05)
    await kit.close()

    assert len(heard) == 1
