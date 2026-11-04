"""A status naming a room fires that room's ON_STATUS_POSTED hooks on a kit used
without ``async with`` (RFC §19.8, RMK-541).

The framework listened to its StatusBus only once opened as a context manager, or
after a ``send_event`` or a WebSocket registration: a kit driven by
``create_room`` / ``process_inbound`` / ``delegate`` posted its entries to the bus
and no room hook ever heard of them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from roomkit import (
    HookExecution,
    HookTrigger,
    InboundMessage,
    KitClosedError,
    RoomKit,
    TextContent,
)
from roomkit.channels.agent import Agent
from roomkit.models.enums import ChannelCategory, RoomStatus
from roomkit.models.event import EventSource, RoomEvent
from roomkit.models.room import Room
from roomkit.orchestration import InMemoryStatusBackend, StatusBus
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


class _LateBackend(InMemoryStatusBackend):
    """A backend whose subscription waits to be answered."""

    def __init__(self) -> None:
        super().__init__()
        self.asked = asyncio.Event()
        self.answer = asyncio.Event()

    async def subscribe(self, callback: Any) -> None:
        self.asked.set()
        await self.answer.wait()
        await super().subscribe(callback)


def _late_kit(backend: _LateBackend, **kwargs: Any) -> RoomKit:
    kit = RoomKit(status_bus=StatusBus(backend=backend), **kwargs)
    kit.register_channel(SimpleChannel("sms"))
    return kit


def _message() -> InboundMessage:
    return InboundMessage(channel_id="sms", sender_id="u1", content=TextContent(body="hi"))


@pytest.mark.parametrize("deferred", [False, True], ids=["awaited", "deferred"])
@pytest.mark.parametrize("answered", [False, True], ids=["past-the-grace", "within-the-grace"])
async def test_a_message_waiting_for_the_subscription_is_cut_by_the_close(
    answered: bool, deferred: bool, close_grace: float
) -> None:
    # Within the close's grace or past it, the call the wait held before its
    # room's lock is cut: nothing of it is stored, no handle for a deferred one.
    backend = _LateBackend()
    kit = _late_kit(backend)
    call = asyncio.ensure_future(kit.process_inbound(_message(), defer_delivery=deferred))
    await backend.asked.wait()

    closing = asyncio.ensure_future(kit.close())
    await asyncio.sleep(0)
    if answered:
        backend.answer.set()
    result = await call
    await closing

    assert (
        result.blocked,
        result.reason,
        result.cancellation_reason,
        result.event,
        result.delivery,
    ) == (True, "kit_closed", "kit_closed", None, None)


async def _seeded(kit: RoomKit) -> None:
    """A room ``r`` the store holds, with two events ``e0`` and ``e1``: no call
    asked for the subscription yet."""
    await kit.store.create_room(Room(id="r"))
    source = EventSource(channel_id="sms", channel_type="sms")
    for index in range(2):
        await kit.store.add_event(
            RoomEvent(
                id=f"e{index}",
                room_id="r",
                index=index,
                source=source,
                content=TextContent(body="hi"),
            )
        )


_CHANGES: dict[str, Callable[[RoomKit], Awaitable[Any]]] = {
    "create_room": lambda kit: kit.create_room(room_id="new"),
    "pause_room": lambda kit: kit.pause_room("r"),
    "close_room": lambda kit: kit.close_room("r"),
    "attach_channel": lambda kit: kit.attach_channel("r", "sms"),
    "update_event": lambda kit: kit.update_event("r", "e0", content=TextContent(body="new")),
    "delete_event": lambda kit: kit.delete_event("r", "e0"),
}


@pytest.mark.parametrize("change", _CHANGES)
async def test_a_call_waiting_for_the_subscription_goes_on_within_the_close_s_grace(
    change: str, close_grace: float
) -> None:
    backend = _LateBackend()
    kit = _late_kit(backend)
    await _seeded(kit)

    async def noted(event: RoomEvent, context: object) -> None:
        return None

    kit.hook(HookTrigger.ON_EVENT_UPDATED, HookExecution.ASYNC)(noted)
    kit.hook(HookTrigger.ON_EVENT_DELETED, HookExecution.ASYNC, name="deleted")(noted)
    call = asyncio.ensure_future(_CHANGES[change](kit))
    await asyncio.wait_for(backend.asked.wait(), 1)

    closing = asyncio.ensure_future(kit.close())
    await asyncio.sleep(0)
    backend.answer.set()
    await asyncio.wait_for(call, 1)
    await closing

    store, room = kit.store, await kit.store.get_room("r")
    written = {
        "create_room": await store.get_room("new") is not None,
        "pause_room": room.status == RoomStatus.PAUSED,
        "close_room": room.status == RoomStatus.CLOSED,
        "attach_channel": await store.get_binding("r", "sms") is not None,
        "update_event": change == "update_event"
        and (await store.get_event("e0")).content.body == "new",
        "delete_event": await store.get_event("e0") is None,
    }
    assert written[change]


async def test_a_call_waiting_for_the_subscription_is_cut_past_the_close_s_grace(
    close_grace: float,
) -> None:
    backend = _LateBackend()
    kit = _late_kit(backend)
    call = asyncio.ensure_future(kit.create_room(room_id="past"))
    await backend.asked.wait()

    await kit.close()

    with pytest.raises(KitClosedError) as cut:
        await call
    # It wrote before it asked for its context: the cut says nothing of that.
    assert await kit.store.get_room("past") is not None
    assert "stored" not in str(cut.value)


async def test_a_change_of_an_event_subscribes_only_for_a_hook_that_reads_its_context() -> None:
    backend = _LateBackend()
    kit = _late_kit(backend)
    await _seeded(kit)

    await asyncio.wait_for(kit.update_event("r", "e0", content=TextContent(body="changed")), 1)
    await asyncio.wait_for(kit.delete_event("r", "e1"), 1)
    assert not backend.asked.is_set()

    async def noted(event: RoomEvent, context: object) -> None:
        return None

    kit.hook(HookTrigger.ON_EVENT_DELETED, HookExecution.ASYNC)(noted)
    call = asyncio.ensure_future(kit.delete_event("r", "e0"))
    await asyncio.wait_for(backend.asked.wait(), 1)
    backend.answer.set()
    assert await asyncio.wait_for(call, 1) == ["e0"]
    await kit.close()
