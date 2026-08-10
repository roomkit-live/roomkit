"""`process_timeout` bounds the whole pre-commit phase (RFC §13.6).

It bounded one region of it — the gates inside the room lock — and the setting
read as a promise it did not keep. Everything before the lock ran unbounded: the
context build, and `handle_inbound`, which is integrator code and can reach a
provider with no timeout of its own. So could the wait for the lock, which is
the pile-up the setting exists to stop: one stuck event holds a room's lock and
every later message queues behind it forever.

A store that hangs is not exotic — an exhausted connection pool and a network
that swallows packets without a reset both look exactly like this.
"""

from __future__ import annotations

import asyncio

from roomkit import RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import RoomEvent, TextContent
from roomkit.store.memory import InMemoryStore
from tests.test_framework import SimpleChannel

FAST = 0.2


class HangingStore(InMemoryStore):
    """Healthy until armed, then one read never returns."""

    armed = False

    async def list_bindings(self, room_id: str):  # noqa: ANN201
        if self.armed:
            await asyncio.sleep(3600)
        return await super().list_bindings(room_id)


class HangingChannel(SimpleChannel):
    """A parser that reaches a provider without a timeout of its own."""

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


async def _kit(channel: SimpleChannel, store: InMemoryStore | None = None) -> RoomKit:
    kit = RoomKit(store=store, process_timeout=FAST) if store else RoomKit(process_timeout=FAST)
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    return kit


def _msg() -> InboundMessage:
    return InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="hi"))


async def _within_budget(kit: RoomKit):  # noqa: ANN202
    """Fails loudly rather than hanging the suite if the bound is gone."""
    return await asyncio.wait_for(kit.process_inbound(_msg()), timeout=5.0)


class TestTheWindowCoversWhatItClaims:
    async def test_a_store_that_hangs_during_the_context_build(self) -> None:
        store = HangingStore()
        kit = await _kit(SimpleChannel("sms1"), store)
        store.armed = True

        result = await _within_budget(kit)

        assert result.blocked is True
        assert result.reason == "process_timeout"

    async def test_a_channel_that_hangs_in_handle_inbound(self) -> None:
        kit = await _kit(HangingChannel("sms1"))

        result = await _within_budget(kit)

        assert result.blocked is True
        assert result.reason == "process_timeout"

    async def test_a_room_lock_held_by_something_stuck(self) -> None:
        """The pile-up: without this bound, every later message queues forever."""
        kit = await _kit(SimpleChannel("sms1"))

        async def squatter() -> None:
            async with kit._lock_manager.locked("r1"):  # noqa: SLF001
                await asyncio.sleep(3600)

        holder = asyncio.create_task(squatter())
        await asyncio.sleep(0.05)
        try:
            result = await _within_budget(kit)
        finally:
            holder.cancel()

        assert result.blocked is True
        assert result.reason == "process_timeout"

    async def test_a_hook_that_hangs_inside_the_lock_still_times_out(self) -> None:
        """The region that already worked keeps working."""
        from roomkit import HookTrigger

        kit = await _kit(SimpleChannel("sms1"))

        @kit.hook(HookTrigger.BEFORE_BROADCAST)
        async def slow(event, ctx):  # noqa: ANN001, ANN202
            await asyncio.sleep(3600)

        result = await _within_budget(kit)

        assert result.blocked is True
        assert result.reason == "process_timeout"


class TestTheWindowStopsWhereItShould:
    async def test_a_healthy_message_commits(self) -> None:
        kit = await _kit(SimpleChannel("sms1"))

        result = await _within_budget(kit)
        timeline = await kit.get_timeline("r1")

        assert result.blocked is False
        assert [e.id for e in timeline if e.id == result.event.id] == [result.event.id]

    async def test_a_refused_message_commits_nothing(self) -> None:
        """§13.6 forbids the converse too: no committed event reported blocked."""
        kit = await _kit(HangingChannel("sms1"))

        result = await _within_budget(kit)
        timeline = await kit.get_timeline("r1")

        assert result.blocked is True
        assert [getattr(e.content, "body", None) for e in timeline].count("hi") == 0

    async def test_the_budget_is_spent_once_across_both_regions(self) -> None:
        """A deadline, not the setting re-applied: 0.2s means 0.2s, not 0.4s."""
        store = HangingStore()
        kit = await _kit(SimpleChannel("sms1"), store)
        store.armed = True

        started = asyncio.get_running_loop().time()
        await _within_budget(kit)
        elapsed = asyncio.get_running_loop().time() - started

        assert elapsed < FAST * 2
