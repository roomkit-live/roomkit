"""BEFORE_DELIVER decides (RFC §9.2, §22.3).

The trigger is SYNC and documented as "can block/modify content". Both firing
sites ran it through `run_async_hooks`, so a moderation hook's `block()` was
discarded and its exceptions swallowed at debug — the delivery went out either
way, and nothing said so.
"""

from __future__ import annotations

from roomkit import HookTrigger, RoomKit
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult
from tests.test_framework import SimpleChannel


async def _room_with_transport() -> tuple[RoomKit, SimpleChannel]:
    kit = RoomKit()
    channel = SimpleChannel("sms1")
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    return kit, channel


async def _delivered_bodies(kit: RoomKit) -> list[str]:
    """What reached the room. `kit.deliver()` injects a synthetic inbound
    message, so the timeline — not the source channel, which rule 5 skips — is
    where a delivery shows up."""
    return [
        e.content.body for e in await kit.get_timeline("r1") if isinstance(e.content, TextContent)
    ]


class TestBlocking:
    async def test_a_blocking_hook_stops_the_delivery(self) -> None:
        kit, _channel = await _room_with_transport()

        @kit.hook(HookTrigger.BEFORE_DELIVER, name="moderation")
        async def block_it(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.block("policy: no proactive nudges")

        await kit.deliver("r1", "buy now")

        assert await _delivered_bodies(kit) == []

    async def test_an_allowing_hook_lets_it_through(self) -> None:
        kit, _channel = await _room_with_transport()

        @kit.hook(HookTrigger.BEFORE_DELIVER, name="permissive")
        async def allow_it(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.allow()

        await kit.deliver("r1", "your table is ready")

        assert await _delivered_bodies(kit) == ["your table is ready"]

    async def test_no_hook_delivers_as_before(self) -> None:
        kit, _channel = await _room_with_transport()
        await kit.deliver("r1", "unfiltered")
        assert await _delivered_bodies(kit) == ["unfiltered"]


class TestRewriting:
    async def test_a_hook_can_rewrite_what_goes_out(self) -> None:
        kit, _channel = await _room_with_transport()

        @kit.hook(HookTrigger.BEFORE_DELIVER, name="redactor")
        async def redact(event: RoomEvent, ctx: RoomContext) -> HookResult:
            body = event.content.body.replace("4111111111111111", "[redacted]")
            return HookResult.modify(event.model_copy(update={"content": TextContent(body=body)}))

        await kit.deliver("r1", "your card 4111111111111111 was charged")

        assert await _delivered_bodies(kit) == ["your card [redacted] was charged"]


class TestFailOpen:
    async def test_a_raising_hook_does_not_stop_the_delivery(self) -> None:
        """BEFORE_DELIVER is not a fail-closed trigger (RFC §9.3): a broken
        hook must not silently swallow the caller's message."""
        kit, _channel = await _room_with_transport()

        @kit.hook(HookTrigger.BEFORE_DELIVER, name="broken")
        async def broken(event: RoomEvent, ctx: RoomContext) -> HookResult:
            raise RuntimeError("moderation service down")

        await kit.deliver("r1", "still goes out")

        assert await _delivered_bodies(kit) == ["still goes out"]
