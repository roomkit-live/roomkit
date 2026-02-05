"""Tests for HookEngine."""

from __future__ import annotations

import asyncio

from roomkit.core.hooks import HookEngine, HookRegistration
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.hook import HookResult, InjectedEvent
from roomkit.models.room import Room
from tests.conftest import make_event


def _ctx() -> RoomContext:
    return RoomContext(room=Room(id="r1"))


class TestSyncHooks:
    async def test_allow_passes(self) -> None:
        engine = HookEngine()

        async def allow_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.allow()

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=allow_hook,
                name="allow",
            )
        )
        result = await engine.run_sync_hooks(
            "r1", HookTrigger.BEFORE_BROADCAST, make_event(), _ctx()
        )
        assert result.allowed is True

    async def test_block_stops_pipeline(self) -> None:
        engine = HookEngine()
        call_order: list[str] = []

        async def block_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            call_order.append("block")
            return HookResult.block("spam")

        async def after_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            call_order.append("after")
            return HookResult.allow()

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=block_hook,
                priority=0,
                name="blocker",
            )
        )
        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=after_hook,
                priority=1,
                name="after",
            )
        )
        result = await engine.run_sync_hooks(
            "r1", HookTrigger.BEFORE_BROADCAST, make_event(), _ctx()
        )
        assert result.allowed is False
        assert result.reason == "spam"
        assert call_order == ["block"]

    async def test_modify_passes_to_next(self) -> None:
        engine = HookEngine()

        async def modify_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            modified = event.model_copy(update={"content": TextContent(body="modified")})
            return HookResult.modify(modified)

        async def check_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            assert isinstance(event.content, TextContent)
            assert event.content.body == "modified"
            return HookResult.allow()

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=modify_hook,
                priority=0,
                name="modifier",
            )
        )
        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=check_hook,
                priority=1,
                name="checker",
            )
        )
        result = await engine.run_sync_hooks(
            "r1", HookTrigger.BEFORE_BROADCAST, make_event(), _ctx()
        )
        assert result.allowed is True
        assert result.event is not None
        assert isinstance(result.event.content, TextContent)
        assert result.event.content.body == "modified"

    async def test_injected_events_collected(self) -> None:
        engine = HookEngine()

        async def inject_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            notice = make_event(body="notice")
            return HookResult.allow(injected=[InjectedEvent(event=notice)])

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=inject_hook,
                name="injector",
            )
        )
        result = await engine.run_sync_hooks(
            "r1", HookTrigger.BEFORE_BROADCAST, make_event(), _ctx()
        )
        assert len(result.injected_events) == 1

    async def test_priority_ordering(self) -> None:
        engine = HookEngine()
        call_order: list[int] = []

        for prio in [2, 0, 1]:

            async def hook(event: RoomEvent, ctx: RoomContext, p: int = prio) -> HookResult:
                call_order.append(p)
                return HookResult.allow()

            engine.register(
                HookRegistration(
                    trigger=HookTrigger.BEFORE_BROADCAST,
                    execution=HookExecution.SYNC,
                    fn=hook,
                    priority=prio,
                    name=f"prio-{prio}",
                )
            )

        await engine.run_sync_hooks("r1", HookTrigger.BEFORE_BROADCAST, make_event(), _ctx())
        assert call_order == [0, 1, 2]

    async def test_error_in_sync_hook_continues(self) -> None:
        engine = HookEngine()

        async def bad_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            raise RuntimeError("oops")

        async def good_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.allow()

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=bad_hook,
                priority=0,
                name="bad",
            )
        )
        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=good_hook,
                priority=1,
                name="good",
            )
        )
        result = await engine.run_sync_hooks(
            "r1", HookTrigger.BEFORE_BROADCAST, make_event(), _ctx()
        )
        assert result.allowed is True

    async def test_room_hooks_merged(self) -> None:
        engine = HookEngine()
        call_order: list[str] = []

        async def global_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            call_order.append("global")
            return HookResult.allow()

        async def room_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            call_order.append("room")
            return HookResult.allow()

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=global_hook,
                priority=0,
                name="global",
            )
        )
        engine.add_room_hook(
            "r1",
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=room_hook,
                priority=1,
                name="room",
            ),
        )
        await engine.run_sync_hooks("r1", HookTrigger.BEFORE_BROADCAST, make_event(), _ctx())
        assert call_order == ["global", "room"]


class TestAsyncHooks:
    async def test_async_hooks_run_concurrently(self) -> None:
        engine = HookEngine()
        results: list[str] = []

        async def hook_a(event: RoomEvent, ctx: RoomContext) -> None:
            await asyncio.sleep(0.01)
            results.append("a")

        async def hook_b(event: RoomEvent, ctx: RoomContext) -> None:
            results.append("b")

        engine.register(
            HookRegistration(
                trigger=HookTrigger.AFTER_BROADCAST,
                execution=HookExecution.ASYNC,
                fn=hook_a,
                name="hook_a",
            )
        )
        engine.register(
            HookRegistration(
                trigger=HookTrigger.AFTER_BROADCAST,
                execution=HookExecution.ASYNC,
                fn=hook_b,
                name="hook_b",
            )
        )
        await engine.run_async_hooks("r1", HookTrigger.AFTER_BROADCAST, make_event(), _ctx())
        assert set(results) == {"a", "b"}

    async def test_async_timeout_isolation(self) -> None:
        engine = HookEngine()
        results: list[str] = []

        async def slow_hook(event: RoomEvent, ctx: RoomContext) -> None:
            await asyncio.sleep(10)
            results.append("slow")

        async def fast_hook(event: RoomEvent, ctx: RoomContext) -> None:
            results.append("fast")

        engine.register(
            HookRegistration(
                trigger=HookTrigger.AFTER_BROADCAST,
                execution=HookExecution.ASYNC,
                fn=slow_hook,
                timeout=0.01,
                name="slow",
            )
        )
        engine.register(
            HookRegistration(
                trigger=HookTrigger.AFTER_BROADCAST,
                execution=HookExecution.ASYNC,
                fn=fast_hook,
                name="fast",
            )
        )
        await engine.run_async_hooks("r1", HookTrigger.AFTER_BROADCAST, make_event(), _ctx())
        assert "fast" in results
        assert "slow" not in results

    async def test_async_error_isolation(self) -> None:
        engine = HookEngine()
        results: list[str] = []

        async def bad_hook(event: RoomEvent, ctx: RoomContext) -> None:
            raise RuntimeError("oops")

        async def good_hook(event: RoomEvent, ctx: RoomContext) -> None:
            results.append("good")

        engine.register(
            HookRegistration(
                trigger=HookTrigger.AFTER_BROADCAST,
                execution=HookExecution.ASYNC,
                fn=bad_hook,
                name="bad",
            )
        )
        engine.register(
            HookRegistration(
                trigger=HookTrigger.AFTER_BROADCAST,
                execution=HookExecution.ASYNC,
                fn=good_hook,
                name="good",
            )
        )
        await engine.run_async_hooks("r1", HookTrigger.AFTER_BROADCAST, make_event(), _ctx())
        assert "good" in results


class TestRoomHookManagement:
    def test_remove_room_hook(self) -> None:
        engine = HookEngine()

        async def hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            return HookResult.allow()

        engine.add_room_hook(
            "r1",
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=hook,
                name="my_hook",
            ),
        )
        assert engine.remove_room_hook("r1", "my_hook") is True
        assert engine.remove_room_hook("r1", "missing") is False


class TestHookFiltering:
    """Tests for hook filtering by channel_type, channel_id, and direction."""

    async def test_filter_by_channel_type_matches(self) -> None:
        """Hook runs when channel_type matches."""
        engine = HookEngine()
        called = []

        async def sms_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            called.append("sms")
            return HookResult.allow()

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=sms_hook,
                name="sms_only",
                channel_types={ChannelType.SMS, ChannelType.MMS},
            )
        )

        # SMS event should trigger hook
        sms_event = make_event(channel_type=ChannelType.SMS)
        await engine.run_sync_hooks("r1", HookTrigger.BEFORE_BROADCAST, sms_event, _ctx())
        assert called == ["sms"]

    async def test_filter_by_channel_type_skips(self) -> None:
        """Hook is skipped when channel_type doesn't match."""
        engine = HookEngine()
        called = []

        async def sms_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            called.append("sms")
            return HookResult.allow()

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=sms_hook,
                name="sms_only",
                channel_types={ChannelType.SMS},
            )
        )

        # Email event should NOT trigger hook
        email_event = make_event(channel_type=ChannelType.EMAIL)
        await engine.run_sync_hooks("r1", HookTrigger.BEFORE_BROADCAST, email_event, _ctx())
        assert called == []

    async def test_filter_by_channel_id_matches(self) -> None:
        """Hook runs when channel_id matches."""
        engine = HookEngine()
        called = []

        async def specific_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            called.append(event.source.channel_id)
            return HookResult.allow()

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=specific_hook,
                name="specific_channel",
                channel_ids={"sms-voicemeup"},
            )
        )

        # Matching channel_id should trigger hook
        event = make_event(channel_id="sms-voicemeup")
        await engine.run_sync_hooks("r1", HookTrigger.BEFORE_BROADCAST, event, _ctx())
        assert called == ["sms-voicemeup"]

    async def test_filter_by_channel_id_skips(self) -> None:
        """Hook is skipped when channel_id doesn't match."""
        engine = HookEngine()
        called = []

        async def specific_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            called.append(event.source.channel_id)
            return HookResult.allow()

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=specific_hook,
                name="specific_channel",
                channel_ids={"sms-voicemeup"},
            )
        )

        # Different channel_id should NOT trigger hook
        event = make_event(channel_id="sms-twilio")
        await engine.run_sync_hooks("r1", HookTrigger.BEFORE_BROADCAST, event, _ctx())
        assert called == []

    async def test_filter_by_direction_matches(self) -> None:
        """Hook runs when direction matches."""
        from roomkit.models.enums import ChannelDirection
        from roomkit.models.event import EventSource

        engine = HookEngine()
        called = []

        async def inbound_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            called.append("inbound")
            return HookResult.allow()

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=inbound_hook,
                name="inbound_only",
                directions={ChannelDirection.INBOUND},
            )
        )

        # Inbound event should trigger hook
        inbound_event = RoomEvent(
            room_id="r1",
            source=EventSource(
                channel_id="sms",
                channel_type=ChannelType.SMS,
                direction=ChannelDirection.INBOUND,
            ),
            content=TextContent(body="hello"),
        )
        await engine.run_sync_hooks("r1", HookTrigger.BEFORE_BROADCAST, inbound_event, _ctx())
        assert called == ["inbound"]

    async def test_filter_by_direction_skips(self) -> None:
        """Hook is skipped when direction doesn't match."""
        from roomkit.models.enums import ChannelDirection
        from roomkit.models.event import EventSource

        engine = HookEngine()
        called = []

        async def inbound_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            called.append("inbound")
            return HookResult.allow()

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=inbound_hook,
                name="inbound_only",
                directions={ChannelDirection.INBOUND},
            )
        )

        # Outbound event should NOT trigger hook
        outbound_event = RoomEvent(
            room_id="r1",
            source=EventSource(
                channel_id="sms",
                channel_type=ChannelType.SMS,
                direction=ChannelDirection.OUTBOUND,
            ),
            content=TextContent(body="hello"),
        )
        await engine.run_sync_hooks("r1", HookTrigger.BEFORE_BROADCAST, outbound_event, _ctx())
        assert called == []

    async def test_combined_filters(self) -> None:
        """Hook with multiple filters only runs when all match."""
        from roomkit.models.enums import ChannelDirection
        from roomkit.models.event import EventSource

        engine = HookEngine()
        called = []

        async def rehost_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            called.append("rehost")
            return HookResult.allow()

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=rehost_hook,
                name="rehost_voicemeup_mms",
                channel_types={ChannelType.SMS, ChannelType.MMS},
                channel_ids={"sms-voicemeup"},
                directions={ChannelDirection.INBOUND},
            )
        )

        # All filters match -> hook runs
        matching_event = RoomEvent(
            room_id="r1",
            source=EventSource(
                channel_id="sms-voicemeup",
                channel_type=ChannelType.MMS,
                direction=ChannelDirection.INBOUND,
            ),
            content=TextContent(body="hello"),
        )
        await engine.run_sync_hooks("r1", HookTrigger.BEFORE_BROADCAST, matching_event, _ctx())
        assert called == ["rehost"]

        # Wrong channel_id -> hook skipped
        called.clear()
        wrong_id_event = RoomEvent(
            room_id="r1",
            source=EventSource(
                channel_id="sms-twilio",
                channel_type=ChannelType.MMS,
                direction=ChannelDirection.INBOUND,
            ),
            content=TextContent(body="hello"),
        )
        await engine.run_sync_hooks("r1", HookTrigger.BEFORE_BROADCAST, wrong_id_event, _ctx())
        assert called == []

        # Wrong direction -> hook skipped
        called.clear()
        wrong_dir_event = RoomEvent(
            room_id="r1",
            source=EventSource(
                channel_id="sms-voicemeup",
                channel_type=ChannelType.MMS,
                direction=ChannelDirection.OUTBOUND,
            ),
            content=TextContent(body="hello"),
        )
        await engine.run_sync_hooks("r1", HookTrigger.BEFORE_BROADCAST, wrong_dir_event, _ctx())
        assert called == []

    async def test_no_filters_matches_all(self) -> None:
        """Hook without filters runs for all events."""
        engine = HookEngine()
        called = []

        async def universal_hook(event: RoomEvent, ctx: RoomContext) -> HookResult:
            called.append(event.source.channel_type)
            return HookResult.allow()

        engine.register(
            HookRegistration(
                trigger=HookTrigger.BEFORE_BROADCAST,
                execution=HookExecution.SYNC,
                fn=universal_hook,
                name="universal",
                # No filters = matches all
            )
        )

        for ct in [ChannelType.SMS, ChannelType.EMAIL, ChannelType.WEBSOCKET]:
            await engine.run_sync_hooks(
                "r1", HookTrigger.BEFORE_BROADCAST, make_event(channel_type=ct), _ctx()
            )

        assert called == [ChannelType.SMS, ChannelType.EMAIL, ChannelType.WEBSOCKET]
