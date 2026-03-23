"""Tests for the framework delivery module (core/delivery.py)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roomkit.core.delivery import (
    DeliveryContext,
    Immediate,
    Queued,
    WaitForIdle,
    _deliver_to_channel,
    _deliver_to_realtime_voice,
    resolve_strategy,
)
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelCategory, ChannelType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kit(
    *,
    bindings: list[ChannelBinding] | None = None,
    channels: dict[str, MagicMock] | None = None,
) -> MagicMock:
    """Build a mock RoomKit with store.list_bindings and get_channel."""
    kit = MagicMock()
    kit.store = MagicMock()
    kit.store.list_bindings = AsyncMock(return_value=bindings or [])
    kit.get_channel = MagicMock(side_effect=lambda cid: (channels or {}).get(cid))
    kit.process_inbound = AsyncMock()
    return kit


def _binding(
    channel_id: str,
    channel_type: ChannelType,
    category: ChannelCategory = ChannelCategory.TRANSPORT,
) -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id="room-1",
        channel_type=channel_type,
        category=category,
    )


def _channel_mock(
    channel_type: ChannelType,
    category: ChannelCategory = ChannelCategory.TRANSPORT,
) -> MagicMock:
    ch = MagicMock()
    ch.channel_type = channel_type
    ch.category = category
    return ch


# ===========================================================================
# DeliveryContext
# ===========================================================================


class TestDeliveryContext:
    """Tests for DeliveryContext.find_transport_channel_id / resolve_channel_id."""

    async def test_find_transport_prefers_voice(self) -> None:
        bindings = [
            _binding("sms-1", ChannelType.SMS),
            _binding("voice-1", ChannelType.VOICE),
        ]
        kit = _make_kit(bindings=bindings)
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hello")

        result = await ctx.find_transport_channel_id()
        assert result == "voice-1"

    async def test_find_transport_prefers_realtime_voice(self) -> None:
        bindings = [
            _binding("sms-1", ChannelType.SMS),
            _binding("rt-1", ChannelType.REALTIME_VOICE),
        ]
        kit = _make_kit(bindings=bindings)
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hello")

        result = await ctx.find_transport_channel_id()
        assert result == "rt-1"

    async def test_find_transport_falls_back_to_text(self) -> None:
        bindings = [
            _binding("sms-1", ChannelType.SMS),
        ]
        kit = _make_kit(bindings=bindings)
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hello")

        result = await ctx.find_transport_channel_id()
        assert result == "sms-1"

    async def test_find_transport_skips_intelligence_channels(self) -> None:
        bindings = [
            _binding("ai-1", ChannelType.AI, ChannelCategory.INTELLIGENCE),
        ]
        kit = _make_kit(bindings=bindings)
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hello")

        result = await ctx.find_transport_channel_id()
        assert result is None

    async def test_find_transport_no_bindings(self) -> None:
        kit = _make_kit(bindings=[])
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hello")

        result = await ctx.find_transport_channel_id()
        assert result is None

    async def test_resolve_channel_id_explicit(self) -> None:
        kit = _make_kit()
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hello", channel_id="explicit-1")

        result = await ctx.resolve_channel_id()
        assert result == "explicit-1"
        # Should NOT call find_transport_channel_id since explicit was provided
        kit.store.list_bindings.assert_not_called()

    async def test_resolve_channel_id_auto_detects(self) -> None:
        bindings = [_binding("sms-1", ChannelType.SMS)]
        kit = _make_kit(bindings=bindings)
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hello")

        result = await ctx.resolve_channel_id()
        assert result == "sms-1"


# ===========================================================================
# Immediate strategy
# ===========================================================================


class TestImmediateStrategy:
    async def test_deliver_with_transport_channel(self) -> None:
        ch = _channel_mock(ChannelType.SMS)
        kit = _make_kit(
            bindings=[_binding("sms-1", ChannelType.SMS)],
            channels={"sms-1": ch},
        )
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hi")

        strategy = Immediate()
        await strategy.deliver(ctx)

        kit.process_inbound.assert_called_once()

    async def test_deliver_no_channel_logs_warning(self) -> None:
        kit = _make_kit(bindings=[])
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hi")

        strategy = Immediate()
        await strategy.deliver(ctx)

        # No process_inbound should be called
        kit.process_inbound.assert_not_called()


# ===========================================================================
# WaitForIdle strategy
# ===========================================================================


class TestWaitForIdleStrategy:
    def test_init_defaults(self) -> None:
        strategy = WaitForIdle()
        assert strategy.buffer == 1.0
        assert strategy.playback_timeout == 15.0

    def test_init_custom(self) -> None:
        strategy = WaitForIdle(buffer=2.0, playback_timeout=10.0)
        assert strategy.buffer == 2.0
        assert strategy.playback_timeout == 10.0

    async def test_deliver_no_channel(self) -> None:
        kit = _make_kit(bindings=[])
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hi")

        strategy = WaitForIdle()
        await strategy.deliver(ctx)

        kit.process_inbound.assert_not_called()

    async def test_deliver_non_voice_channel(self) -> None:
        ch = _channel_mock(ChannelType.SMS)
        kit = _make_kit(
            bindings=[_binding("sms-1", ChannelType.SMS)],
            channels={"sms-1": ch},
        )
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hi")

        strategy = WaitForIdle(buffer=0)
        await strategy.deliver(ctx)

        # Should deliver without waiting for voice idle
        kit.process_inbound.assert_called_once()

    async def test_deliver_voice_channel_waits(self) -> None:
        ch = _channel_mock(ChannelType.VOICE)
        kit = _make_kit(
            bindings=[_binding("voice-1", ChannelType.VOICE)],
            channels={"voice-1": ch},
        )
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hi")

        strategy = WaitForIdle(buffer=0, playback_timeout=1.0)

        with patch(
            "roomkit.core.delivery._wait_for_voice_idle", new_callable=AsyncMock
        ) as mock_wait:
            await strategy.deliver(ctx)
            mock_wait.assert_called_once_with(ch, "room-1", 1.0, 0)

        kit.process_inbound.assert_called_once()


# ===========================================================================
# Queued strategy
# ===========================================================================


class TestQueuedStrategy:
    def test_init_defaults(self) -> None:
        strategy = Queued()
        assert strategy.buffer == 1.0
        assert strategy.playback_timeout == 15.0
        assert strategy.separator == "\n\n"
        assert strategy._queue == []
        assert strategy._delivering is False

    def test_init_custom(self) -> None:
        strategy = Queued(buffer=0.5, playback_timeout=5.0, separator=" | ")
        assert strategy.buffer == 0.5
        assert strategy.playback_timeout == 5.0
        assert strategy.separator == " | "

    async def test_deliver_single_item(self) -> None:
        ch = _channel_mock(ChannelType.SMS)
        kit = _make_kit(
            bindings=[_binding("sms-1", ChannelType.SMS)],
            channels={"sms-1": ch},
        )
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hello")

        strategy = Queued(buffer=0)
        await strategy.deliver(ctx)

        kit.process_inbound.assert_called_once()
        # After delivery, _delivering should be reset to False
        assert strategy._delivering is False

    async def test_deliver_no_channel_clears_queue(self) -> None:
        kit = _make_kit(bindings=[])
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hello")

        strategy = Queued()
        await strategy.deliver(ctx)

        assert strategy._queue == []
        assert strategy._delivering is False
        kit.process_inbound.assert_not_called()

    async def test_deliver_voice_channel_waits(self) -> None:
        ch = _channel_mock(ChannelType.VOICE)
        kit = _make_kit(
            bindings=[_binding("voice-1", ChannelType.VOICE)],
            channels={"voice-1": ch},
        )
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hi")

        strategy = Queued(buffer=0, playback_timeout=2.0)

        with patch(
            "roomkit.core.delivery._wait_for_voice_idle", new_callable=AsyncMock
        ) as mock_wait:
            await strategy.deliver(ctx)
            mock_wait.assert_called_once_with(ch, "room-1", 2.0, 0)

    async def test_batching_multiple_deliveries(self) -> None:
        """When a second deliver() arrives while the first is in-flight,
        items should be batched together."""
        ch = _channel_mock(ChannelType.VOICE)
        kit = _make_kit(
            bindings=[_binding("voice-1", ChannelType.VOICE)],
            channels={"voice-1": ch},
        )

        strategy = Queued(buffer=0, playback_timeout=0.5, separator=" | ")

        # Simulate voice idle wait that allows second item to enqueue
        async def slow_wait(*_args: object) -> None:
            await asyncio.sleep(0.05)

        with patch("roomkit.core.delivery._wait_for_voice_idle", side_effect=slow_wait):
            ctx1 = DeliveryContext(kit=kit, room_id="room-1", content="first")
            task1 = asyncio.create_task(strategy.deliver(ctx1))

            await asyncio.sleep(0.01)  # Let first delivery start

            # Second deliver should queue and return immediately since _delivering=True
            ctx2 = DeliveryContext(kit=kit, room_id="room-1", content="second")
            await strategy.deliver(ctx2)

            await task1

        # The process_inbound call should contain the batched content
        assert kit.process_inbound.call_count == 1
        call_args = kit.process_inbound.call_args
        inbound_msg = call_args[0][0]
        assert inbound_msg.content.body == "first | second"


# ===========================================================================
# resolve_strategy
# ===========================================================================


class TestResolveStrategy:
    def test_none_returns_none(self) -> None:
        assert resolve_strategy(None) is None

    def test_instance_returns_same(self) -> None:
        s = Immediate()
        assert resolve_strategy(s) is s

    def test_immediate_string(self) -> None:
        result = resolve_strategy("immediate")
        assert isinstance(result, Immediate)

    def test_wait_for_idle_string(self) -> None:
        result = resolve_strategy("wait_for_idle")
        assert isinstance(result, WaitForIdle)

    def test_queued_string(self) -> None:
        result = resolve_strategy("queued")
        assert isinstance(result, Queued)

    def test_unknown_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown delivery strategy"):
            resolve_strategy("invalid_strategy")


# ===========================================================================
# _deliver_to_channel internals
# ===========================================================================


class TestDeliverToChannel:
    async def test_channel_not_found(self) -> None:
        kit = _make_kit()
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hi")

        await _deliver_to_channel(ctx, "missing-channel")
        kit.process_inbound.assert_not_called()

    async def test_intelligence_channel_redirects_to_transport(self) -> None:
        ai_ch = _channel_mock(ChannelType.AI, ChannelCategory.INTELLIGENCE)
        sms_ch = _channel_mock(ChannelType.SMS)
        kit = _make_kit(
            bindings=[_binding("sms-1", ChannelType.SMS)],
            channels={"ai-1": ai_ch, "sms-1": sms_ch},
        )
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hi")

        await _deliver_to_channel(ctx, "ai-1")

        kit.process_inbound.assert_called_once()
        # Verify the inbound message was sent to the transport channel
        call_args = kit.process_inbound.call_args
        inbound_msg = call_args[0][0]
        assert inbound_msg.channel_id == "sms-1"

    async def test_intelligence_no_transport_available(self) -> None:
        ai_ch = _channel_mock(ChannelType.AI, ChannelCategory.INTELLIGENCE)
        kit = _make_kit(
            bindings=[],
            channels={"ai-1": ai_ch},
        )
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hi")

        await _deliver_to_channel(ctx, "ai-1")
        kit.process_inbound.assert_not_called()

    async def test_intelligence_redirect_transport_not_found(self) -> None:
        """Transport channel binding exists but channel is not registered."""
        ai_ch = _channel_mock(ChannelType.AI, ChannelCategory.INTELLIGENCE)
        kit = _make_kit(
            bindings=[_binding("sms-1", ChannelType.SMS)],
            channels={"ai-1": ai_ch},  # sms-1 not in channels dict
        )
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="hi")

        await _deliver_to_channel(ctx, "ai-1")
        kit.process_inbound.assert_not_called()

    async def test_realtime_voice_channel_uses_inject(self) -> None:
        rt_ch = _channel_mock(ChannelType.REALTIME_VOICE)
        rt_ch.get_room_sessions = MagicMock(return_value=[MagicMock(id="session-1")])
        rt_ch.inject_text = AsyncMock()

        kit = _make_kit(
            bindings=[_binding("rt-1", ChannelType.REALTIME_VOICE)],
            channels={"rt-1": rt_ch},
        )
        ctx = DeliveryContext(kit=kit, room_id="room-1", content="injected text")

        await _deliver_to_channel(ctx, "rt-1")

        rt_ch.inject_text.assert_called_once()
        kit.process_inbound.assert_not_called()

    async def test_regular_channel_sends_inbound_message(self) -> None:
        ch = _channel_mock(ChannelType.SMS)
        kit = _make_kit(
            bindings=[_binding("sms-1", ChannelType.SMS)],
            channels={"sms-1": ch},
        )
        ctx = DeliveryContext(
            kit=kit,
            room_id="room-1",
            content="hello",
            metadata={"key": "val"},
        )

        await _deliver_to_channel(ctx, "sms-1")

        kit.process_inbound.assert_called_once()
        call_args = kit.process_inbound.call_args
        inbound_msg = call_args[0][0]
        assert inbound_msg.content.body == "hello"
        assert inbound_msg.sender_id == "system"
        assert inbound_msg.metadata == {"key": "val"}
        assert call_args[1]["room_id"] == "room-1"


# ===========================================================================
# _deliver_to_realtime_voice
# ===========================================================================


class TestDeliverToRealtimeVoice:
    async def test_no_sessions_skips(self) -> None:
        channel = MagicMock()
        channel.get_room_sessions = MagicMock(return_value=[])
        channel.inject_text = AsyncMock()

        ctx = DeliveryContext(kit=MagicMock(), room_id="room-1", content="hello")

        await _deliver_to_realtime_voice(channel, ctx)
        channel.inject_text.assert_not_called()

    async def test_injects_to_all_sessions(self) -> None:
        s1 = MagicMock(id="s1")
        s2 = MagicMock(id="s2")
        channel = MagicMock()
        channel.get_room_sessions = MagicMock(return_value=[s1, s2])
        channel.inject_text = AsyncMock()

        ctx = DeliveryContext(kit=MagicMock(), room_id="room-1", content="msg")

        await _deliver_to_realtime_voice(channel, ctx)

        assert channel.inject_text.call_count == 2
        channel.inject_text.assert_any_call(s1, "msg")
        channel.inject_text.assert_any_call(s2, "msg")


# ===========================================================================
# _wait_for_voice_idle
# ===========================================================================


class TestWaitForVoiceIdle:
    async def test_voice_channel_wait(self) -> None:
        from roomkit.core.delivery import _wait_for_voice_idle

        # Patch the imports inside _wait_for_voice_idle
        with (
            patch(
                "roomkit.core.delivery._wait_for_voice_idle.__module__",
                create=True,
            ),
        ):
            from roomkit.channels.voice import VoiceChannel

            ch = MagicMock(spec=VoiceChannel)
            ch.wait_playback_done = AsyncMock()

            await _wait_for_voice_idle(ch, "room-1", timeout=5.0, buffer=0)
            ch.wait_playback_done.assert_called_once_with("room-1", timeout=5.0)

    async def test_realtime_voice_channel_wait(self) -> None:
        from roomkit.channels.realtime_voice import (
            RealtimeVoiceChannel as _RtVC,
        )
        from roomkit.core.delivery import _wait_for_voice_idle

        ch = MagicMock(spec=_RtVC)
        ch.wait_idle = AsyncMock()

        await _wait_for_voice_idle(ch, "room-1", timeout=5.0, buffer=0)
        ch.wait_idle.assert_called_once_with("room-1", timeout=5.0)

    async def test_non_voice_channel_returns_immediately(self) -> None:
        from roomkit.core.delivery import _wait_for_voice_idle

        ch = MagicMock()  # No spec = not isinstance of either
        # Ensure it doesn't have the spec of VoiceChannel or RealtimeVoiceChannel
        ch.__class__ = type("OtherChannel", (), {})

        # Should return without error or waiting
        await _wait_for_voice_idle(ch, "room-1", timeout=5.0, buffer=0)

    async def test_buffer_adds_sleep(self) -> None:
        from roomkit.channels.voice import VoiceChannel
        from roomkit.core.delivery import _wait_for_voice_idle

        ch = MagicMock(spec=VoiceChannel)
        ch.wait_playback_done = AsyncMock()

        with patch("roomkit.core.delivery.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await _wait_for_voice_idle(ch, "room-1", timeout=5.0, buffer=1.5)
            mock_sleep.assert_called_once_with(1.5)
