"""Integration tests for RealtimeVoiceChannel with full RoomKit."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import (
    Access,
    Channel,
    ChannelBinding,
    ChannelCapabilities,
    ChannelOutput,
    RoomContext,
    RoomKit,
    TextContent,
)
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.channels.websocket import WebSocketChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType
from roomkit.models.event import EventSource, RoomEvent
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport


class ObserverChannel(Channel):
    """Simple observer channel that records all events it sees."""

    channel_type = ChannelType.WEBSOCKET

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.seen_events: list[RoomEvent] = []
        self.delivered_events: list[RoomEvent] = []

    async def handle_inbound(
        self, message: InboundMessage, context: RoomContext
    ) -> RoomEvent:
        return RoomEvent(
            room_id=context.room.id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
            ),
            content=message.content,
        )

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.seen_events.append(event)
        return ChannelOutput.empty()

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.delivered_events.append(event)
        return ChannelOutput.empty()


class TestSupervisorTextInjection:
    async def test_supervisor_text_injection_e2e(self) -> None:
        """Supervisor sends text via WebSocket dashboard → injected into realtime AI."""
        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()

        rt_channel = RealtimeVoiceChannel(
            "rt-voice",
            provider=provider,
            transport=transport,
            system_prompt="You are a helpful agent.",
        )

        # Supervisor channel (sends text)
        supervisor = ObserverChannel("supervisor")

        kit = RoomKit()
        kit.register_channel(rt_channel)
        kit.register_channel(supervisor)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-voice")
        await kit.attach_channel(room.id, "supervisor")

        # Start realtime voice session
        session = await rt_channel.start_session(room.id, "user-1", "fake-ws")

        # Supervisor sends a message through the framework
        await kit.send_event(
            room.id,
            "supervisor",
            TextContent(body="Offer the customer a 20% discount"),
        )
        await asyncio.sleep(0.1)

        # Verify the text was injected into the realtime provider
        assert len(provider.injected_texts) == 1
        assert provider.injected_texts[0][1] == "Offer the customer a 20% discount"
        assert provider.injected_texts[0][2] == "system"

        await kit.close()


class TestTranscriptionsVisibleToAllChannels:
    async def test_transcriptions_visible_to_all_channels(self) -> None:
        """AI transcript from realtime voice → emitted as RoomEvent → other channels see it."""
        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()

        rt_channel = RealtimeVoiceChannel(
            "rt-voice",
            provider=provider,
            transport=transport,
        )

        observer = ObserverChannel("observer")

        kit = RoomKit()
        kit.register_channel(rt_channel)
        kit.register_channel(observer)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-voice")
        await kit.attach_channel(room.id, "observer")

        session = await rt_channel.start_session(room.id, "user-1", "fake-ws")

        # Simulate AI producing a transcription
        await provider.simulate_transcription(
            session, "I can help you with that.", "assistant", True
        )
        await asyncio.sleep(0.2)

        # Observer should have received the transcription event
        assert len(observer.delivered_events) > 0
        texts = [
            e.content.body
            for e in observer.delivered_events
            if isinstance(e.content, TextContent)
        ]
        assert "I can help you with that." in texts

        await kit.close()


class TestMutedChannelStillReceivesEvents:
    async def test_muted_realtime_voice_still_receives_on_event(self) -> None:
        """Even when muted, the realtime voice channel still gets on_event() calls."""
        provider = MockRealtimeProvider()
        transport = MockRealtimeTransport()

        rt_channel = RealtimeVoiceChannel(
            "rt-voice",
            provider=provider,
            transport=transport,
        )

        supervisor = ObserverChannel("supervisor")

        kit = RoomKit()
        kit.register_channel(rt_channel)
        kit.register_channel(supervisor)

        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-voice")
        await kit.attach_channel(room.id, "supervisor")

        session = await rt_channel.start_session(room.id, "user-1", "fake-ws")

        # Mute the realtime voice channel
        await kit.mute(room.id, "rt-voice")

        # Supervisor sends a message
        await kit.send_event(
            room.id,
            "supervisor",
            TextContent(body="Muted but still injecting"),
        )
        await asyncio.sleep(0.1)

        # on_event is still called on muted channels (per EventRouter)
        # So text should still be injected into the provider
        assert len(provider.injected_texts) == 1
        assert provider.injected_texts[0][1] == "Muted but still injecting"

        await kit.close()
