"""An interrupted turn keeps what it already said.

The console's Esc cancels the in-flight ``process_inbound``. Whatever the
agent had streamed is already on the user's screen, so the timeline must
hold it too — otherwise the room disagrees with what the human read, and the
agent's next context is missing what it already said.

Cancellation is not an error: nobody failed, so ON_ERROR stays silent and the
``CancelledError`` propagates untouched.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.core.hooks import HookRegistration
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.base import AIContext, StreamEvent, StreamTextDelta
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


class _SlowStreamProvider(MockAIProvider):
    """Streams a few words, then hangs — a turn worth interrupting."""

    def __init__(self) -> None:
        super().__init__(streaming=True)
        self.streaming_started = asyncio.Event()

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        for word in ("Voici ", "le début ", "de la réponse"):
            yield StreamTextDelta(text=word)
        self.streaming_started.set()
        await asyncio.sleep(3600)  # the part the user never waits for


class _StreamingTransport(SimpleChannel):
    """A transport that renders a stream, like the console does."""

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        self.seen: list[str] = getattr(self, "seen", [])
        async for chunk in text_stream:
            if isinstance(chunk, str):
                self.seen.append(chunk)
        return ChannelOutput.empty()


async def _interrupted_turn() -> tuple[RoomKit, list[RoomEvent], _StreamingTransport]:
    """Submit a turn, cancel it mid-stream, return what the room kept."""
    provider = _SlowStreamProvider()
    kit = RoomKit()
    transport = _StreamingTransport("cli")
    ai = AIChannel("ai", provider=provider, system_prompt="Test.")
    kit.register_channel(transport)
    kit.register_channel(ai)
    await kit.create_room(room_id="room-1")
    await kit.attach_channel("room-1", "cli")
    await kit.attach_channel("room-1", "ai", category=ChannelCategory.INTELLIGENCE)

    errors: list[RoomEvent] = []

    async def on_error(event: RoomEvent, ctx: RoomContext) -> None:
        errors.append(event)

    kit.hook_engine.register(
        HookRegistration(
            trigger=HookTrigger.ON_ERROR,
            execution=HookExecution.ASYNC,
            fn=on_error,
            name="capture_errors",
        )
    )

    turn = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="cli", sender_id="user", content=TextContent(body="go"))
        )
    )
    await asyncio.wait_for(provider.streaming_started.wait(), timeout=5)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    await asyncio.sleep(0.05)  # let the async ON_ERROR hooks settle, if any

    kit._captured_errors = errors  # type: ignore[attr-defined]
    return kit, errors, transport


class TestInterruptedTurn:
    async def test_partial_answer_is_kept(self) -> None:
        kit, _errors, transport = await _interrupted_turn()

        events = await kit.store.list_events("room-1")
        agent_text = [
            e.content.body
            for e in events
            if e.source.channel_id == "ai" and e.type == EventType.MESSAGE
        ]
        assert agent_text == ["Voici le début de la réponse"]
        await kit.close()

    async def test_what_was_shown_is_what_was_stored(self) -> None:
        # The screen and the timeline must not disagree.
        kit, _errors, transport = await _interrupted_turn()

        events = await kit.store.list_events("room-1")
        stored = next(
            e.content.body
            for e in events
            if e.source.channel_id == "ai" and e.type == EventType.MESSAGE
        )
        assert "".join(transport.seen) == stored
        await kit.close()

    async def test_the_segment_says_it_was_cut_short(self) -> None:
        kit, _errors, _transport = await _interrupted_turn()

        events = await kit.store.list_events("room-1")
        segment = next(
            e for e in events if e.source.channel_id == "ai" and e.type == EventType.MESSAGE
        )
        assert segment.metadata.get("cancelled") is True
        await kit.close()

    async def test_cancelling_is_not_an_error(self) -> None:
        # Nobody failed. A user changed their mind.
        kit, errors, _transport = await _interrupted_turn()

        assert errors == []
        await kit.close()
