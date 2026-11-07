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

from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.core.hooks import HookRegistration
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, EventType, HookExecution, HookTrigger
from roomkit.models.event import RoomEvent, TextContent
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AITool,
    AIToolCall,
    StreamEvent,
    StreamTextDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


class _SlowStreamProvider(MockAIProvider):
    """Streams a few words, then hangs — a turn worth interrupting."""

    def __init__(self) -> None:
        super().__init__(streaming=True)

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        for word in ("Voici ", "le début ", "de la réponse"):
            yield StreamTextDelta(text=word)
        await asyncio.sleep(3600)  # the part the user never waits for


SAID = "Voici le début de la réponse"


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


async def _rendered(transport: _StreamingTransport) -> None:
    """Until *transport* rendered what the provider streamed before it hung:
    the reader read it before the test cuts the turn (RFC §10.6.4)."""
    async with asyncio.timeout(5):
        while "".join(getattr(transport, "seen", [])) != SAID:
            await asyncio.sleep(0.005)


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
    await _rendered(transport)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    await kit.hook_engine.drain_observers()  # an ASYNC ON_ERROR hook, were one fired

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


async def test_deferred_cancellation_retains_persisted_response_events() -> None:
    provider = _SlowStreamProvider()
    kit = RoomKit()
    transport = _StreamingTransport("cli")
    kit.register_channel(transport)
    kit.register_channel(AIChannel("ai", provider=provider))
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "cli")
    await kit.attach_channel("room", "ai", category=ChannelCategory.INTELLIGENCE)
    try:
        result = await kit.process_inbound(
            InboundMessage(channel_id="cli", sender_id="user", content=TextContent(body="go")),
            room_id="room",
            defer_delivery=True,
        )
        await _rendered(transport)
        await kit.close()
        assert result.delivery is not None
        result = await result.delivery.wait()
        assert [e.content.body for e in result.response_events] == ["Voici le début de la réponse"]
        assert result.response_events[0].metadata["cancelled"] is True
        assert await kit.store.get_event(result.response_events[0].id) == result.response_events[0]
    finally:
        await kit.close()


async def test_transport_failure_closes_stream_before_returning() -> None:
    class FailingTransport(_StreamingTransport):
        async def deliver_stream(
            self,
            text_stream: AsyncIterator[Any],
            event: RoomEvent,
            binding: ChannelBinding,
            context: RoomContext,
        ) -> ChannelOutput:
            await anext(text_stream)
            raise RuntimeError("rendering failed")

    kit = RoomKit()
    ai = AIChannel("ai", provider=_SlowStreamProvider())
    kit.register_channel(FailingTransport("cli"))
    kit.register_channel(ai)
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "cli")
    await kit.attach_channel("room", "ai", category=ChannelCategory.INTELLIGENCE)
    try:
        result = await kit.process_inbound(
            InboundMessage(channel_id="cli", sender_id="user", content=TextContent(body="go")),
            room_id="room",
        )
        assert str(result.error) == "rendering failed"
        assert ai.active_turns == 0
    finally:
        await kit.close()


class _FailsPastTheEnd(_StreamingTransport):
    """Renders a stream; once its read ends, a cut's included, flushes for
    *flush* seconds, then fails: a target whose ``finally`` fails."""

    def __init__(self, channel_id: str, *, flush: float) -> None:
        super().__init__(channel_id)
        self._flush = flush

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        try:
            await super().deliver_stream(text_stream, event, binding, context)
        finally:
            await asyncio.sleep(self._flush)
            raise ValueError("the flush failed")  # noqa: B012


class _PastTheCut:
    """A kit whose agent says a word, then calls a tool that never returns,
    its streaming target failing *flush* seconds past its stream's end, and
    the ON_ERROR payloads it reports."""

    def __init__(self, flush: float) -> None:
        self.running = asyncio.Event()
        self.told = asyncio.Event()
        self.errors: list[RoomEvent] = []
        self.kit = RoomKit()
        self.kit.register_channel(_FailsPastTheEnd("cli", flush=flush))
        provider = MockAIProvider(
            ai_responses=[
                AIResponse(
                    content="Hello.", tool_calls=[AIToolCall(id="t1", name="wait", arguments={})]
                ),
                AIResponse(content="Bye."),
            ],
            streaming=True,
        )

        async def wait(name: str, arguments: dict[str, Any]) -> str:
            self.running.set()
            await asyncio.Event().wait()
            return "never"

        tool = AITool(name="wait", description="d", parameters={"type": "object"})
        self.kit.register_channel(Agent("ai", provider=provider, tools=[tool], tool_handler=wait))

        @self.kit.hook(HookTrigger.ON_ERROR, execution=HookExecution.ASYNC)
        async def on_error(event: RoomEvent, ctx: RoomContext) -> None:
            self.errors.append(event)
            self.told.set()

    async def room(self) -> None:
        await self.kit.create_room(room_id="room-1")
        await self.kit.attach_channel("room-1", "cli")

    def reported(self) -> list[tuple[str, str | None, str | None]]:
        """What each ON_ERROR payload says: its source, its error, its category."""
        return [
            (e.source.channel_id, e.metadata.get("error"), e.metadata.get("error_category"))
            for e in self.errors
        ]


FLUSHES = pytest.mark.parametrize("flush", [0, 0.01], ids=["at-once", "after-a-flush"])


@FLUSHES
async def test_a_target_that_fails_past_the_close_s_cut_fires_on_error(flush: float) -> None:
    """RFC §12.2 step 13s (RMK-1330): the kit's close cuts the turn a target
    reads; the target that fails past its stream's end is reported to
    ON_ERROR once, as the response's source."""
    cut = _PastTheCut(flush)
    await cut.room()
    await cut.kit.attach_channel("room-1", "ai", category=ChannelCategory.INTELLIGENCE)
    turn = asyncio.create_task(
        cut.kit.process_inbound(
            InboundMessage(channel_id="cli", sender_id="user", content=TextContent(body="go")),
            room_id="room-1",
        )
    )
    await asyncio.wait_for(cut.running.wait(), 5)

    await cut.kit.close()
    await asyncio.wait_for(turn, 5)

    assert cut.reported() == [("ai", "the flush failed", "streaming")]


@FLUSHES
async def test_a_target_that_fails_past_a_task_s_cancel_fires_on_error(flush: float) -> None:
    """RFC §12.2 step 13s, §23.3 (RMK-1330): cancel_task cuts the worker's
    turn a shared target reads; the target that fails past its stream's end
    is reported to ON_ERROR once, as the response's source."""
    cut = _PastTheCut(flush)
    await cut.room()
    task = await cut.kit.delegate("room-1", "ai", "Go.", share_channels=["cli"])
    await asyncio.wait_for(cut.running.wait(), 5)

    assert await asyncio.wait_for(cut.kit.cancel_task(task.id), 5)
    await asyncio.wait_for(cut.told.wait(), 5)
    await cut.kit.hook_engine.drain_observers()

    assert cut.reported() == [("ai", "the flush failed", "streaming")]
    await cut.kit.close()
