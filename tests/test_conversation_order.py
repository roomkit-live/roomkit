"""Where a streamed text reads in the conversation (RFC §6.4, RMK-2096).

A record the room receives while a generation is under way, before its first
text, is stored before that text. The text names the row it follows
(``metadata["follows_index"]``), the stream's placeholder names it before the
first word, and a channel that places each text after the row it follows
reads the same conversation live and from the store.

The stored rows' keys are the shared scenario's
(``streaming/a-text-follows-the-row-the-room-held-when-its-writing-began``);
this file holds what a streaming channel reads.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from roomkit import RoomKit
from roomkit.channels.base import Channel
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType, Visibility
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.streaming import LoopEndMarker, SegmentBreakMarker
from tests.test_framework import SimpleChannel

ASK = "Ask the other session."
WAIT = "I wait for the answer."
REPLY = "Reply from @review: MAC-ERABLE"
VALUE = "The value is MAC-ERABLE."


def _place(event: RoomEvent) -> tuple[int, float, str]:
    """Where a stored row reads: after the row it follows, else at its index."""
    return (event.metadata.get("follows_index", event.index), event.index, event.content.body)


class _Waiting(Channel):
    """An agent whose turn says it waits, then, asked again within the turn,
    gives the value. Its first generation tells it started, then writes
    nothing until it is released: a model call under way before its first text."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        if event.type is not EventType.MESSAGE or self.started.is_set():
            return ChannelOutput.empty()
        return ChannelOutput(responded=True, response_stream=self._turn())

    async def _turn(self) -> Any:
        self.started.set()
        await self.release.wait()
        yield WAIT
        yield SegmentBreakMarker()
        yield VALUE
        yield LoopEndMarker(reason="completed")

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()


class _Terminal(SimpleChannel):
    """A streaming transport that shows each text after the row it follows:
    the text under way from the placeholder's row, then from the last row of
    its stream."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id, ChannelType.WEBSOCKET)
        self.placeholder: RoomEvent | None = None
        self.shown: list[tuple[int, float, str]] = []
        self.writing: tuple[int, float, str] | None = None
        # What it showed at each delta, the text under way included.
        self.at_each_delta: list[list[str]] = []

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    def show(self, event: RoomEvent) -> None:
        self.shown.append(_place(event))

    def read(self) -> list[str]:
        under_way = [self.writing] if self.writing is not None else []
        return [text for _, _, text in sorted(self.shown + under_way)]

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.show(event)
        return ChannelOutput.empty()

    async def deliver_stream(self, text_stream, event, binding, context):  # type: ignore[no-untyped-def]
        self.placeholder = event
        follows = event.metadata["follows_index"]
        async for item in text_stream:
            if isinstance(item, str):
                written = self.writing[2] if self.writing is not None else ""
                self.writing = (follows, math.inf, written + item)
                self.at_each_delta.append(self.read())
            elif isinstance(item, RoomEvent):
                self.writing = None
                self.show(item)
                follows = item.index
        return ChannelOutput.empty()


async def test_a_streaming_channel_reads_each_text_after_the_row_it_follows() -> None:
    """Another session's answer arrives between the start of a generation and
    its first text. The placeholder names the message the turn answers, so
    the channel shows the wait above the answer from its first word; live and
    read back from the store, the conversation is the ask, the wait, the
    answer, the value, each once."""
    kit = RoomKit()
    agent, terminal = _Waiting("agent"), _Terminal("terminal")
    await kit.create_room(room_id="r1")
    for channel in (SimpleChannel("sms"), terminal):
        kit.register_channel(channel)
        await kit.attach_channel("r1", channel.channel_id)
    kit.register_channel(agent)
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    try:
        message = InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body=ASK))
        asked = asyncio.create_task(kit.process_inbound(message))
        await asyncio.wait_for(agent.started.wait(), 5)
        answer = RoomEvent(
            room_id="r1",
            type=EventType.SYSTEM,
            source=EventSource(channel_id="terminal", channel_type=ChannelType.WEBSOCKET),
            content=TextContent(body=REPLY),
        )
        terminal.show(await kit.commit_event("r1", answer))
        agent.release.set()
        await asyncio.wait_for(asked, 5)

        stored = await kit.store.list_events("r1", limit=100)
        conversation = [e for e in stored if e.visibility != Visibility.INTERNAL]
        [ask] = [e for e in conversation if e.content.body == ASK]
        assert terminal.placeholder is not None
        assert terminal.placeholder.metadata["follows_index"] == ask.index
        assert terminal.at_each_delta[0] == [ASK, WAIT, REPLY]
        read_back = [text for _, _, text in sorted(_place(e) for e in conversation)]
        assert terminal.read() == read_back == [ASK, WAIT, REPLY, VALUE]
    finally:
        await asyncio.wait_for(kit.close(), 5)
