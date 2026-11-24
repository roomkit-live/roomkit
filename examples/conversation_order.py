"""Read a conversation in the order it was written, live and from the store.

An agent asks another session a question and starts writing "I wait for its
answer." The answer arrives before the agent's first word: the application
stores it in the room and hands it to the running turn, which then gives the
value. The room's timeline keeps the order the rows were stored in, the answer
before the wait. Each streamed text also names the row the room held when its
writing began (``metadata["follows_index"]``), and the event a streaming
channel is handed with the stream names it before the first word. A screen
that places each text after the row it follows shows the wait, the answer,
the value, while the turn runs and when the conversation is opened again.

Run with: uv run python examples/conversation_order.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import math
from typing import Any

from shared import setup_logging

from roomkit import (
    Agent,
    Channel,
    ChannelBinding,
    ChannelCategory,
    ChannelOutput,
    ChannelType,
    EventSource,
    EventType,
    InboundMessage,
    RoomContext,
    RoomEvent,
    RoomKit,
    TextContent,
    Visibility,
    WebSocketChannel,
)
from roomkit.models import InjectMessage
from roomkit.providers.ai import AIContext, AIResponse, MockAIProvider

logger = setup_logging("conversation_order")

ROOM = "session"
EXPECTED = [
    "Ask @review for the value in its file.",
    "I asked @review and wait for its answer.",
    "Reply from @review: MAC-ERABLE",
    "The value is MAC-ERABLE.",
]


def place(event: RoomEvent) -> tuple[int, float, str]:
    """Where a stored row reads: after the row it follows, else at its index."""
    follows = event.metadata.get("follows_index", event.index)
    return (follows, event.index, getattr(event.content, "body", ""))


def texts(places: list[tuple[int, float, str]]) -> list[str]:
    return [text for _, _, text in sorted(places)]


class Asker(MockAIProvider):
    """Says it waits; asked again once the answer came, gives the value."""

    def __init__(self) -> None:
        super().__init__(streaming=True)
        self.generating = asyncio.Event()
        self.first_word = asyncio.Event()

    async def generate(self, context: AIContext) -> AIResponse:
        if any("MAC-ERABLE" in str(message.content) for message in context.messages):
            return AIResponse(content=EXPECTED[3])
        self.generating.set()
        await self.first_word.wait()  # the model call is under way, its first word not written
        return AIResponse(content=EXPECTED[1])


class Screen(Channel):
    """A streaming display: each text goes after the row it follows."""

    channel_type = ChannelType.WEBSOCKET

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.shown: list[tuple[int, float, str]] = []
        self.writing: tuple[int, float, str] | None = None

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    def show(self, event: RoomEvent) -> None:
        self.shown.append(place(event))

    def read(self) -> list[str]:
        return texts(self.shown + ([self.writing] if self.writing else []))

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        self.show(event)
        return ChannelOutput.empty()

    async def deliver_stream(
        self, text_stream: Any, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        # The event the response is rendered under names the row its first
        # text follows; each row of the stream is the one the next follows.
        follows = event.metadata["follows_index"]
        async for item in text_stream:
            if isinstance(item, str):
                written = self.writing[2] if self.writing else ""
                self.writing = (follows, math.inf, written + item)
                logger.info("Live, at a delta: %s", self.read())
            elif isinstance(item, RoomEvent):
                self.writing = None
                self.show(item)
                follows = item.index
        return ChannelOutput.empty()


async def main() -> None:
    provider, screen = Asker(), Screen("screen")
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("ws"))
    kit.register_channel(screen)
    kit.register_channel(Agent("api", provider=provider))
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "ws")
    await kit.attach_channel(ROOM, "screen")
    await kit.attach_channel(ROOM, "api", category=ChannelCategory.INTELLIGENCE)

    asked = InboundMessage(channel_id="ws", sender_id="dev", content=TextContent(body=EXPECTED[0]))
    turn = asyncio.create_task(kit.process_inbound(asked, room_id=ROOM))
    await provider.generating.wait()

    # The other session answers while the model call is under way: the
    # application keeps the answer in the room and hands it to the turn.
    answer = RoomEvent(
        room_id=ROOM,
        type=EventType.SYSTEM,
        source=EventSource(channel_id="screen", channel_type=ChannelType.WEBSOCKET),
        content=TextContent(body=EXPECTED[2]),
        visibility="screen",
    )
    screen.show(await kit.commit_event(ROOM, answer))
    await kit.steer(ROOM, "api", InjectMessage(content="@review answered: MAC-ERABLE"))
    provider.first_word.set()
    await turn

    stored = [e for e in await kit.get_timeline(ROOM) if e.visibility != Visibility.INTERNAL]
    logger.info("Stored, by index: %s", [getattr(e.content, "body", "") for e in stored])
    read_back = texts([place(event) for event in stored])
    logger.info("Live, at the end: %s", screen.read())
    logger.info("Opened again:     %s", read_back)
    await kit.close()
    if not screen.read() == read_back == EXPECTED:
        raise SystemExit("the conversation does not read in the order it was written")


if __name__ == "__main__":
    asyncio.run(main())
