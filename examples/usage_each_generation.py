"""What a long turn uses, step by step, and after it is cut.

A streaming channel is told after each generation what it used, in its order
among the text and the rows (`UsageMarker`), so a screen can follow a long turn
instead of waiting for its end. A turn cut from outside keeps what its
generations told: its record reads as a finished turn's, its end `cancelled`.
Shows:
- a `deliver_stream` reading `UsageMarker`, skipped by a channel that shows none
- the turn's record of a turn that finished, and of one cut in its second call

Run with:
    uv run python examples/usage_each_generation.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from roomkit import (
    AIChannel,
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
)
from roomkit.models import UsageMarker
from roomkit.providers.ai import (
    AIResponse,
    AITool,
    AIToolCall,
    MockAIProvider,
)


def _step(text: str, tokens: int, *, calls: bool = True) -> AIResponse:
    """One generation: its text, a call when the turn goes on, what it used."""
    called = [AIToolCall(id=f"call-{tokens}", name="read_file", arguments={})] if calls else []
    return AIResponse(
        content=text,
        tool_calls=called,
        usage={"input_tokens": tokens, "output_tokens": 10},
    )


class Screen(Channel):
    """A channel that shows what each generation of the turn used."""

    channel_type = ChannelType.WEBSOCKET

    def __init__(self, channel_id: str) -> None:
        super().__init__(channel_id)
        self.second_call_runs = asyncio.Event()

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        source = EventSource(channel_id=self.channel_id, channel_type=self.channel_type)
        return RoomEvent(room_id=context.room.id, source=source, content=message.content)

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        return ChannelOutput.empty()

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        async for item in text_stream:
            if isinstance(item, UsageMarker):
                print(f"  context: {item.usage['input_tokens']} tokens on {item.model}")
        return ChannelOutput.empty()


def _record(event: RoomEvent) -> dict[str, Any]:
    keys = ("loop_end_reason", "ai_usage", "ai_last_usage", "ai_usage_complete", "ai_model")
    return {key: event.metadata.get(key) for key in keys}


async def _last_message(kit: RoomKit, room_id: str) -> RoomEvent:
    timeline = await kit.get_timeline(room_id)
    said = [event for event in timeline if event.source.channel_id == "ai"]
    return next(event for event in reversed(said) if event.type == EventType.MESSAGE)


async def _run_turn(*, cut: bool) -> None:
    calls = 0

    async def read_file(name: str, arguments: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        if cut and calls == 2:
            screen.second_call_runs.set()
            await asyncio.Future()  # a long call, which the cut stops
        return "the file"

    provider = MockAIProvider(
        ai_responses=[
            _step("Reading.", 1000),
            _step("Reading more.", 2000),
            _step("Done.", 3000, calls=False),
        ],
        streaming=True,
    )
    kit = RoomKit()
    screen = Screen("screen")
    kit.register_channel(screen)
    kit.register_channel(
        AIChannel(
            "ai",
            provider=provider,
            tools=[AITool(name="read_file", description="Read a file.", parameters={})],
            tool_handler=read_file,
        )
    )
    await kit.create_room(room_id="review")
    await kit.attach_channel("review", "screen")
    await kit.attach_channel("review", "ai", category=ChannelCategory.INTELLIGENCE)
    asked = InboundMessage(
        channel_id="screen",
        sender_id="sylvain",
        content=TextContent(body="Review this."),
        delivery_key="the-review",
    )
    turn = asyncio.create_task(kit.process_inbound(asked, room_id="review"))
    if cut:
        await screen.second_call_runs.wait()
        await kit.cancel_delivery("the-review")
    await turn
    print("  record:", _record(await _last_message(kit, "review")))
    await kit.close()


async def main() -> None:
    print("A turn that finishes:")
    await _run_turn(cut=False)
    print("A turn cut while its second call runs:")
    await _run_turn(cut=True)


if __name__ == "__main__":
    asyncio.run(main())
