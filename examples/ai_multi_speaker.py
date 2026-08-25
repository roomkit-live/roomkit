"""Speaker attribution in multi-speaker rooms.

Every event that is not the AI's own becomes a ``user`` turn in the model's
history. In a room where several people speak, that erases who said what —
the model guesses the addressee and guesses wrong. When the history window
holds two or more distinct speakers, AIChannel prefixes each attributable
user turn with its speaker ("Alice: ...") and appends a one-line note to the
system prompt. A single-speaker room is left byte-identical.

Nothing to configure. The speaker is read from ``metadata["sender_name"]``
(stamped at ingress by the Teams and WhatsApp Personal ingress, or by the
host on ``InboundMessage``), falling back to the room's participant record.

Run with (no API key needed — MockAIProvider records what it was asked):
    uv run python examples/ai_multi_speaker.py
"""

from __future__ import annotations

import asyncio

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.providers.ai.base import AIContext
from roomkit.providers.ai.mock import MockAIProvider

ROOM = "planning"
CHANNEL = "ws-team"


def show(title: str, context: AIContext) -> None:
    """Print what the provider was asked to generate from."""
    print(f"\n== {title}")
    for message in context.messages:
        print(f"  {message.role:<9} {message.content!r}")
    has_note = "Several people take part" in (context.system_prompt or "")
    print(f"  attribution note in system prompt: {has_note}")


async def say(kit: RoomKit, sender_id: str, body: str, *, name: str | None = None) -> None:
    """Send one message; ``name`` is what an ingress stamps as sender_name."""
    await kit.process_inbound(
        InboundMessage(
            channel_id=CHANNEL,
            sender_id=sender_id,
            content=TextContent(body=body),
            metadata={"sender_name": name} if name else {},
        )
    )


async def main() -> None:
    kit = RoomKit()
    provider = MockAIProvider(
        responses=[
            "Tuesday it is, pending the others.",
            "Two proposals on the table: Tuesday and Thursday.",
            "Alice proposed Tuesday, Bob proposed Thursday.",
        ]
    )
    kit.register_channel(WebSocketChannel(CHANNEL))
    kit.register_channel(
        AIChannel(
            "ai-assistant",
            provider=provider,
            system_prompt="You are the team's scheduling assistant.",
        )
    )
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, CHANNEL)
    await kit.attach_channel(ROOM, "ai-assistant", category=ChannelCategory.INTELLIGENCE)

    # 1. One speaker in the window: the prompt is exactly what it always was.
    await say(kit, "u-alice", "Tuesday works for me.", name="Alice")
    show("one speaker — nothing changes", provider.calls[-1])

    # 2. A second named speaker: every attributable user turn carries its
    #    speaker, history included, and the note is appended once.
    await say(kit, "u-bob", "I would rather ship Thursday.", name="Bob")
    show("two speakers — user turns carry their speaker", provider.calls[-1])

    # 3. No sender_name on the event: the participant record is the fallback.
    await kit.ensure_participant(ROOM, CHANNEL, "u-carol", display_name="Carol")
    await say(kit, "u-carol", "Who proposed what?")
    show("third speaker named by the participant record", provider.calls[-1])

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
