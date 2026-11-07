"""Directing an agent without putting words in anyone's mouth (RFC §10.1.1).

An application sometimes needs an agent to speak because of something nobody
in the room said: a handoff asking the new agent to introduce itself, a
scheduled nudge. Sent as an ordinary inbound message, that direction is stored
as a participant's words and read by the model as a user turn. Sent as an
``INSTRUCTION``, it reaches only the agent it addresses, is never stored as
the room's, and the pipeline records its fingerprint (never its text) on the
agent's reply. The agent keeps its own copy, an ``INSTRUCTION_READ`` row no
channel is delivered, and its later turns read it where the turn read it.

A ``standalone`` instruction goes further: the turn it opens reads nothing of
the room, for a pass that must start from a blank page (a summary re-run that
would otherwise copy its previous answer).

The example directs the agent three times (a greeting, then the same summary
request without and with ``standalone``) and prints what the model was sent
each time and what the room kept: the summary's turn reads the greeting's
instruction before the greeting, the standalone one reads nothing, and only
the instructions that were not standalone are kept.

Run with:
    uv run python examples/instruction_event.py
"""

from __future__ import annotations

import asyncio

from roomkit import AIChannel, ChannelCategory, EventType, InboundMessage, RoomKit, TextContent
from roomkit.channels import WebSocketChannel
from roomkit.providers.ai import MockAIProvider

ROOM = "support"


async def main() -> None:
    kit = RoomKit()
    kit.register_channel(WebSocketChannel("ws"))
    provider = MockAIProvider(
        ["Hello, I'm the advisor. How can I help?", "Summary: greeting.", "Summary: greeting."]
    )
    kit.register_channel(AIChannel("advisor", provider=provider))
    await kit.create_room(room_id=ROOM)
    await kit.attach_channel(ROOM, "ws", category=ChannelCategory.TRANSPORT)
    await kit.attach_channel(ROOM, "advisor", category=ChannelCategory.INTELLIGENCE)

    # The application directs the advisor. Nobody in the room says this.
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws",
            sender_id="system",
            event_type=EventType.INSTRUCTION,
            content=TextContent(body="You just took over the call. Introduce yourself."),
            addressed_to=["advisor"],
        ),
        room_id=ROOM,
    )

    # The same request twice: the second must not read the room (no history,
    # no memory provider call), so it cannot copy an earlier answer.
    for standalone in (False, True):
        await kit.process_inbound(
            InboundMessage(
                channel_id="ws",
                sender_id="system",
                event_type=EventType.INSTRUCTION,
                content=TextContent(body="Summarize the call so far in one line."),
                addressed_to=["advisor"],
                standalone=standalone,
            ),
            room_id=ROOM,
        )

    print("What the model was sent:")
    labels = ["greeting", "summary", "summary, standalone"]
    for label, call in zip(labels, provider.calls, strict=True):
        print(f"  {label}: {len(call.messages)} message(s)")

    print("What the room holds:")
    for event in await kit.store.list_events(ROOM):
        body = event.content.body  # type: ignore[union-attr]
        if event.type == EventType.MESSAGE:
            print(f"  [{event.source.channel_id}] {body}")
            print(f"      instruction: {event.metadata.get('instruction')!r}")
        elif event.type == EventType.INSTRUCTION_READ:
            print(f"  [{event.source.channel_id}'s own copy, visible to it alone] {body}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
