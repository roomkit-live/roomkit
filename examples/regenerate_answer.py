"""RoomKit -- Regenerate the agent's last answer.

A user asks, the agent answers, and the host regenerates: it asks RoomKit which
message a regenerate would replay (``regenerate_target``), removes the answer
that follows it, then calls ``regenerate_response``. The user's message keeps
its id and index, and only the agent reacts: no transport sees it twice. The
host names the trigger it prepared for (``trigger_id=``), so a message that
lands in between is refused rather than answered twice; the same call on a
closed room is refused before the agent runs.

Everything is mock, so it runs without keys.

Run with:
    uv run python examples/regenerate_answer.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import setup_logging

from roomkit import ChannelCategory, RoomClosedError, RoomKit, TextContent
from roomkit.channels import SMSChannel
from roomkit.channels.ai import AIChannel
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import EventType
from roomkit.providers.ai.mock import MockAIProvider


async def show(kit: RoomKit, room_id: str, title: str) -> None:
    print(f"\n{title}")
    for event in await kit.store.list_events(room_id):
        if event.type == EventType.MESSAGE:
            body = getattr(event.content, "body", "")
            print(f"  #{event.index} {event.source.channel_id:>5}: {body}")


async def main() -> None:
    setup_logging("regenerate_answer")

    kit = RoomKit()
    kit.register_channel(SMSChannel("sms"))
    kit.register_channel(
        AIChannel(
            "agent",
            provider=MockAIProvider(
                responses=[
                    "We open at nine.",
                    "We open at nine and close at six.",
                    "Saturdays too, until noon.",
                ]
            ),
        )
    )
    room = await kit.create_room(room_id="support-1")
    await kit.attach_channel(room.id, "sms")
    await kit.attach_channel(room.id, "agent", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(
            channel_id="sms",
            sender_id="+15551234567",
            content=TextContent(body="What time do you open?"),
        )
    )
    await show(kit, room.id, "First turn:")

    # Ask the primitive which message it would replay, and act on that answer:
    # the regenerated response replaces the one that followed the trigger.
    trigger = await kit.regenerate_target(room.id)
    assert trigger is not None
    print(f"\nregenerate would replay #{trigger.index}: {trigger.content.body!r}")
    for event in await kit.store.list_events(room.id, after_index=trigger.index):
        if event.type == EventType.MESSAGE and event.source.channel_id == "agent":
            await kit.delete_event(room.id, event.id)

    result = await kit.regenerate_response(room.id, trigger_id=trigger.id)
    assert result is not None
    assert result.event is not None
    print(f"regenerated on #{result.event.index}, same message: {result.event.id == trigger.id}")
    await show(kit, room.id, "After regenerate:")

    # The target is read outside the room lock: a message that lands between
    # that read and the regenerate is answered by the pipeline, and naming the
    # trigger refuses the stale regenerate instead of answering it twice.
    stale = await kit.regenerate_target(room.id)
    assert stale is not None
    await kit.process_inbound(
        InboundMessage(
            channel_id="sms",
            sender_id="+15551234567",
            content=TextContent(body="And on Saturdays?"),
        )
    )
    moved = await kit.regenerate_response(room.id, trigger_id=stale.id)
    assert moved is not None
    print(f"\ntrigger moved: blocked={moved.blocked} reason={moved.reason}")
    await show(kit, room.id, "After the message that landed in between:")

    # A closed room refuses the regenerate before the agent runs (RFC §5.1).
    await kit.close_room(room.id)
    refused = await kit.regenerate_response(room.id)
    assert refused is not None
    print(f"\nclosed room: blocked={refused.blocked} reason={refused.reason}")
    try:
        await kit.regenerate_target(room.id)
    except RoomClosedError as exc:
        print(f"closed room: regenerate_target raised RoomClosedError: {exc}")

    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
