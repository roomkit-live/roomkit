"""A room tested without a provider: a transport that records what it is
handed, and an agent that answers with a fixed reply.

roomkit.testing gives a test or an example the two channels it needs to see
a room work, as roomkit-rs's roomkit::testing does. Shows:
- RecordingTransport: a text transport of any type; deliveries holds each
  event as transcoded for it, with its text, and texts the texts alone
- EchoAgent: an intelligence channel answering every event it is asked with
  its reply, the event's text after it when quote is set

Run with:
    uv run python examples/testing_channels.py
"""

from __future__ import annotations

import asyncio

from shared import setup_logging

from roomkit import ChannelCategory, ChannelType, InboundMessage, RoomKit, TextContent
from roomkit.testing import EchoAgent, RecordingTransport


async def main() -> None:
    setup_logging("testing_channels")
    kit = RoomKit()
    sms = RecordingTransport("sms", ChannelType.SMS)
    ws = RecordingTransport("ws", ChannelType.WEBSOCKET)
    kit.register_channel(sms)
    kit.register_channel(ws)
    kit.register_channel(EchoAgent("echo", "You said: ", quote=True))

    await kit.create_room(room_id="support")
    await kit.attach_channel("support", "sms")
    await kit.attach_channel("support", "ws")
    await kit.attach_channel("support", "echo", category=ChannelCategory.INTELLIGENCE)

    await kit.process_inbound(
        InboundMessage(
            channel_id="sms", sender_id="+15145550100", content=TextContent(body="Hello.")
        ),
        room_id="support",
    )

    # The WebSocket read the person's message and the agent's answer; the SMS
    # sender, only the answer.
    print("ws: ", ws.texts)
    print("sms:", sms.texts)
    for delivery in ws.deliveries:
        print(f"  {delivery.event.source.channel_id}: {delivery.text}")
    await kit.close()


if __name__ == "__main__":
    asyncio.run(main())
