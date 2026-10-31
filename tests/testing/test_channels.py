"""The channels ``roomkit.testing`` gives tests and examples, as roomkit-rs's
``roomkit::testing`` gives them (RMK-1021 items 24 and 25)."""

from __future__ import annotations

import pytest

from roomkit import RoomKit
from roomkit.models.channel import ChannelCapabilities
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import EditContent, TextContent
from roomkit.testing import EchoAgent, RecordingTransport


async def _room(
    *, quote: bool = False, capabilities: ChannelCapabilities | None = None
) -> tuple[RoomKit, RecordingTransport]:
    """Room ``r1``: an SMS sender, a WebSocket that records, an echoing agent."""
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    sms = RecordingTransport("sms", ChannelType.SMS)
    ws = RecordingTransport("ws", ChannelType.WEBSOCKET, capabilities=capabilities)
    for channel in (sms, ws):
        kit.register_channel(channel)
        await kit.attach_channel("r1", channel.channel_id)
    kit.register_channel(EchoAgent("ai", "Echo: ", quote=quote))
    await kit.attach_channel("r1", "ai", category=ChannelCategory.INTELLIGENCE)
    return kit, ws


def _said(text: str) -> InboundMessage:
    return InboundMessage(
        channel_id="sms", sender_id="+15145550100", content=TextContent(body=text)
    )


@pytest.mark.parametrize(("quote", "answer"), [(False, "Echo: "), (True, "Echo: Hi.")])
async def test_the_agent_answers_and_the_transport_records_both(quote: bool, answer: str) -> None:
    kit, ws = await _room(quote=quote)

    await kit.process_inbound(_said("Hi."), room_id="r1")
    await kit.close()

    assert ws.texts == ["Hi.", answer]
    assert [delivery.event.source.channel_id for delivery in ws.deliveries] == ["sms", "ai"]


@pytest.mark.parametrize("edits", [False, True])
async def test_an_edit_reaches_the_transport_as_its_capabilities_say(edits: bool) -> None:
    """A transport that edits natively receives the edit as it is; one that
    does not, its corrected text (RFC §6.6)."""
    kit, ws = await _room(capabilities=ChannelCapabilities(supports_edit=True) if edits else None)
    said = await kit.process_inbound(_said("Hi."), room_id="r1")
    assert said.event is not None
    edit = EditContent(target_event_id=said.event.id, new_content=TextContent(body="Hello."))

    await kit.process_inbound(
        InboundMessage(
            channel_id="sms", sender_id="+15145550100", event_type=EventType.EDIT, content=edit
        ),
        room_id="r1",
    )
    await kit.close()

    *_, received = (d.event.content for d in ws.deliveries if d.event.source.channel_id == "sms")
    assert isinstance(received, EditContent) is edits


def test_an_sms_transport_writes_its_senders_in_e164() -> None:
    assert RecordingTransport("sms", ChannelType.SMS).normalize_address("+1 (514) 555-0100") == (
        "+15145550100"
    )
