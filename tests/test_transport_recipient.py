"""A room the framework opens for an inbound sender replies to that sender.

The framework records the sender as the binding's correspondent (RFC §10.4);
a transport channel delivers to the address in the binding's metadata under its
recipient key. These tests pin that the two meet: the sender's address is
written where the channel reads it, a recipient already there is kept, and a
binding with no recipient is refused before any send, without tripping the
channel's circuit breaker.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from roomkit import HookTrigger, InboundMessage, RoomKit, TextContent
from roomkit.channels import EmailChannel, SMSChannel, WebSocketChannel, WhatsAppChannel
from roomkit.channels.ai import AIChannel
from roomkit.models.framework_event import FrameworkEvent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.email.mock import MockEmailProvider
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.providers.whatsapp.mock import MockWhatsAppProvider

ALICE = "+15550000001"
BOB = "+15550000002"


async def _eventually(predicate: Callable[[], Any], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def _kit_with_agent(channel: Any) -> RoomKit:
    """A kit with *channel* and an agent that joins every room it creates."""
    kit = RoomKit()
    kit.register_channel(channel)
    kit.register_channel(AIChannel("ai", provider=MockAIProvider(responses=["Hello back"])))

    @kit.hook(HookTrigger.ON_ROOM_CREATED)
    async def agent_joins(event: Any, ctx: Any) -> None:
        await kit.attach_channel(ctx.room.id, "ai")

    return kit


def _text(channel_id: str, sender: str, body: str = "Hi") -> InboundMessage:
    return InboundMessage(channel_id=channel_id, sender_id=sender, content=TextContent(body=body))


@pytest.mark.parametrize(
    ("factory", "provider_cls", "recipient_key", "sender"),
    [
        (lambda p: SMSChannel("ch", provider=p), MockSMSProvider, "phone_number", ALICE),
        (lambda p: WhatsAppChannel("ch", provider=p), MockWhatsAppProvider, "phone_number", ALICE),
        (lambda p: EmailChannel("ch", provider=p), MockEmailProvider, "email_address", "a@x.test"),
    ],
    ids=["sms", "whatsapp", "email"],
)
async def test_room_opened_for_a_sender_replies_to_them(
    factory: Any, provider_cls: Any, recipient_key: str, sender: str
) -> None:
    provider = provider_cls()
    kit = _kit_with_agent(factory(provider))

    result = await kit.process_inbound(_text("ch", sender))
    assert result.event is not None
    await _eventually(lambda: provider.sent)

    assert [m["to"] for m in provider.sent] == [sender]
    binding = await kit.store.get_binding(result.event.room_id, "ch")
    assert binding is not None
    assert binding.participant_id == sender
    assert binding.metadata[recipient_key] == sender
    await kit.close()


async def test_two_senders_on_one_number_each_get_their_own_replies() -> None:
    sms = MockSMSProvider()
    kit = _kit_with_agent(SMSChannel("sms", provider=sms))

    await kit.process_inbound(_text("sms", ALICE))
    await kit.process_inbound(_text("sms", BOB))
    await _eventually(lambda: len(sms.sent) == 2)

    assert sorted(m["to"] for m in sms.sent) == [ALICE, BOB]
    await kit.close()


async def test_a_recipient_the_host_set_is_kept() -> None:
    sms = MockSMSProvider()
    kit = _kit_with_agent(SMSChannel("sms", provider=sms))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai")
    await kit.attach_channel("r1", "sms", metadata={"phone_number": "+15559990000"})

    result = await kit.process_inbound(_text("sms", ALICE))
    assert result.event is not None and result.event.room_id == "r1"
    await _eventually(lambda: sms.sent)

    binding = await kit.store.get_binding("r1", "sms")
    assert binding is not None
    assert binding.participant_id == ALICE
    assert binding.metadata["phone_number"] == "+15559990000"
    assert [m["to"] for m in sms.sent] == ["+15559990000"]
    await kit.close()


async def test_a_binding_naming_the_sender_without_recipient_gets_one() -> None:
    """A binding recorded before it carried a recipient is completed by the next message."""
    sms = MockSMSProvider()
    kit = _kit_with_agent(SMSChannel("sms", provider=sms))
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai")
    await kit.attach_channel("r1", "sms", participant_id=ALICE)

    result = await kit.process_inbound(_text("sms", ALICE))
    assert result.event is not None and result.event.room_id == "r1"
    await _eventually(lambda: sms.sent)

    binding = await kit.store.get_binding("r1", "sms")
    assert binding is not None and binding.metadata["phone_number"] == ALICE
    assert [m["to"] for m in sms.sent] == [ALICE]
    await kit.close()


async def test_a_group_binding_records_no_recipient() -> None:
    sms = MockSMSProvider()
    kit = _kit_with_agent(SMSChannel("sms", provider=sms))
    await kit.create_room(room_id="team")
    await kit.attach_channel("team", "sms", group=True)

    result = await kit.process_inbound(_text("sms", ALICE))
    assert result.event is not None and result.event.room_id == "team"

    binding = await kit.store.get_binding("team", "sms")
    assert binding is not None
    assert binding.participant_id is None
    assert "phone_number" not in binding.metadata
    await kit.close()


async def test_no_recipient_is_refused_before_any_send_and_spares_the_breaker() -> None:
    """Six refusals in one room; another room on the same channel still delivers."""
    sms = MockSMSProvider()
    kit = RoomKit()
    kit.register_channel(SMSChannel("sms", provider=sms))
    kit.register_channel(WebSocketChannel("bank"))
    failed: list[FrameworkEvent] = []

    @kit.on("delivery_failed")
    async def on_failed(event: FrameworkEvent) -> None:
        failed.append(event)

    for room_id, metadata in (("nobody", {}), ("alice", {"phone_number": ALICE})):
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "bank")
        await kit.attach_channel(room_id, "sms", metadata=metadata)

    for n in range(6):
        await kit.send_event("nobody", "bank", TextContent(body=f"notice {n}"))
    await _eventually(lambda: len(failed) == 6)
    assert sms.sent == []
    assert {e.data.get("error") for e in failed} == {"no_recipient"}

    await kit.send_event("alice", "bank", TextContent(body="your appointment"))
    await _eventually(lambda: sms.sent)
    assert [m["to"] for m in sms.sent] == [ALICE]
    await kit.close()
