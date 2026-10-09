"""A room the framework opens for an inbound sender replies to that sender.

The framework records the sender as the binding's correspondent (RFC §10.4);
a transport channel delivers to the address in the binding's metadata under its
recipient key. These tests pin that the two meet: the sender's address is
written where the channel reads it, a recipient already there is kept, and a
binding with no recipient is refused before any send, without tripping the
channel's circuit breaker. Only a channel whose replies go to the address the
sender writes from (a phone number, an email address) records it; a webhook URL
or a chat id is not the sender's address.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from roomkit import HookTrigger, InboundMessage, RoomKit, TextContent
from roomkit.channels import (
    EmailChannel,
    HTTPChannel,
    SMSChannel,
    TelegramChannel,
    WebSocketChannel,
    WhatsAppChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.framework_event import FrameworkEvent
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.email.mock import MockEmailProvider
from roomkit.providers.http.mock import MockHTTPProvider
from roomkit.providers.sms.mock import MockSMSProvider
from roomkit.providers.telegram.mock import MockTelegramProvider
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


async def test_a_room_prepared_for_someone_else_does_not_take_a_stranger() -> None:
    """The host prepared Alice's room; Bob, writing first, gets a room of his own."""
    sms = MockSMSProvider()
    kit = _kit_with_agent(SMSChannel("sms", provider=sms))
    await kit.create_room(room_id="alice-room")
    await kit.attach_channel("alice-room", "ai")
    await kit.attach_channel("alice-room", "sms", metadata={"phone_number": ALICE})

    result = await kit.process_inbound(_text("sms", BOB, "Hello, I'm Bob"))
    assert result.event is not None and result.event.room_id != "alice-room"
    await _eventually(lambda: sms.sent)

    assert [m["to"] for m in sms.sent] == [BOB]
    events = await kit.store.list_events("alice-room", limit=100)
    assert all(e.source.participant_id != BOB for e in events)
    binding = await kit.store.get_binding("alice-room", "sms")
    assert binding is not None and binding.participant_id == ALICE
    await kit.close()


async def test_each_correspondent_reaches_the_room_prepared_for_them() -> None:
    sms = MockSMSProvider()
    kit = _kit_with_agent(SMSChannel("sms", provider=sms))
    for room_id, number in (("alice-room", ALICE), ("bob-room", BOB)):
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "ai")
        await kit.attach_channel(room_id, "sms", metadata={"phone_number": number})

    to_bob = await kit.process_inbound(_text("sms", BOB))
    to_alice = await kit.process_inbound(_text("sms", ALICE))

    assert to_bob.event is not None and to_bob.event.room_id == "bob-room"
    assert to_alice.event is not None and to_alice.event.room_id == "alice-room"
    await kit.close()


async def test_a_recipient_alone_keeps_a_stranger_out() -> None:
    """A binding stored with a recipient and no correspondent (before this rule)."""
    sms = MockSMSProvider()
    kit = _kit_with_agent(SMSChannel("sms", provider=sms))
    await kit.create_room(room_id="alice-room")
    binding = await kit.attach_channel("alice-room", "sms", metadata={"phone_number": ALICE})
    await kit.store.update_binding(binding.model_copy(update={"participant_id": None}))

    result = await kit.process_inbound(_text("sms", BOB))

    assert result.event is not None and result.event.room_id != "alice-room"
    await kit.close()


@pytest.mark.parametrize(
    ("factory", "provider_cls", "spelling"),
    [
        (
            lambda p: SMSChannel("ch", provider=p, default_country_code="1"),
            MockSMSProvider,
            "15550000001",
        ),
        (
            lambda p: WhatsAppChannel("ch", provider=p),
            MockWhatsAppProvider,
            "whatsapp:+15550000001",
        ),
        (
            lambda p: SMSChannel("ch", provider=p, default_country_code="1"),
            MockSMSProvider,
            "(555) 000-0001",
        ),
    ],
    ids=["sms-without-plus", "whatsapp-scheme", "sms-national"],
)
async def test_a_provider_spelling_reaches_the_e164_room(
    factory: Any, provider_cls: Any, spelling: str
) -> None:
    provider = provider_cls()
    kit = _kit_with_agent(factory(provider))
    for room_id, number in (("alice-room", ALICE), ("bob-room", BOB)):
        await kit.create_room(room_id=room_id)
        await kit.attach_channel(room_id, "ai")
        await kit.attach_channel(room_id, "ch", metadata={"phone_number": number})

    result = await kit.process_inbound(_text("ch", spelling))
    assert result.event is not None and result.event.room_id == "alice-room"
    assert result.event.source.participant_id == ALICE
    await _eventually(lambda: provider.sent)

    assert [m["to"] for m in provider.sent] == [ALICE]
    await kit.close()


async def test_a_stored_recipient_names_its_correspondent_when_claimed() -> None:
    """A binding stored with a recipient and no correspondent, and a router that
    sends bob there: bob is never recorded on it, so alice keeps her room."""
    sms = MockSMSProvider()
    kit = _kit_with_agent(SMSChannel("sms", provider=sms))
    default_route = kit._inbound_router.route  # noqa: SLF001

    async def route(channel_id: str, channel_type: Any, participant_id: Any = None, **kw: Any):
        if participant_id == BOB:
            return "alice-room"
        return await default_route(channel_id, channel_type, participant_id=participant_id, **kw)

    kit._inbound_router.route = route  # type: ignore[method-assign]  # noqa: SLF001
    await kit.create_room(room_id="alice-room")
    binding = await kit.attach_channel("alice-room", "sms", metadata={"phone_number": ALICE})
    await kit.store.update_binding(binding.model_copy(update={"participant_id": None}))

    await kit.process_inbound(_text("sms", BOB))
    alice = await kit.process_inbound(_text("sms", ALICE))

    stored = await kit.store.get_binding("alice-room", "sms")
    assert stored is not None and stored.participant_id == ALICE
    assert alice.event is not None and alice.event.room_id == "alice-room"
    await kit.close()


async def test_a_host_address_is_written_in_e164() -> None:
    kit = _kit_with_agent(SMSChannel("sms", provider=MockSMSProvider(), default_country_code="1"))
    await kit.create_room(room_id="r1")

    await kit.create_room(room_id="r2")

    binding = await kit.attach_channel("r1", "sms", participant_id="555-000-0001")
    member = await kit.add_member("r1", "sms", "1 (555) 000-0002")
    recipient_only = await kit.attach_channel(
        "r2", "sms", metadata={"phone_number": "555.000.0002"}
    )

    assert binding.participant_id == ALICE
    assert member.id == BOB
    assert recipient_only.participant_id == BOB
    assert recipient_only.metadata["phone_number"] == BOB
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


async def test_a_webhook_records_no_sender_as_recipient() -> None:
    """A webhook URL is not the sender's address."""
    kit = _kit_with_agent(HTTPChannel("ch", provider=MockHTTPProvider()))

    result = await kit.process_inbound(_text("ch", "user-7"))

    assert result.event is not None
    binding = await kit.store.get_binding(result.event.room_id, "ch")
    assert binding is not None and binding.participant_id == "user-7"
    assert "recipient_id" not in binding.metadata
    await kit.close()


@pytest.mark.parametrize("chat_id", ["7", "-100500"], ids=["private-chat", "group-chat"])
async def test_a_chat_channel_replies_to_the_chat_the_message_came_from(chat_id: str) -> None:
    """Telegram: a private chat's id is its user's, a group's is the group's; the
    chat is the conversation, so the binding names it."""
    telegram = MockTelegramProvider()
    kit = _kit_with_agent(TelegramChannel("tg", provider=telegram))
    message = InboundMessage(
        channel_id="tg",
        sender_id="7",
        content=TextContent(body="Hi"),
        metadata={"chat_id": chat_id},
    )

    result = await kit.process_inbound(message)
    assert result.event is not None
    await _eventually(lambda: telegram.sent)

    assert [m["to"] for m in telegram.sent] == [chat_id]
    binding = await kit.store.get_binding(result.event.room_id, "tg")
    assert binding is not None and binding.participant_id == chat_id
    assert binding.metadata["telegram_chat_id"] == chat_id
    assert result.event.source.participant_id == "7"
    await kit.close()


async def test_a_room_opened_for_a_webhook_message_replies_to_the_webhook() -> None:
    """The webhook provider posts to its configured URL: no recipient is needed
    (RFC §10.2 step 3d), so the room replies as it did before the refusal."""
    http = MockHTTPProvider()
    kit = _kit_with_agent(HTTPChannel("ch", provider=http))

    result = await kit.process_inbound(_text("ch", "user-7"))
    assert result.event is not None
    await _eventually(lambda: http.sent)

    assert [(m["to"], m["event"].content.body) for m in http.sent] == [("", "Hello back")]
    await kit.close()


async def test_a_webhook_recipient_names_no_correspondent() -> None:
    """The room delivers to the CRM's URL; the CRM writes as ``crm-system``."""
    kit = _kit_with_agent(HTTPChannel("crm", provider=MockHTTPProvider()))
    await kit.create_room(room_id="bridge")
    url = "https://crm.example.com/api/messages"
    binding = await kit.attach_channel("bridge", "crm", metadata={"recipient_id": url})

    result = await kit.process_inbound(_text("crm", "crm-system", "Subscription expires soon"))

    assert binding.participant_id is None
    assert result.event is not None and result.event.room_id == "bridge"
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


async def test_no_recipient_is_logged_as_a_refusal_without_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    kit = RoomKit()
    kit.register_channel(SMSChannel("sms", provider=MockSMSProvider()))
    kit.register_channel(WebSocketChannel("bank"))
    await kit.create_room(room_id="nobody")
    await kit.attach_channel("nobody", "bank")
    await kit.attach_channel("nobody", "sms")

    with caplog.at_level("WARNING", logger="roomkit"):
        await kit.send_event("nobody", "bank", TextContent(body="notice"))
        await _eventually(lambda: any("refused" in r.getMessage() for r in caplog.records))

    refusals = [r for r in caplog.records if "refused" in r.getMessage()]
    assert [r.levelname for r in refusals] == ["WARNING"]
    assert "phone_number" in refusals[0].getMessage()
    assert refusals[0].exc_info is None
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
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
