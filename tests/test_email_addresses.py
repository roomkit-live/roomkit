"""Email addresses in one form: the same mailbox is one correspondent."""

from __future__ import annotations

import pytest

from roomkit import InboundMessage, RoomKit, TextContent
from roomkit.channels import EmailChannel
from roomkit.channels._email import normalize_email_address
from roomkit.providers.email.mock import MockEmailProvider


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("alice@example.com", "alice@example.com"),
        ("Alice@Example.COM", "alice@example.com"),
        ("  alice@example.com ", "alice@example.com"),
        ("Alice Martin <Alice@Example.com>", "alice@example.com"),
        ('"Martin, Alice" <alice@example.com>', "alice@example.com"),
        ("mailto:Alice@example.com", "alice@example.com"),
    ],
)
def test_an_address_whatever_its_case_or_wrapping(address: str, expected: str) -> None:
    assert normalize_email_address(address) == expected


@pytest.mark.parametrize("address", ["", "system", "user-42", "not an address", "a@b@c"])
def test_anything_else_is_returned_unchanged(address: str) -> None:
    assert normalize_email_address(address) == address


async def test_one_mailbox_written_two_ways_is_one_room() -> None:
    kit = RoomKit()
    kit.register_channel(EmailChannel("email", provider=MockEmailProvider()))

    async def room_of(sender: str) -> str:
        result = await kit.process_inbound(
            InboundMessage(channel_id="email", sender_id=sender, content=TextContent(body="hi"))
        )
        assert result.event is not None
        return result.event.room_id

    first = await room_of("Alice Martin <Alice@Example.com>")
    second = await room_of("alice@example.com")

    assert first == second
    binding = await kit.store.get_binding(first, "email")
    assert binding is not None
    assert binding.participant_id == "alice@example.com"
    assert binding.metadata["email_address"] == "alice@example.com"
    await kit.close()
