"""How a channel's addresses are written, and where its recipient and its chat are.

A channel's type decides how its addresses are written and compared (RFC
§5.7, Appendix A): a phone number in E.164, an email address lower-case
without its display name, any other id as written. Its declaration names
the binding metadata key its recipient is written under and, on a channel
that answers chats, the metadata key its inbound messages carry their chat
under; a declaration may name other keys, or none. :func:`addressing_of`
gives a type's defaults, and a transport channel built for a type takes
them unless it declares its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from roomkit.channels._email import normalize_email_address
from roomkit.channels._phone import normalize_phone_number
from roomkit.models.enums import ChannelType

__all__ = ["Addressing", "ChannelAddressing", "addressing_of"]


class Addressing(StrEnum):
    """How a channel's senders are addressed (RFC §5.7)."""

    PHONE_NUMBER = "phone_number"
    """By phone number, compared and stored in E.164 (``+15550000001``)."""

    EMAIL_ADDRESS = "email_address"
    """By email address, lower-case and without a display name."""

    CHAT = "chat"
    """By the chat a message was posted in: replies go to the chat, not to
    its sender (Telegram, Teams)."""

    AS_WRITTEN = "as_written"
    """By an id kept as written."""


@dataclass(frozen=True)
class ChannelAddressing:
    """A channel's addressing declaration (RFC §5.7, Appendix A).

    *addressing* is how its addresses are written. *recipient_key* is the
    binding metadata key its recipient is written under, ``None`` for a
    channel that delivers to a destination of its own (a WebSocket's
    connections, a voice session). *chat_key* is the inbound metadata key a
    message carries its chat under, on a channel that answers chats.
    *replies_to_sender* says whether replies go to the address the
    correspondent writes from (a phone number, an email address, a Messenger
    id): the framework then records an inbound sender as a binding's
    recipient. On a channel with a *chat_key* and *replies_to_sender*
    (WhatsApp Personal), a message carrying its chat is its group's and any
    other its sender's; with a *chat_key* alone, replies go to a chat only,
    and a message that does not carry its chat has no address.
    """

    addressing: Addressing
    recipient_key: str | None = None
    chat_key: str | None = None
    replies_to_sender: bool = False

    @property
    def names_conversations(self) -> bool:
        """Whether a binding's recipient is a conversation's address: the
        sender's, or a chat. A recipient a channel merely forwards (a
        webhook's) names no one."""
        return self.recipient_key is not None and (
            self.replies_to_sender or self.chat_key is not None
        )

    def normalize(self, address: str, country_code: str | None = None) -> str:
        """*address* in the form the channel compares and stores it in; a
        phone number written without its country code takes *country_code*
        (``"1"``, ``"+44"``), or is kept as written without one.

        Raises:
            ValueError: *country_code* is not a country code
                (:func:`checked_country_code`).
        """
        if country_code is not None:
            country_code = checked_country_code(country_code)
        if self.addressing is Addressing.PHONE_NUMBER:
            return normalize_phone_number(address, country_code)
        if self.addressing is Addressing.EMAIL_ADDRESS:
            return normalize_email_address(address)
        return address


def checked_country_code(code: str) -> str:
    """*code*, a country code written with or without its ``+`` (``"1"``,
    ``"+44"``), as its digits.

    Raises:
        ValueError: *code* is not one to three digits, the first not ``0``.
    """
    digits = code.removeprefix("+")
    if not (1 <= len(digits) <= 3 and digits.isascii() and digits.isdigit()) or digits[0] == "0":
        raise ValueError(f"{code!r} is not a country code: one to three digits, the first not 0")
    return digits


_PHONE = ChannelAddressing(
    Addressing.PHONE_NUMBER, recipient_key="phone_number", replies_to_sender=True
)

_BY_TYPE: dict[ChannelType, ChannelAddressing] = {
    ChannelType.SMS: _PHONE,
    ChannelType.MMS: _PHONE,
    ChannelType.RCS: _PHONE,
    ChannelType.WHATSAPP: _PHONE,
    ChannelType.WHATSAPP_PERSONAL: ChannelAddressing(
        Addressing.PHONE_NUMBER,
        recipient_key="phone_number",
        chat_key="group_jid",
        replies_to_sender=True,
    ),
    ChannelType.VOICE: ChannelAddressing(Addressing.PHONE_NUMBER),
    ChannelType.EMAIL: ChannelAddressing(
        Addressing.EMAIL_ADDRESS, recipient_key="email_address", replies_to_sender=True
    ),
    ChannelType.MESSENGER: ChannelAddressing(
        Addressing.AS_WRITTEN, recipient_key="facebook_user_id", replies_to_sender=True
    ),
    ChannelType.TELEGRAM: ChannelAddressing(
        Addressing.CHAT, recipient_key="telegram_chat_id", chat_key="chat_id"
    ),
    ChannelType.TEAMS: ChannelAddressing(
        Addressing.CHAT, recipient_key="teams_conversation_id", chat_key="conversation_id"
    ),
}

_AS_WRITTEN = ChannelAddressing(Addressing.AS_WRITTEN)


def addressing_of(channel_type: ChannelType) -> ChannelAddressing:
    """The addressing a channel of *channel_type* takes by default (RFC
    §5.7, Appendix A): SMS, MMS, RCS, WhatsApp and WhatsApp Personal by
    phone number under ``phone_number``, WhatsApp Personal's groups under
    ``group_jid``; VOICE by phone number with no recipient; EMAIL by email
    address under ``email_address``; Messenger as written under
    ``facebook_user_id``; Telegram and Teams by chat, under
    ``telegram_chat_id`` and ``teams_conversation_id``, their chats under
    ``chat_id`` and ``conversation_id``; any other type as written, with no
    recipient and no chat key."""
    return _BY_TYPE.get(channel_type, _AS_WRITTEN)


@dataclass(frozen=True)
class UnreadableRecipient:
    """A value under a recipient key that is not a recipient (RFC §5.7): a
    boolean, a float, a list, an object, any value that is neither a
    recipient nor no value.

    It is not "no recipient": routing reads it as a recipient that is not
    the sender, whoever writes, so it equals no address. *kind* is the
    value's type name, which a log may show; the value itself, perhaps a
    person's number, is not kept.
    """

    kind: str


def read_recipient(
    metadata: Mapping[str, Any], key: str
) -> str | int | UnreadableRecipient | None:
    """What *metadata* holds under *key* as a recipient (RFC §5.7), the one
    rule every read of a recipient takes.

    The recipient as written: a string that is not blank, or an integer (a
    chat id). ``None`` for no value: the key absent, null, or a blank
    string. An :class:`UnreadableRecipient` for any other value (a boolean,
    a float, a list, an object): a delivery to it is refused
    ``no_recipient``, as one to no value is, while routing reads it as
    someone else's and the record of a routed sender never writes over it.
    """
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return UnreadableRecipient(type(value).__name__)


def written_recipient(metadata: Mapping[str, Any], key: str) -> str | int | None:
    """The recipient *metadata* holds under *key*, as written (RFC §5.7, §10.2
    step 3d), by :func:`read_recipient`'s rule: a string that is not blank,
    or an integer (a chat id); ``None`` for no value and for a value that is
    not a recipient. A delivery reads it, refusing ``no_recipient`` where
    it is ``None``, and so do the stores' recipient lookup
    (``find_room_id_by_binding``), where it finds no one, and the chat a
    message carries under its channel's chat key."""
    held = read_recipient(metadata, key)
    return None if isinstance(held, UnreadableRecipient) else held
