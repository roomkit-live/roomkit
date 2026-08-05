"""Telegram webhook parsing helpers.

Two layers, deliberately separate. :func:`parse_telegram_message` reads a
Telegram ``message`` and says what it contains — nothing more.
:func:`parse_telegram_webhook` builds an :class:`InboundMessage` on top of it,
and in doing so decides that the sender is ``message.from.id``.

That decision is not universal. Under a one-bot-per-user model a direct message
belongs to the bot's owner, not to the Telegram account that typed it, and a
consumer holding that rule needs the payload without the attribution. Reading a
media file's ``file_id`` must not cost anyone their identity model — hence the
lower layer being public.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from roomkit.models.delivery import InboundMessage
from roomkit.models.event import LocationContent, TextContent

# Media updates carrying a resolvable file: Telegram nests the file object under
# a key named for its kind, and an update carries at most one of them.
_MEDIA_FIELDS = ("voice", "audio", "video_note", "video", "document", "photo")

# Present on some kinds and not others (a voice note has a duration but no file
# name, a document the reverse). Copied through when Telegram sends them, so a
# caller never has to go back to the raw update.
_MEDIA_ATTRS = ("duration", "mime_type", "file_name", "file_size")


@dataclass(frozen=True)
class TelegramMessageParts:
    """What a Telegram ``message`` carries, before anyone decides who sent it.

    Attributes:
        content: The message content — text, a caption for media (empty when
            there is none), or a location.
        metadata: ``chat_id`` and ``date``, plus ``file_id``, ``media_type``
            and whichever of ``duration``, ``mime_type``, ``file_name``,
            ``file_size`` Telegram supplied for a media message.
        message_id: Telegram's message id, as a string.
        sender_id: The Telegram account that typed it. Offered, not imposed —
            a consumer is free to attribute the message to someone else.
    """

    content: TextContent | LocationContent
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str = ""
    sender_id: str = ""


def _parse_media(msg: dict[str, Any]) -> tuple[TextContent, dict[str, Any]] | None:
    """Extract the file reference of a media message, or None if it carries none.

    The body is the caption, which media often lacks — a voice note never has
    one. That empty body is the right answer: the ``file_id`` in metadata is
    what carries the message, and resolving it to bytes belongs to the provider
    that holds the bot token (:meth:`TelegramBotProvider.get_file`).
    """
    for kind in _MEDIA_FIELDS:
        if kind not in msg:
            continue
        media = msg[kind]
        if kind == "photo":
            # Telegram sends multiple sizes; take the largest (last).
            media = media[-1] if media else {}
        metadata: dict[str, Any] = {
            "file_id": media.get("file_id", ""),
            "media_type": kind,
        }
        metadata.update({attr: media[attr] for attr in _MEDIA_ATTRS if attr in media})
        return TextContent(body=msg.get("caption", "")), metadata
    return None


def parse_telegram_message(msg: dict[str, Any]) -> TelegramMessageParts | None:
    """Read a Telegram ``message`` object into its parts, attributing nothing.

    This is the layer to use when your identity model is not Telegram's — it
    hands back what the message says and leaves who sent it to you.
    :func:`parse_telegram_webhook` is this function plus the ordinary
    attribution.

    Text, media (``photo``, ``voice``, ``audio``, ``video_note``, ``video``,
    ``document``) and ``location`` are understood; anything else — a sticker,
    a poll — returns None.

    Args:
        msg: The ``message`` object of a Telegram Update, not the Update itself.

    Returns:
        The message's parts, or None if it carries no content this understands.
    """
    content: TextContent | LocationContent | None = None
    extra_metadata: dict[str, Any] = {}

    if "text" in msg:
        content = TextContent(body=msg["text"])
    elif (media := _parse_media(msg)) is not None:
        content, extra_metadata = media
    elif "location" in msg:
        loc = msg["location"]
        content = LocationContent(
            latitude=loc["latitude"],
            longitude=loc["longitude"],
        )

    if content is None:
        return None

    return TelegramMessageParts(
        content=content,
        metadata={
            "chat_id": str(msg.get("chat", {}).get("id", "")),
            "date": msg.get("date", 0),
            **extra_metadata,
        },
        message_id=str(msg.get("message_id", "")),
        sender_id=str(msg.get("from", {}).get("id", "")),
    )


def parse_telegram_webhook(
    payload: dict[str, Any],
    channel_id: str,
) -> list[InboundMessage]:
    """Convert a Telegram Update payload into InboundMessages.

    Telegram sends one update at a time (unless using ``getUpdates``).
    Only ``message`` updates are processed; edits, channel posts, and
    callback queries are silently skipped.

    Content is read by :func:`parse_telegram_message`; what this adds is the
    attribution — the sender is the Telegram account that typed the message. A
    consumer whose identity model differs should call that function directly
    rather than unpick the result here.

    Media messages — ``photo``, ``voice``, ``audio``, ``video_note``,
    ``video`` and ``document`` — store their ``file_id`` in metadata along
    with a ``media_type`` naming the kind, plus whichever of ``duration``,
    ``mime_type``, ``file_name`` and ``file_size`` Telegram supplied. The
    body is the caption, and is empty when there is none. Callers resolve
    the ``file_id`` to bytes via :meth:`TelegramBotProvider.get_file` and
    :meth:`TelegramBotProvider.download_file`.
    """
    msg = payload.get("message")
    if msg is None:
        return []

    parts = parse_telegram_message(msg)
    if parts is None:
        return []

    return [
        InboundMessage(
            channel_id=channel_id,
            sender_id=parts.sender_id,
            content=parts.content,
            external_id=parts.message_id,
            idempotency_key=parts.message_id,
            metadata=parts.metadata,
        )
    ]
