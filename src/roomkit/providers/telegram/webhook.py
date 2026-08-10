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

from dataclasses import dataclass
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
            a consumer is free to attribute the message to someone else. It is
            required rather than defaulted so that an unattributed instance is
            something you write on purpose, not something you forget.
        entities: The message's entities — ``caption_entities`` for media,
            since a caption's markup is carried under its own key. Their
            offsets count UTF-16 code units;
            :func:`~roomkit.providers.telegram.entities.entity_text` is what
            slices by them correctly.
        reply_to_message_id: The message this one replies to, or None. What
            ties an answer to the question it answers, when the question was
            asked with a ``force_reply`` prompt.
        media_group_id: Shared by the several messages Telegram splits one
            album into, or None. Messages carrying it are one post, delivered
            as many.
    """

    content: TextContent | LocationContent
    metadata: dict[str, Any]
    message_id: str
    sender_id: str
    entities: list[dict[str, Any]]
    reply_to_message_id: str | None
    media_group_id: str | None


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
            if not isinstance(media, list) or not media:
                return None
            media = media[-1]
        if not isinstance(media, dict):
            return None
        file_id = media.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            return None
        caption = msg.get("caption", "")
        if not isinstance(caption, str):
            return None
        metadata: dict[str, Any] = {
            "file_id": file_id,
            "media_type": kind,
        }
        metadata.update({attr: media[attr] for attr in _MEDIA_ATTRS if attr in media})
        return TextContent(body=caption), metadata
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

    if not isinstance(msg, dict):
        return None

    text = msg.get("text")
    if isinstance(text, str):
        content = TextContent(body=text)
    elif "text" in msg:
        return None
    elif (media := _parse_media(msg)) is not None:
        content, extra_metadata = media
    elif "location" in msg:
        loc = msg["location"]
        if not isinstance(loc, dict):
            return None
        latitude = loc.get("latitude")
        longitude = loc.get("longitude")
        if (
            not isinstance(latitude, (int, float))
            or isinstance(latitude, bool)
            or not -90 <= latitude <= 90
            or not isinstance(longitude, (int, float))
            or isinstance(longitude, bool)
            or not -180 <= longitude <= 180
        ):
            return None
        content = LocationContent(
            latitude=latitude,
            longitude=longitude,
        )

    if content is None:
        return None

    chat = msg.get("chat")
    sender = msg.get("from")
    reply = msg.get("reply_to_message")
    chat = chat if isinstance(chat, dict) else {}
    sender = sender if isinstance(sender, dict) else {}
    reply = reply if isinstance(reply, dict) else {}
    reply_to = reply.get("message_id")
    raw_entities = msg.get("entities") or msg.get("caption_entities") or []
    entities = (
        [entity for entity in raw_entities if isinstance(entity, dict)]
        if isinstance(raw_entities, list)
        else []
    )
    media_group_id = msg.get("media_group_id")
    return TelegramMessageParts(
        content=content,
        metadata={
            "chat_id": str(chat.get("id", "")),
            "date": msg.get("date", 0),
            **extra_metadata,
        },
        message_id=str(msg.get("message_id", "")),
        sender_id=str(sender.get("id", "")),
        entities=entities,
        reply_to_message_id=str(reply_to) if reply_to is not None else None,
        media_group_id=str(media_group_id) if media_group_id is not None else None,
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
    if not isinstance(payload, dict):
        return []
    msg = payload.get("message")
    if not isinstance(msg, dict):
        return []

    parts = parse_telegram_message(msg)
    if parts is None:
        return []
    chat_id = parts.metadata["chat_id"]
    if not chat_id or not parts.message_id:
        return []
    external_id = f"{chat_id}:{parts.message_id}"

    return [
        InboundMessage(
            channel_id=channel_id,
            sender_id=parts.sender_id,
            content=parts.content,
            external_id=external_id,
            provider_message_id=str(parts.message_id),
            idempotency_key=external_id,
            metadata=parts.metadata,
            # RFC §5.2 — the Update as Telegram sent it. The message object
            # alone would drop the update envelope, which carries the
            # ``update_id`` a reader needs to place this in Telegram's own
            # sequence.
            raw_payload=dict(payload),
        )
    ]
