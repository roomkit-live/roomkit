"""WhatsApp Personal provider backed by a neonize source.

Does not import neonize directly — accesses the client through the shared
``WhatsAppPersonalSourceProvider`` instance, so importing this module never
requires the optional dependency.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import (
    AudioContent,
    CompositeContent,
    LocationContent,
    MediaContent,
    RoomEvent,
    TextContent,
    VideoContent,
)
from roomkit.providers.whatsapp.base import WhatsAppProvider
from roomkit.sources.base import SourceStatus

if TYPE_CHECKING:
    from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

logger = logging.getLogger("roomkit.providers.whatsapp.personal")


def _build_jid_str(phone: str) -> str:
    """Convert a phone number to a WhatsApp JID string.

    Strips a leading ``+`` and appends ``@s.whatsapp.net`` when no domain
    is present.
    """
    phone = phone.lstrip("+")
    if "@" not in phone:
        return f"{phone}@s.whatsapp.net"
    return phone


def _build_jid(phone: str) -> Any:
    """Convert a phone number to a neonize JID protobuf."""
    from roomkit.sources.neonize import WhatsAppPersonalSourceProvider
    return WhatsAppPersonalSourceProvider._parse_jid(_build_jid_str(phone))


class WhatsAppPersonalProvider(WhatsAppProvider):
    """Outbound WhatsApp delivery via a shared neonize source.

    The provider delegates all sends to the neonize client owned by the
    paired ``WhatsAppPersonalSourceProvider``.  It does **not** manage
    client lifecycle — that responsibility stays with the source.
    """

    def __init__(self, source: WhatsAppPersonalSourceProvider) -> None:
        self._source = source
        self.on_sent: Callable[[str, str], Any] | None = None  # (event_id, wa_msg_id)

    @property
    def name(self) -> str:
        return "whatsapp-personal"

    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        """Send a message through the neonize client.

        Maps ``RoomEvent.content`` types to the appropriate neonize send
        method.
        """
        client = self._source.client
        if client is None or self._source.status != SourceStatus.CONNECTED:
            return ProviderResult(success=False, error="WhatsApp personal source not connected")

        jid = _build_jid(to)
        content = event.content

        try:
            resp = None
            if isinstance(content, CompositeContent):
                resp = await self._send_composite(client, jid, content)
            elif isinstance(content, TextContent):
                resp = await client.send_message(jid, content.body)
            elif isinstance(content, MediaContent):
                resp = await self._send_media(client, jid, content)
            elif isinstance(content, AudioContent):
                ptt = content.mime_type == "audio/ogg"
                resp = await client.send_audio(jid, content.url, ptt=ptt)
            elif isinstance(content, VideoContent):
                resp = await client.send_video(jid, content.url)
            elif isinstance(content, LocationContent):
                await client.send_location(
                    jid,
                    content.latitude,
                    content.longitude,
                    name=content.label or "",
                )
            else:
                return ProviderResult(
                    success=False,
                    error=f"Unsupported content type: {type(content).__name__}",
                )

            wa_msg_id = getattr(resp, "ID", "") if resp else ""
            if wa_msg_id and self.on_sent:
                try:
                    result = self.on_sent(event.id, wa_msg_id)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.debug("on_sent callback failed", exc_info=True)

            return ProviderResult(success=True, provider_message_id=wa_msg_id or None)

        except Exception as exc:
            logger.error("Failed to send WhatsApp personal message: %s", exc)
            return ProviderResult(success=False, error=str(exc))

    @staticmethod
    async def _send_media(client: Any, jid: Any, content: MediaContent) -> Any:
        """Send a single media part via the appropriate neonize method."""
        mime = (content.mime_type or "").lower()
        if mime.startswith("image/"):
            return await client.send_image(jid, content.url, caption=content.caption or "")
        elif mime.startswith("video/"):
            return await client.send_video(jid, content.url, caption=content.caption or "")
        elif mime.startswith("audio/"):
            ptt = mime == "audio/ogg"
            return await client.send_audio(jid, content.url, ptt=ptt)
        else:
            return await client.send_document(
                jid, content.url, filename=content.filename or "file",
            )

    async def _send_composite(self, client: Any, jid: Any, content: CompositeContent) -> Any:
        """Send a CompositeContent — use text as caption on media when possible."""
        text_parts = [p for p in content.parts if isinstance(p, TextContent)]
        media_parts = [p for p in content.parts if isinstance(p, MediaContent)]
        caption = text_parts[0].body if text_parts else ""
        resp = None

        if media_parts:
            for media in media_parts:
                # Use text as caption on the first media part
                if caption and not media.caption:
                    media = MediaContent(
                        url=media.url,
                        mime_type=media.mime_type,
                        caption=caption,
                        filename=getattr(media, "filename", None),
                    )
                    caption = ""  # only use caption once
                resp = await self._send_media(client, jid, media)
        elif caption:
            resp = await client.send_message(jid, caption)
        return resp

    async def send_typing(self, to: str, *, is_typing: bool = True, media: str = "text") -> None:
        """Send a typing indicator to a WhatsApp chat.

        Args:
            to: Phone number or full JID.
            is_typing: ``True`` for composing, ``False`` for paused.
            media: ``"text"`` for typing or ``"audio"`` for recording.
        """
        jid = _build_jid(to)
        if is_typing:
            await self._source.send_composing(jid, media=media)
        else:
            await self._source.send_paused(jid)

    async def mark_read(
        self,
        message_ids: list[str],
        chat: str,
        sender: str,
    ) -> None:
        """Send read receipts (blue ticks) for messages.

        Args:
            message_ids: List of message IDs to mark as read.
            chat: Phone number or full JID of the chat.
            sender: Phone number or full JID of the sender.
        """
        await self._source.mark_read(
            message_ids,
            chat=_build_jid(chat),
            sender=_build_jid(sender),
        )

    async def send_reaction(
        self,
        chat: str,
        sender: str,
        message_id: str,
        emoji: str,
    ) -> None:
        """Send a reaction to a WhatsApp message.

        Args:
            chat: Chat JID of the conversation.
            sender: Sender JID of the message being reacted to.
            message_id: WhatsApp message ID to react to.
            emoji: Emoji reaction (empty string to remove).
        """
        await self._source.send_reaction(chat, sender, message_id, emoji)

    async def close(self) -> None:  # noqa: B027
        """No-op — the source owns the client lifecycle."""
