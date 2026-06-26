"""Discord bot provider backed by a shared gateway source.

Does not import ``discord`` at module load — the gateway client is owned by
the paired :class:`~roomkit.sources.discord.DiscordGatewaySource`, so importing
this module never requires the optional dependency. ``discord`` is imported
lazily only where its types are needed (embeds, file uploads, replies).
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import time
from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import ProviderResult
from roomkit.models.event import MediaContent, RichContent, RoomEvent
from roomkit.providers.discord.base import DiscordProvider
from roomkit.providers.utils import extract_event_text
from roomkit.sources.base import SourceStatus
from roomkit.telemetry.noop import NoopTelemetryProvider

if TYPE_CHECKING:
    from roomkit.sources.discord import DiscordGatewaySource

logger = logging.getLogger("roomkit.providers.discord")


class DiscordBotProvider(DiscordProvider):
    """Outbound Discord delivery via a shared gateway source.

    The provider delegates every send to the ``discord.Client`` owned by the
    paired :class:`~roomkit.sources.discord.DiscordGatewaySource`. It does
    **not** manage the client lifecycle — that stays with the source.
    """

    def __init__(self, source: DiscordGatewaySource) -> None:
        self._source = source
        self._telemetry: Any = None

    @property
    def name(self) -> str:
        return "discord"

    async def send(self, event: RoomEvent, to: str) -> ProviderResult:
        """Send ``event`` to Discord channel ``to`` (a channel snowflake)."""
        client = self._connected_client()
        if client is None:
            return ProviderResult(success=False, error="discord_not_connected")
        channel = client.get_channel(int(to)) or await client.fetch_channel(int(to))
        if channel is None:
            return ProviderResult(success=False, error="channel_not_found")

        kwargs = self._build_kwargs(event)
        if kwargs is None:
            return ProviderResult(success=False, error="empty_message")
        self._apply_reference(kwargs, event, channel)

        t0 = time.monotonic()
        try:
            sent = await channel.send(**kwargs)
        except Exception as exc:  # discord.HTTPException / Forbidden / etc.
            logger.warning("Discord send failed: %s", exc)
            return ProviderResult(success=False, error=str(exc))
        self._record_send_ms((time.monotonic() - t0) * 1000)
        return ProviderResult(success=True, provider_message_id=str(sent.id))

    async def send_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        client = self._connected_client()
        if client is None:
            return
        cid = int(channel_id)
        channel = client.get_channel(cid) or await client.fetch_channel(cid)
        if channel is None:
            return
        message = await channel.fetch_message(int(message_id))
        await message.add_reaction(emoji)

    # -- internals ----------------------------------------------------------

    def _connected_client(self) -> Any:
        """Return the live gateway client, or None when not yet connected."""
        client = self._source.client
        if client is None or self._source.status != SourceStatus.CONNECTED:
            return None
        return client

    def _build_kwargs(self, event: RoomEvent) -> dict[str, Any] | None:
        """Map ``event.content`` to ``channel.send`` keyword arguments.

        Returns ``None`` when the event carries no sendable content.
        """
        content = event.content
        if isinstance(content, RichContent):
            return self._embed_kwargs(content)
        if isinstance(content, MediaContent):
            return self._media_kwargs(content)
        text = extract_event_text(event)
        return {"content": text} if text else None

    @staticmethod
    def _embed_kwargs(content: RichContent) -> dict[str, Any]:
        import discord

        body = content.plain_text or content.body
        return {"embed": discord.Embed(description=body)}

    @staticmethod
    def _media_kwargs(content: MediaContent) -> dict[str, Any]:
        """Send media: http(s) URLs ride in content (Discord auto-embeds);
        ``data:`` URIs are decoded and uploaded as a file."""
        if content.url.startswith("data:"):
            file = _decode_data_uri(content.url, content.filename)
            if file is not None:
                kwargs: dict[str, Any] = {"file": file}
                if content.caption:
                    kwargs["content"] = content.caption
                return kwargs
        body = content.url if not content.caption else f"{content.caption}\n{content.url}"
        return {"content": body}

    @staticmethod
    def _apply_reference(kwargs: dict[str, Any], event: RoomEvent, channel: Any) -> None:
        """Reply to ``channel_data.thread_id`` (a message snowflake) when set."""
        thread_id = event.channel_data.thread_id if event.channel_data else None
        if not thread_id:
            return
        import discord

        kwargs["reference"] = discord.MessageReference(
            message_id=int(thread_id),
            channel_id=channel.id,
            fail_if_not_exists=False,
        )

    def _record_send_ms(self, send_ms: float) -> None:
        tel = self._telemetry or NoopTelemetryProvider()
        tel.record_metric(
            "roomkit.delivery.send_ms",
            send_ms,
            unit="ms",
            attributes={"provider": "DiscordBotProvider"},
        )


def _decode_data_uri(uri: str, filename: str | None) -> Any:
    """Decode a ``data:<mime>;base64,<payload>`` URI into a ``discord.File``."""
    import discord

    try:
        header, _, payload = uri.partition(",")
        raw = base64.b64decode(payload)
    except (binascii.Error, ValueError):
        logger.debug("Failed to decode data URI for Discord upload")
        return None
    name = filename or _default_filename(header)
    return discord.File(io.BytesIO(raw), filename=name)


def _default_filename(header: str) -> str:
    """Derive a filename from a data-URI header (``data:image/png;base64``)."""
    mime = header[5:].split(";", 1)[0] if header.startswith("data:") else ""
    ext = mime.split("/", 1)[1] if "/" in mime else "bin"
    return f"attachment.{ext}"
