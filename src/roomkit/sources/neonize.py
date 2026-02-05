"""WhatsApp Personal (neonize) event source for RoomKit.

Uses neonize (Python wrapper around whatsmeow) for the unofficial WhatsApp
Web multidevice protocol.  Personal use and experimentation only.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from roomkit.models.delivery import InboundMessage
from roomkit.models.event import (
    AudioContent,
    LocationContent,
    MediaContent,
    TextContent,
    VideoContent,
)
from roomkit.sources.base import BaseSourceProvider, EmitCallback, SourceStatus

# Optional dependency --------------------------------------------------------
try:
    from neonize.aioze.client import NewAClient  # type: ignore[import-untyped]

    HAS_NEONIZE = True
except ImportError:
    NewAClient = None  # type: ignore[assignment, misc]
    HAS_NEONIZE = False

logger = logging.getLogger("roomkit.sources.neonize")

# Type aliases ---------------------------------------------------------------
NeonizeEventCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]
NeonizeMessageParser = Callable[..., Awaitable[InboundMessage | None] | InboundMessage | None]


# ---------------------------------------------------------------------------
# Default message parser
# ---------------------------------------------------------------------------


def default_message_parser(
    channel_id: str,
    *,
    self_chat: bool = False,
) -> NeonizeMessageParser:
    """Create a parser that converts neonize MessageEv into InboundMessage.

    Handles text, image, audio, video, document, location, and sticker
    messages.

    Args:
        channel_id: Channel ID to stamp on every emitted message.
        self_chat: When ``True``, process own messages (self-chat mode
            for testing on your own number).

    Returns:
        An async parser callable.
    """

    async def parser(
        client: Any,
        event: Any,
    ) -> InboundMessage | None:
        try:
            info = event.Info
            msg = event.Message
            src = info.MessageSource

            # Skip own messages unless self-chat mode is enabled
            if src.IsFromMe and not self_chat:
                return None

            # Sender JID → readable ID (strip @s.whatsapp.net)
            sender_jid = src.Sender
            user = getattr(sender_jid, "User", "")
            server = getattr(sender_jid, "Server", "")
            raw_jid = f"{user}@{server}" if user and server else user
            sender_id = user or ""
            if not sender_id:
                return None

            chat_jid = src.Chat
            chat_server = getattr(chat_jid, "Server", "")
            is_group = chat_server == "g.us"

            chat_user = getattr(chat_jid, "User", "")
            chat_raw = f"{chat_user}@{chat_server}" if chat_user and chat_server else chat_user

            metadata: dict[str, Any] = {
                "raw_jid": raw_jid,
                "chat_jid": chat_raw,
                "chat_id": chat_user,
                "is_from_me": bool(src.IsFromMe),
                "is_group": is_group,
                "timestamp": str(info.Timestamp),
                "push_name": getattr(info, "Pushname", None),
            }

            content = None

            # --- Text ---
            # Note: protobuf3 sub-messages are always truthy even when unset,
            # so we must use HasField() for reliable detection.
            if msg.conversation:
                content = TextContent(body=msg.conversation)
            elif msg.HasField("extendedTextMessage"):
                text = getattr(msg.extendedTextMessage, "text", None) or ""
                content = TextContent(body=text)

            # --- Image ---
            elif msg.HasField("imageMessage"):
                try:
                    data = await client.download_any(msg)
                    import base64

                    b64 = base64.b64encode(data).decode()
                    mime = getattr(msg.imageMessage, "mimetype", "image/jpeg")
                    url = f"data:{mime};base64,{b64}"
                    caption = getattr(msg.imageMessage, "caption", None)
                    content = MediaContent(url=url, mime_type=mime, caption=caption)
                except Exception:
                    logger.debug("Failed to download image, using placeholder")
                    content = MediaContent(
                        url="data:image/jpeg;base64,",
                        mime_type="image/jpeg",
                    )

            # --- Audio ---
            elif msg.HasField("audioMessage"):
                try:
                    data = await client.download_any(msg)
                    import base64

                    b64 = base64.b64encode(data).decode()
                    ptt = getattr(msg.audioMessage, "ptt", False)
                    mime = "audio/ogg" if ptt else getattr(msg.audioMessage, "mimetype", "audio/mp4")
                    url = f"data:{mime};base64,{b64}"
                    duration = getattr(msg.audioMessage, "seconds", None)
                    dur_float = float(duration) if duration else None
                    content = AudioContent(url=url, mime_type=mime, duration_seconds=dur_float)
                except Exception:
                    logger.debug("Failed to download audio, using placeholder")
                    content = AudioContent(url="data:audio/ogg;base64,", mime_type="audio/ogg")

            # --- Video (stored as MediaContent since VideoContent rejects data: URIs) ---
            elif msg.HasField("videoMessage"):
                try:
                    data = await client.download_any(msg)
                    import base64

                    b64 = base64.b64encode(data).decode()
                    mime = getattr(msg.videoMessage, "mimetype", "video/mp4")
                    url = f"data:{mime};base64,{b64}"
                    content = MediaContent(url=url, mime_type=mime)
                except Exception:
                    logger.debug("Failed to download video, using placeholder")
                    content = MediaContent(
                        url="data:video/mp4;base64,",
                        mime_type="video/mp4",
                    )

            # --- Document ---
            elif msg.HasField("documentMessage"):
                try:
                    data = await client.download_any(msg)
                    import base64

                    b64 = base64.b64encode(data).decode()
                    mime = getattr(msg.documentMessage, "mimetype", "application/octet-stream")
                    filename = getattr(msg.documentMessage, "fileName", None)
                    url = f"data:{mime};base64,{b64}"
                    content = MediaContent(url=url, mime_type=mime, filename=filename)
                except Exception:
                    logger.debug("Failed to download document, using placeholder")
                    content = MediaContent(
                        url="data:application/octet-stream;base64,",
                        mime_type="application/octet-stream",
                    )

            # --- Location ---
            elif msg.HasField("locationMessage"):
                loc = msg.locationMessage
                content = LocationContent(
                    latitude=float(loc.degreesLatitude),
                    longitude=float(loc.degreesLongitude),
                    label=getattr(loc, "name", None),
                    address=getattr(loc, "address", None),
                )

            # --- Sticker ---
            elif msg.HasField("stickerMessage"):
                try:
                    data = await client.download_any(msg)
                    import base64

                    b64 = base64.b64encode(data).decode()
                    url = f"data:image/webp;base64,{b64}"
                    content = MediaContent(url=url, mime_type="image/webp")
                except Exception:
                    logger.debug("Failed to download sticker, using placeholder")
                    content = MediaContent(
                        url="data:image/webp;base64,",
                        mime_type="image/webp",
                    )

            if content is None:
                logger.debug("Unsupported message type, skipping")
                return None

            return InboundMessage(
                channel_id=channel_id,
                sender_id=sender_id,
                content=content,
                external_id=getattr(info, "ID", None),
                metadata=metadata,
            )
        except Exception:
            logger.debug("Failed to parse neonize message", exc_info=True)
            return None

    return parser


# ---------------------------------------------------------------------------
# Source provider
# ---------------------------------------------------------------------------


class WhatsAppPersonalSourceProvider(BaseSourceProvider):
    """Event-driven WhatsApp personal account source via neonize.

    Maintains a persistent connection to WhatsApp using the multidevice
    protocol.  Inbound messages are parsed and emitted into RoomKit; lifecycle
    events (QR codes, auth, disconnect) are forwarded to ``on_event``.

    Example:
        from roomkit.sources.neonize import WhatsAppPersonalSourceProvider

        async def handle_events(event_type: str, data: dict):
            if event_type == "qr":
                print(f"Scan QR: {data['codes'][0]}")

        source = WhatsAppPersonalSourceProvider(
            db="wa-session.db",
            channel_id="wa-personal",
            on_event=handle_events,
        )
        await kit.attach_source("wa-personal", source)
    """

    # Platform type names that map to neonize DeviceProps enum values.
    PLATFORMS: dict[str, int] = {
        "chrome": 1,
        "firefox": 2,
        "safari": 5,
        "edge": 6,
        "desktop": 7,
    }

    # Receipt type codes from neonize protobuf → human-readable labels.
    RECEIPT_TYPES: dict[int, str] = {
        1: "delivered",
        2: "sender",
        3: "retry",
        4: "read",
        5: "read_self",
        6: "played",
        7: "played_self",
        8: "server_error",
        9: "inactive",
        10: "peer_msg",
        11: "history_sync",
    }

    def __init__(
        self,
        db: str = "whatsapp-session.db",
        channel_id: str = "whatsapp-personal",
        *,
        parser: NeonizeMessageParser | None = None,
        on_event: NeonizeEventCallback | None = None,
        device_name: str = "RoomKit",
        device_platform: str = "chrome",
        self_chat: bool = False,
    ) -> None:
        """Initialize WhatsApp personal source.

        Args:
            db: Neonize database path (SQLite file or PostgreSQL URI).
            channel_id: Channel ID for emitted inbound messages.
            parser: Async callable ``(client, event) -> InboundMessage | None``.
                If ``None``, uses the built-in multi-type parser.
            on_event: Optional async callback for lifecycle events.
                Called as ``on_event(event_type, data_dict)``.
            device_name: Device name shown in WhatsApp linked devices list.
            device_platform: Browser/platform label shown before device_name.
                One of: ``"chrome"``, ``"firefox"``, ``"safari"``, ``"edge"``,
                ``"desktop"``.  Defaults to ``"chrome"``.
            self_chat: When ``True``, process own messages so you can test
                by messaging yourself.
        """
        super().__init__()
        self._db = db
        self._channel_id = channel_id
        self._parser = parser or default_message_parser(channel_id, self_chat=self_chat)
        self._on_event = on_event
        self._device_name = device_name
        self._device_platform = device_platform.lower()
        self._client: Any = None
        # Cache LID → push_name for resolving presence event senders
        self._lid_names: dict[str, str] = {}

    @property
    def name(self) -> str:
        return f"neonize:{self._db}"

    @property
    def client(self) -> Any:
        """Expose the underlying neonize client for outbound use."""
        return self._client

    # -- lifecycle -----------------------------------------------------------

    async def start(self, emit: EmitCallback) -> None:  # noqa: C901
        """Connect to WhatsApp and start receiving messages."""
        if not HAS_NEONIZE:
            raise ImportError(
                "neonize is required for WhatsAppPersonalSourceProvider. "
                "Install it with: pip install roomkit[whatsapp-personal]"
            )

        self._reset_stop()
        self._set_status(SourceStatus.CONNECTING)

        # Import event types locally (only reachable when HAS_NEONIZE is True)
        import neonize.aioze.events as _neonize_events  # type: ignore[import-untyped]
        from neonize.aioze.events import (  # type: ignore[import-untyped]
            ChatPresenceEv,
            ConnectedEv,
            DisconnectedEv,
            LoggedOutEv,
            MessageEv,
            PairStatusEv,
            ReceiptEv,
        )

        # neonize creates its own event loop (event_global_loop) but never
        # starts it, so events dispatched via run_coroutine_threadsafe are
        # queued but never executed.  Patch it to the current running loop
        # BEFORE creating the client so all callbacks fire correctly.
        # Both modules must be patched because `from .events import
        # event_global_loop` in client.py creates a separate binding.
        import neonize.aioze.client as _neonize_client  # type: ignore[import-untyped]

        _running_loop = asyncio.get_running_loop()
        _neonize_events.event_global_loop = _running_loop
        _neonize_client.event_global_loop = _running_loop

        from neonize.proto.waCompanionReg.WAWebProtobufsCompanionReg_pb2 import (  # type: ignore[import-untyped]
            DeviceProps,
        )

        platform_type = self.PLATFORMS.get(self._device_platform, DeviceProps.CHROME)
        props = DeviceProps(os=self._device_name, platformType=platform_type)
        client = NewAClient(self._db, props=props)
        self._client = client

        # -- register event handlers -----------------------------------------

        @client.qr
        async def _on_qr(_: Any, data_qr: bytes) -> None:
            # data_qr is a comma-separated byte string with QR code refs
            codes = data_qr.decode(errors="replace").split(",") if data_qr else []
            await self._fire_event("qr", {"codes": codes})

        @client.event(PairStatusEv)
        async def _on_paired(_, event: Any) -> None:
            jid_obj = getattr(event, "ID", None)
            user = getattr(jid_obj, "User", "") if jid_obj else ""
            device = getattr(jid_obj, "Device", "") if jid_obj else ""
            raw = str(jid_obj) if jid_obj else ""
            await self._fire_event("authenticated", {
                "jid": raw,
                "user": user,
                "device": str(device),
            })

        @client.event(ConnectedEv)
        async def _on_connected(c: Any, __: Any) -> None:
            self._set_status(SourceStatus.CONNECTED)
            # Mark ourselves as "available" so WhatsApp relays presence
            # events (typing indicators) from other users.
            try:
                from neonize.utils.enum import Presence  # type: ignore[import-untyped]

                await c.send_presence(Presence.AVAILABLE)
                logger.debug("Presence set to AVAILABLE")
            except Exception:
                logger.debug("Failed to set presence", exc_info=True)
            await self._fire_event("connected", {})

        @client.event(LoggedOutEv)
        async def _on_logged_out(_, __: Any) -> None:
            self._set_status(SourceStatus.ERROR, "logged_out")
            await self._fire_event("logged_out", {})

        @client.event(DisconnectedEv)
        async def _on_disconnected(_, __: Any) -> None:
            self._set_status(SourceStatus.RECONNECTING)
            await self._fire_event("disconnected", {})

        @client.event(MessageEv)
        async def _on_message(c: Any, event: Any) -> None:
            try:
                # Cache JID/LID → push_name for presence event resolution.
                # Presence events use LID JIDs (xxx@lid) while message events
                # use phone JIDs (xxx@s.whatsapp.net).  Cache both forms.
                info = event.Info
                push_name = getattr(info, "Pushname", None)
                if push_name:
                    src = info.MessageSource
                    for jid_field in ("Sender", "SenderAlt"):
                        jid_str = self._format_jid(getattr(src, jid_field, None))
                        if jid_str:
                            self._lid_names[jid_str] = push_name

                # Reaction messages — fire as event, not as inbound message
                msg = event.Message
                if msg.HasField("reactionMessage"):
                    react = msg.reactionMessage
                    src = info.MessageSource
                    chat_user = getattr(src.Chat, "User", "")
                    chat_server = getattr(src.Chat, "Server", "")
                    sender_user = getattr(src.Sender, "User", "")
                    # key.ID is the message being reacted to
                    key = getattr(react, "key", None)
                    target_msg_id = getattr(key, "ID", "") if key else ""
                    emoji = getattr(react, "text", "") or ""
                    await self._fire_event("reaction", {
                        "chat_id": chat_user,
                        "chat_jid": f"{chat_user}@{chat_server}" if chat_user else "",
                        "sender_id": sender_user,
                        "is_from_me": bool(src.IsFromMe),
                        "target_message_id": target_msg_id,
                        "emoji": emoji,
                        "push_name": push_name,
                    })
                    return

                result = self._parser(c, event)
                if asyncio.iscoroutine(result):
                    message = await result
                else:
                    message = result
                if message is not None:
                    await emit(message)
                    self._record_message()
            except Exception:
                logger.warning("Error processing message", exc_info=True)

        @client.event(ReceiptEv)
        async def _on_receipt(_, event: Any) -> None:
            try:
                src = getattr(event, "MessageSource", None)
                raw_type = getattr(event, "Type", 0)
                receipt_type = self.RECEIPT_TYPES.get(raw_type, f"unknown({raw_type})")
                message_ids = list(getattr(event, "MessageIDs", []))
                sender_jid = self._format_jid(getattr(src, "Sender", None)) if src else ""
                sender_name = self._lid_names.get(sender_jid, "")
                await self._fire_event("receipt", {
                    "type": receipt_type,
                    "raw_type": raw_type,
                    "chat": self._format_jid(getattr(src, "Chat", None)) if src else "",
                    "sender": sender_jid,
                    "sender_name": sender_name,
                    "message_ids": message_ids,
                    "timestamp": getattr(event, "Timestamp", 0),
                })
            except Exception:
                logger.warning("Error processing receipt event", exc_info=True)

        @client.event(ChatPresenceEv)
        async def _on_presence(_, event: Any) -> None:
            try:
                src = getattr(event, "MessageSource", None)
                # State: 1=composing, 2=paused; Media: 1=text, 2=audio
                raw_state = getattr(event, "State", 0)
                raw_media = getattr(event, "Media", 0)
                state = "composing" if raw_state == 1 else "paused"
                media = "audio" if raw_media == 2 else "text"
                sender_jid = self._format_jid(getattr(src, "Sender", None)) if src else ""
                # Resolve LID to push_name if known from previous messages
                sender_name = self._lid_names.get(sender_jid, "")
                await self._fire_event("presence", {
                    "chat": self._format_jid(getattr(src, "Chat", None)) if src else "",
                    "sender": sender_jid,
                    "sender_name": sender_name,
                    "state": state,
                    "media": media,
                })
            except Exception:
                logger.warning("Error processing presence event", exc_info=True)

        # -- connect and idle ------------------------------------------------
        try:
            await client.connect()

            while not self._should_stop():
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._set_status(SourceStatus.ERROR, str(e))
            raise

    async def stop(self) -> None:
        """Disconnect from WhatsApp and stop receiving."""
        await super().stop()
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                logger.debug("Error disconnecting neonize client", exc_info=True)
            self._client = None
        logger.info("WhatsApp personal source stopped")

    # -- convenience send helpers --------------------------------------------

    async def send(self, jid: str, text: str) -> None:
        """Send a text message through the neonize client.

        Args:
            jid: Full JID (e.g. ``1234567890@s.whatsapp.net``).
            text: Message text.

        Raises:
            RuntimeError: If the client is not connected.
        """
        if self._client is None or self._status != SourceStatus.CONNECTED:
            raise RuntimeError("WhatsApp personal source not connected")
        await self._client.send_message(jid, text)

    async def send_composing(self, jid: str, *, media: str = "text") -> None:
        """Send a "composing" (typing) indicator to a chat.

        Args:
            jid: Full JID (e.g. ``1234567890@s.whatsapp.net``).
            media: ``"text"`` for typing or ``"audio"`` for recording.

        Raises:
            RuntimeError: If the client is not connected.
        """
        await self._send_presence(jid, composing=True, media=media)

    async def send_paused(self, jid: str) -> None:
        """Send a "paused" (stopped typing) indicator to a chat.

        Args:
            jid: Full JID (e.g. ``1234567890@s.whatsapp.net``).

        Raises:
            RuntimeError: If the client is not connected.
        """
        await self._send_presence(jid, composing=False)

    async def _send_presence(
        self, jid: str, *, composing: bool = True, media: str = "text"
    ) -> None:
        if self._client is None or self._status != SourceStatus.CONNECTED:
            raise RuntimeError("WhatsApp personal source not connected")
        from neonize.utils.enum import ChatPresence, ChatPresenceMedia  # type: ignore[import-untyped]

        state = (
            ChatPresence.CHAT_PRESENCE_COMPOSING
            if composing
            else ChatPresence.CHAT_PRESENCE_PAUSED
        )
        media_val = (
            ChatPresenceMedia.CHAT_PRESENCE_MEDIA_AUDIO
            if media == "audio"
            else ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT
        )
        jid_obj = self._parse_jid(jid)
        await self._client.send_chat_presence(jid_obj, state, media_val)

    async def mark_read(
        self,
        message_ids: list[str],
        chat: str,
        sender: str,
    ) -> None:
        """Mark messages as read (send blue ticks).

        Args:
            message_ids: List of message IDs to mark as read.
            chat: Chat JID (e.g. ``1234567890@s.whatsapp.net``).
            sender: Sender JID.

        Raises:
            RuntimeError: If the client is not connected.
        """
        if self._client is None or self._status != SourceStatus.CONNECTED:
            raise RuntimeError("WhatsApp personal source not connected")
        from neonize.utils.enum import ReceiptType  # type: ignore[import-untyped]
        from neonize.utils.jid import build_jid  # type: ignore[import-untyped]

        chat_jid = self._parse_jid(chat)
        sender_jid = self._parse_jid(sender)
        await self._client.mark_read(
            *message_ids,
            chat=chat_jid,
            sender=sender_jid,
            receipt=ReceiptType.READ,
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
            chat: Chat JID (e.g. ``1234567890@s.whatsapp.net``).
            sender: Sender JID of the message being reacted to.
            message_id: WhatsApp message ID to react to.
            emoji: Emoji reaction (empty string to remove).
        """
        if self._client is None or self._status != SourceStatus.CONNECTED:
            raise RuntimeError("WhatsApp personal source not connected")

        chat_jid = self._parse_jid(chat)
        sender_jid = self._parse_jid(sender)
        reaction_msg = await self._client.build_reaction(
            chat_jid, sender_jid, message_id, emoji,
        )
        await self._client.send_message(chat_jid, reaction_msg)

    @staticmethod
    def _parse_jid(jid: str) -> Any:
        """Parse a ``user@server`` string into a neonize JID protobuf."""
        from neonize.utils.jid import build_jid  # type: ignore[import-untyped]

        if "@" in jid:
            user, server = jid.split("@", 1)
            return build_jid(user, server)
        return build_jid(jid)

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _format_jid(jid_obj: Any) -> str:
        """Format a neonize JID protobuf as ``user@server``."""
        if jid_obj is None:
            return ""
        user = getattr(jid_obj, "User", "")
        server = getattr(jid_obj, "Server", "")
        if user and server:
            return f"{user}@{server}"
        return user or ""

    async def _fire_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Invoke the on_event callback if set."""
        if self._on_event is None:
            return
        result = self._on_event(event_type, data)
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            await result
