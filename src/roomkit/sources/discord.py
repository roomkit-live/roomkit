"""Discord gateway event source for RoomKit.

Holds a single ``discord.Client`` that the paired
:class:`~roomkit.providers.discord.DiscordBotProvider` reuses for outbound
sends — one persistent gateway connection serves both directions, mirroring
the WhatsApp-personal / neonize pattern.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import warnings
from collections.abc import Awaitable, Callable
from typing import Any

from roomkit.models.delivery import InboundMessage
from roomkit.models.event import EventContent, MediaContent, TextContent
from roomkit.providers.discord.config import DiscordConfig
from roomkit.sources.base import BaseSourceProvider, EmitCallback, SourceStatus

# Optional dependency --------------------------------------------------------
# discord.py imports the stdlib ``audioop`` module (for voice), which is
# deprecated in Python 3.12+. Suppress that third-party DeprecationWarning at
# the import site so it never leaks into apps running under ``-W error``.
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import discord

    HAS_DISCORD = True
except ImportError:
    discord = None  # ty: ignore[invalid-assignment]
    HAS_DISCORD = False

logger = logging.getLogger("roomkit.sources.discord")

# Type aliases ---------------------------------------------------------------
# A parser maps a discord Message (duck-typed) + the bot's own user id to an
# InboundMessage, or None to skip the message.
DiscordMessageParser = Callable[[Any, int | None], InboundMessage | None]
# Reaction lifecycle callback (sync or async), receiving a normalised dict.
DiscordEventCallback = Callable[[dict[str, str]], Awaitable[None] | None]


def _build_content(text: str, attachments: list[Any]) -> EventContent | None:
    """Build event content from message text + attachments.

    The first attachment becomes a :class:`MediaContent` (with the text as its
    caption); text-only messages become :class:`TextContent`. Returns ``None``
    when the message carries neither.
    """
    if attachments:
        att = attachments[0]
        url = getattr(att, "url", "")
        if url:
            return MediaContent(
                url=url,
                mime_type=getattr(att, "content_type", None) or "application/octet-stream",
                filename=getattr(att, "filename", None),
                size_bytes=getattr(att, "size", None),
                caption=text or None,
            )
    if text:
        return TextContent(body=text)
    return None


def parse_discord_message(
    message: Any,
    channel_id: str,
    *,
    bot_user_id: int | None = None,
    ignore_bots: bool = True,
) -> InboundMessage | None:
    """Convert a ``discord.Message`` into an :class:`InboundMessage`.

    Duck-typed (uses ``getattr``) so it can be unit-tested with a plain stub
    object — no ``discord`` instance required. Returns ``None`` to skip the
    bot's own messages, other bots (when ``ignore_bots``), and empty messages.
    """
    author = getattr(message, "author", None)
    author_id = getattr(author, "id", None)
    if bot_user_id is not None and author_id == bot_user_id:
        return None
    if ignore_bots and getattr(author, "bot", False):
        return None

    attachments = list(getattr(message, "attachments", []) or [])
    text = getattr(message, "content", "") or ""
    content = _build_content(text, attachments)
    if content is None:
        return None

    channel = getattr(message, "channel", None)
    guild = getattr(message, "guild", None)
    message_id = str(getattr(message, "id", ""))

    metadata: dict[str, Any] = {
        "guild_id": str(getattr(guild, "id", "")) if guild is not None else "",
        "channel_id": str(getattr(channel, "id", "")),
        "channel_name": getattr(channel, "name", "") or "",
        "author_name": getattr(author, "display_name", None) or getattr(author, "name", ""),
        "author_bot": bool(getattr(author, "bot", False)),
        "message_id": message_id,
    }
    if len(attachments) > 1:
        metadata["attachment_urls"] = [getattr(a, "url", "") for a in attachments]

    reference = getattr(message, "reference", None)
    ref_id = getattr(reference, "message_id", None) if reference is not None else None
    thread_id = str(ref_id) if ref_id else None

    return InboundMessage(
        channel_id=channel_id,
        sender_id=str(author_id) if author_id is not None else "",
        content=content,
        external_id=message_id,
        idempotency_key=message_id,
        thread_id=thread_id,
        metadata=metadata,
    )


def default_message_parser(channel_id: str, *, ignore_bots: bool = True) -> DiscordMessageParser:
    """Create a parser bound to ``channel_id`` and the ``ignore_bots`` policy."""

    def parser(message: Any, bot_user_id: int | None) -> InboundMessage | None:
        return parse_discord_message(
            message, channel_id, bot_user_id=bot_user_id, ignore_bots=ignore_bots
        )

    return parser


class DiscordGatewaySource(BaseSourceProvider):
    """Persistent Discord gateway connection emitting inbound messages.

    Owns the ``discord.Client`` and exposes it via :attr:`client` so the
    paired provider can send through the same connection. Reactions are
    surfaced to ``on_event`` (not the message pipeline), matching how Teams
    and WhatsApp-personal handle reaction lifecycle events.
    """

    def __init__(
        self,
        config: DiscordConfig,
        channel_id: str = "discord",
        *,
        parser: DiscordMessageParser | None = None,
        on_event: DiscordEventCallback | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._channel_id = channel_id
        self._parser = parser or default_message_parser(channel_id, ignore_bots=config.ignore_bots)
        self._on_event = on_event
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Expose the underlying discord client for outbound use."""
        return self._client

    @property
    def name(self) -> str:
        return f"discord:{self._channel_id}"

    async def start(self, emit: EmitCallback) -> None:
        if not HAS_DISCORD:
            raise ImportError(
                "discord.py is required for DiscordGatewaySource. "
                "Install it with: pip install roomkit[discord]"
            )

        self._reset_stop()
        self._set_status(SourceStatus.CONNECTING)

        intents = discord.Intents.default()
        intents.message_content = self._config.intents_message_content
        client = discord.Client(intents=intents)
        self._client = client

        @client.event
        async def on_ready() -> None:
            self._set_status(SourceStatus.CONNECTED)
            guilds = ", ".join(g.name for g in client.guilds) or "<none>"
            logger.info(
                "Discord gateway ready as %s in %d server(s): %s",
                client.user,
                len(client.guilds),
                guilds,
            )

        @client.event
        async def on_message(message: Any) -> None:
            bot_user_id = client.user.id if client.user else None
            parsed = self._parser(message, bot_user_id)
            if parsed is not None:
                await emit(parsed)
                self._record_message()

        @client.event
        async def on_raw_reaction_add(payload: Any) -> None:
            await self._dispatch_reaction("add", payload)

        @client.event
        async def on_raw_reaction_remove(payload: Any) -> None:
            await self._dispatch_reaction("remove", payload)

        try:
            await client.start(self._config.bot_token.get_secret_value())
        except Exception as exc:
            self._set_status(SourceStatus.ERROR, str(exc))
            raise
        finally:
            # Close the client even when start() failed (e.g. bad token) so the
            # aiohttp connector it opened during login is released, not leaked.
            if not client.is_closed():
                with contextlib.suppress(Exception):
                    await client.close()
            self._client = None

    async def stop(self) -> None:
        """Disconnect from the Discord gateway and stop receiving."""
        await super().stop()
        client = self._client
        if client is not None:
            try:
                await client.close()
            except Exception:
                logger.debug("Error closing Discord client", exc_info=True)
            self._client = None
        logger.info("Discord source stopped")

    async def _dispatch_reaction(self, action: str, payload: Any) -> None:
        """Forward a raw reaction event to ``on_event`` as a normalised dict."""
        if self._on_event is None:
            return
        emoji = getattr(payload, "emoji", None)
        normalised = {
            "action": action,
            "emoji": str(emoji) if emoji is not None else "",
            "user_id": str(getattr(payload, "user_id", "")),
            "message_id": str(getattr(payload, "message_id", "")),
            "channel_id": str(getattr(payload, "channel_id", "")),
        }
        result = self._on_event(normalised)
        if inspect.isawaitable(result):
            await result
