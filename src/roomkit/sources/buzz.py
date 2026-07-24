"""Buzz (Nostr relay) event source for RoomKit.

Wraps a :class:`buzzkit.BuzzClient`: authenticates to the relay (NIP-42) and
streams one channel's messages into the inbound pipeline. The paired
:class:`~roomkit.providers.buzz.BuzzProvider` reuses the same client for
outbound sends, so a single Nostr identity serves both directions.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from roomkit.providers.buzz.config import BuzzConfig
from roomkit.sources.base import BaseSourceProvider, EmitCallback, SourceStatus

# Optional dependency --------------------------------------------------------
# ``buzzkit`` is a compiled wheel kept out of RoomKit's own dev/CI env. It is
# typed as Any (via the TYPE_CHECKING branch) so type-checking stays stable
# whether or not the package is resolvable; the runtime guard handles absence.
if TYPE_CHECKING:
    BuzzClient: Any = None
    HAS_BUZZKIT = True
else:
    try:
        from buzzkit import BuzzClient

        HAS_BUZZKIT = True
    except ImportError:
        BuzzClient = None
        HAS_BUZZKIT = False

logger = logging.getLogger("roomkit.sources.buzz")

# A parser maps a Nostr event (dict) + the agent's own pubkey hex to an
# InboundMessage, or None to skip the event.
BuzzMessageParser = Callable[[dict[str, Any], str | None], InboundMessage | None]

_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 30.0
_PRESENCE_INTERVAL = 55.0  # presence TTL is 90s; re-announce well within it


def parse_buzz_event(
    event: dict[str, Any],
    channel_id: str,
    *,
    own_pubkey: str | None = None,
    ignore_own: bool = True,
) -> InboundMessage | None:
    """Convert a Nostr event dict into an :class:`InboundMessage`.

    Duck-typed on a plain dict so it can be unit-tested without a relay.
    Returns ``None`` to skip the agent's own events (echo guard) and events
    with no text content.
    """
    pubkey = str(event.get("pubkey", ""))
    if ignore_own and own_pubkey and pubkey == own_pubkey:
        return None
    text = event.get("content", "") or ""
    if not text:
        return None

    event_id = str(event.get("id", ""))
    tags = event.get("tags") or []
    relay_channel = next((t[1] for t in tags if len(t) >= 2 and t[0] == "h"), "")
    metadata: dict[str, Any] = {
        "nostr_event_id": event_id,
        "nostr_kind": event.get("kind"),
        "buzz_channel_id": relay_channel,
    }
    return InboundMessage(
        channel_id=channel_id,
        sender_id=pubkey,
        content=TextContent(body=text),
        external_id=event_id,
        idempotency_key=event_id,
        metadata=metadata,
    )


def default_message_parser(channel_id: str, *, ignore_own: bool = True) -> BuzzMessageParser:
    """Create a parser bound to ``channel_id`` and the ``ignore_own`` policy."""

    def parser(event: dict[str, Any], own_pubkey: str | None) -> InboundMessage | None:
        return parse_buzz_event(event, channel_id, own_pubkey=own_pubkey, ignore_own=ignore_own)

    return parser


class BuzzRelaySource(BaseSourceProvider):
    """Persistent Buzz relay connection emitting one channel's messages.

    Owns the :class:`buzzkit.BuzzClient` and exposes it via :attr:`client` so
    the paired provider can send through the same identity. Subscribes to a
    single relay channel (``relay_channel_id``); register one source per Buzz
    channel and bind each to its RoomKit room.
    """

    def __init__(
        self,
        config: BuzzConfig,
        channel_id: str = "buzz",
        *,
        relay_channel_id: str,
        parser: BuzzMessageParser | None = None,
        kinds: list[int] | None = None,
    ) -> None:
        """``kinds`` selects the Nostr event kinds to subscribe to (default:
        chat messages, kind 9). Pass other kinds — e.g. huddle announcements,
        kind 48100 — together with a ``parser`` that knows how to convert
        them; the default parser only understands text messages."""
        super().__init__()
        if not HAS_BUZZKIT:
            raise ImportError(
                "buzzkit is required for BuzzRelaySource. "
                "Install it with: pip install roomkit[buzz]"
            )
        self._config = config
        self._channel_id = channel_id
        self._relay_channel_id = relay_channel_id
        self._kinds = kinds
        self._parser = parser or default_message_parser(channel_id, ignore_own=config.ignore_own)
        self._client: Any = BuzzClient(
            config.relay_url,
            config.private_key.get_secret_value(),
            auth_tag=config.auth_tag,
        )

    @property
    def client(self) -> Any:
        """Expose the underlying BuzzClient for outbound use."""
        return self._client

    @property
    def name(self) -> str:
        return f"buzz:{self._channel_id}"

    async def _join_channel(self) -> None:
        """Best-effort NIP-29 self-join so the agent is a channel member."""
        try:
            result = await self._client.join_channel(self._relay_channel_id)
        except Exception as exc:
            logger.warning("Buzz auto-join failed for %s: %s", self._relay_channel_id, exc)
            return
        if not result.get("accepted", False):
            logger.info(
                "Buzz auto-join not accepted for %s: %s",
                self._relay_channel_id,
                result.get("message", ""),
            )

    async def _presence_loop(self) -> None:
        """Announce presence (kind 20001) on connect, then heartbeat within TTL."""
        while not self._should_stop():
            try:
                await self._client.publish_presence("online")
            except Exception as exc:
                logger.debug("Buzz presence publish failed: %s", exc)
                return
            await asyncio.sleep(_PRESENCE_INTERVAL)

    async def start(self, emit: EmitCallback) -> None:
        self._reset_stop()
        self._set_status(SourceStatus.CONNECTING)
        backoff = _INITIAL_BACKOFF
        while not self._should_stop():
            presence_task: asyncio.Task | None = None
            try:
                await self._client.connect()
                self._set_status(SourceStatus.CONNECTED)
                backoff = _INITIAL_BACKOFF
                if self._config.auto_join:
                    await self._join_channel()
                if self._config.announce_presence:
                    presence_task = asyncio.create_task(self._presence_loop())
                async for event in self._client.subscribe_channel(
                    self._relay_channel_id, kinds=self._kinds
                ):
                    if self._should_stop():
                        break
                    parsed = self._parser(event, self._client.pubkey_hex)
                    if parsed is not None:
                        await emit(parsed)
                        self._record_message()
            except Exception as exc:
                self._set_status(SourceStatus.ERROR, str(exc))
                logger.warning("Buzz source %s error: %s", self._channel_id, exc)
            finally:
                if presence_task is not None:
                    presence_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await presence_task
                with contextlib.suppress(Exception):
                    await self._client.close()
            if self._should_stop():
                break
            self._set_status(SourceStatus.RECONNECTING)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF)
        self._set_status(SourceStatus.STOPPED)

    async def stop(self) -> None:
        """Stop receiving and close the relay connection."""
        await super().stop()
        with contextlib.suppress(Exception):
            await self._client.close()
        logger.info("Buzz source %s stopped", self._channel_id)
