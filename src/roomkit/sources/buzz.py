"""Buzz (Nostr relay) event source for RoomKit.

Wraps a :class:`buzzkit.BuzzClient`: authenticates to the relay (NIP-42) and
streams one channel's messages into the inbound pipeline. The paired
:class:`~roomkit.providers.buzz.BuzzProvider` reuses the same client for
outbound sends, so a single Nostr identity serves both directions.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
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

# Out-of-band event callback (reactions, deletions): receives a normalised
# dict and may be sync or async — matching how Discord and WhatsApp-personal
# surface reaction lifecycle events outside the message pipeline.
BuzzEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

#: Nostr kind for Buzz channel chat messages.
KIND_STREAM_MESSAGE = 9
#: Nostr kind for NIP-25 reactions (content = emoji, ``e`` tag = target).
KIND_REACTION = 7
#: Nostr kind for NIP-09 deletions (Buzz uses it to retract reactions).
KIND_DELETION = 5

_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 30.0
# Presence heartbeat cadence. Relays at buzz >= 0.5.x hold presence for a 180 s
# TTL and expect a beat every 60 s; older relays used 90 s / 30 s. 30 s is the
# cadence buzzkit documents as safe on both when the relay version is unknown.
_PRESENCE_INTERVAL = 30.0
# WebSocket close code the relay sends on graceful restart (buzzkit >= 0.2.0
# exposes it as ``BuzzClient.close_code``): reconnect promptly, not an error.
_CLOSE_GRACEFUL_RESTART = 1012


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
    # NIP-10 threading: a direct reply carries one ["e", <root>, "", "reply"]
    # tag; a nested reply adds ["e", <root>, "", "root"] beside it. Either way
    # the thread ROOT is the "root"-marked id when present, else the "reply"
    # one — matching RoomKit's flat two-level model (thread_id = root, like
    # Slack's thread_ts).
    root_id = next((t[1] for t in tags if len(t) >= 4 and t[0] == "e" and t[3] == "root"), "")
    reply_id = next((t[1] for t in tags if len(t) >= 4 and t[0] == "e" and t[3] == "reply"), "")
    thread_root = root_id or reply_id
    if reply_id:
        metadata["nostr_reply_to"] = reply_id
    return InboundMessage(
        channel_id=channel_id,
        sender_id=pubkey,
        content=TextContent(body=text),
        external_id=event_id,
        idempotency_key=event_id,
        thread_id=thread_root or None,
        metadata=metadata,
    )


def default_message_parser(channel_id: str, *, ignore_own: bool = True) -> BuzzMessageParser:
    """Create a parser bound to ``channel_id`` and the ``ignore_own`` policy."""

    def parser(event: dict[str, Any], own_pubkey: str | None) -> InboundMessage | None:
        return parse_buzz_event(event, channel_id, own_pubkey=own_pubkey, ignore_own=ignore_own)

    return parser


def parse_buzz_reaction(event: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise a kind-7 reaction or kind-5 deletion into a flat dict.

    Returns ``None`` for other kinds or a reaction with no target. The dict
    mirrors the Discord/WhatsApp-personal reaction shape: ``action`` is
    ``"add"`` (kind 7) or ``"remove"`` (kind 5 — Buzz retracts a reaction by
    deleting the reaction event, so ``reaction_event_id`` is what got removed
    and the emoji/target are unknown to the wire).
    """
    kind = event.get("kind")
    tags = event.get("tags") or []
    target = next((t[1] for t in tags if len(t) >= 2 and t[0] == "e"), "")
    if not target:
        return None
    if kind == KIND_REACTION:
        return {
            "action": "add",
            "emoji": str(event.get("content", "") or ""),
            "user_id": str(event.get("pubkey", "")),
            "target_event_id": target,
            "reaction_event_id": str(event.get("id", "")),
        }
    if kind == KIND_DELETION:
        return {
            "action": "remove",
            "user_id": str(event.get("pubkey", "")),
            "reaction_event_id": target,
        }
    return None


#: Nostr kind for Buzz huddle announcements ("a huddle just started here").
KIND_HUDDLE_STARTED = 48100


def huddle_announcement_parser(
    channel_id: str, *, started_after: int | None = None
) -> BuzzMessageParser:
    """Parser for huddle announcements (kind 48100).

    Emits one :class:`InboundMessage` per announcement with the ephemeral
    huddle id in ``metadata["ephemeral_channel_id"]``. Subscribe the source
    with ``kinds=[KIND_HUDDLE_STARTED]``.

    ``started_after`` (unix seconds) drops announcements replayed from relay
    history — the subscription replays recent events before EOSE, and a
    restarted agent must not chase long-dead huddles.
    """

    def parser(event: dict[str, Any], own_pubkey: str | None) -> InboundMessage | None:
        if event.get("kind") != KIND_HUDDLE_STARTED:
            return None
        if started_after is not None and int(event.get("created_at") or 0) < started_after:
            return None
        try:
            huddle_id = json.loads(event.get("content") or "{}")["ephemeral_channel_id"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        event_id = str(event.get("id", ""))
        return InboundMessage(
            channel_id=channel_id,
            sender_id=str(event.get("pubkey", "")),
            content=TextContent(body=f"huddle started: {huddle_id}"),
            external_id=event_id,
            idempotency_key=event_id,
            metadata={"ephemeral_channel_id": str(huddle_id)},
        )

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
        on_event: BuzzEventCallback | None = None,
    ) -> None:
        """``kinds`` selects the Nostr event kinds to subscribe to (default:
        chat messages, kind 9). Pass other kinds — e.g. huddle announcements,
        kind 48100 — together with a ``parser`` that knows how to convert
        them; the default parser only understands text messages.

        ``on_event`` surfaces reaction lifecycle events (kind 7 add, kind 5
        remove) as normalised dicts, outside the message pipeline — matching
        how Discord and WhatsApp-personal handle reactions. Providing it
        widens the default subscription to kinds 9, 7 and 5; requires a relay
        that scopes reactions to their target's channel (buzzkit>=0.2.0)."""
        super().__init__()
        if not HAS_BUZZKIT:
            raise ImportError(
                "buzzkit is required for BuzzRelaySource. "
                "Install it with: pip install roomkit[buzz]"
            )
        self._config = config
        self._channel_id = channel_id
        self._relay_channel_id = relay_channel_id
        if kinds is None and on_event is not None:
            kinds = [KIND_STREAM_MESSAGE, KIND_REACTION, KIND_DELETION]
        self._kinds = kinds
        self._on_event = on_event
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

    async def _dispatch_reaction(self, event: dict[str, Any]) -> None:
        """Forward a reaction/deletion to ``on_event`` — never into the pipeline."""
        if self._on_event is None:
            return
        # The agent's own reactions echo back like any event; skip them under
        # the same policy as messages.
        if self._config.ignore_own and str(event.get("pubkey", "")) == self._client.pubkey_hex:
            return
        normalised = parse_buzz_reaction(event)
        if normalised is None:
            return
        normalised["channel_id"] = self._relay_channel_id
        try:
            result = self._on_event(normalised)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Buzz on_event callback failed")

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
                    if event.get("kind") in (KIND_REACTION, KIND_DELETION):
                        await self._dispatch_reaction(event)
                        continue
                    parsed = self._parser(event, self._client.pubkey_hex)
                    if parsed is not None:
                        await emit(parsed)
                        self._record_message()
            except Exception as exc:
                if getattr(self._client, "close_code", None) == _CLOSE_GRACEFUL_RESTART:
                    # Deliberate relay restart — replayed events are deduped by
                    # id (idempotency_key), so reconnect promptly and quietly.
                    logger.info(
                        "Buzz source %s: relay restarting (close code 1012), reconnecting",
                        self._channel_id,
                    )
                else:
                    self._set_status(SourceStatus.ERROR, str(exc))
                    logger.warning("Buzz source %s error: %s", self._channel_id, exc)
            finally:
                if presence_task is not None:
                    presence_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await presence_task
                graceful = getattr(self._client, "close_code", None) == _CLOSE_GRACEFUL_RESTART
                with contextlib.suppress(Exception):
                    await self._client.close()
            if self._should_stop():
                break
            self._set_status(SourceStatus.RECONNECTING)
            if graceful:
                backoff = _INITIAL_BACKOFF
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF)
        self._set_status(SourceStatus.STOPPED)

    async def stop(self) -> None:
        """Stop receiving and close the relay connection."""
        await super().stop()
        if self._config.leave_on_stop:
            # Best-effort NIP-29 leave (kind 9022) while the socket may still
            # be up. Opt-in: see ``BuzzConfig.leave_on_stop`` for the private-
            # channel lockout caveat.
            with contextlib.suppress(Exception):
                await self._client.leave_channel(self._relay_channel_id)
        with contextlib.suppress(Exception):
            await self._client.close()
        logger.info("Buzz source %s stopped", self._channel_id)
