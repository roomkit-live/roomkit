"""Delivery and provider result models."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from roomkit.models.enums import EventType, Visibility
from roomkit.models.event import EventContent, RoomEvent
from roomkit.models.response_metadata import ResponseMetadata

if TYPE_CHECKING:

    class _CascadeLike(Protocol):
        """What a DeliveryHandle reads off the in-flight cascade.

        Structural on purpose: models do not import from
        :mod:`roomkit.core`, where :class:`DeliveryCascade` implements this.
        """

        delivery_results: dict[str, Any]
        error: Exception | None

        def waiter_would_deadlock(self) -> bool: ...


class ProviderResult(BaseModel):
    """Result from a provider delivery attempt."""

    success: bool
    provider_message_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InboundMessage(BaseModel):
    """A message received from an external provider.

    For stateful channels (voice, persistent WebSocket), set ``session``
    to the session object.  After the hook pipeline passes,
    ``process_inbound`` will call ``channel.connect_session()`` to bind
    the long-lived session to the room.
    """

    channel_id: str
    sender_id: str
    content: EventContent
    event_type: EventType = EventType.MESSAGE
    external_id: str | None = None
    # Provider-native thread reference (Slack ``thread_ts``, Discord message
    # snowflake, Teams ``replyToId``) — opaque, passed straight through to the
    # provider. NOT the in-app threading key; see ``parent_event_id``.
    thread_id: str | None = None
    # In-app threading: the event this message replies to. RoomKit normalises it
    # to the thread ROOT (flat two-level model) and the AI's reply inherits it,
    # so the response lands in the same thread. See ``RoomEvent.parent_event_id``.
    parent_event_id: str | None = None
    idempotency_key: str | None = None
    # The provider's payload exactly as it arrived, before any parsing (RFC
    # §5.2). It is the audit trail and the source of truth for provider-specific
    # data: a parser reads the handful of fields RoomKit models, and everything
    # else — delivery receipts, carrier annotations, fields a provider added
    # last week — survives only here. Carried onto ``EventSource.raw_payload``
    # unmodified.
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    # The provider's own id for this message, distinct from ``external_id``
    # (which parsers have historically used for the same value on some
    # providers and for the conversation id on others).
    provider_message_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    session: Any | None = None
    # Event visibility scope. The default ``"all"`` reaches every channel
    # (transports + intelligence). Set ``"transport"`` to deliver into a room
    # WITHOUT triggering its intelligence channel — e.g. a proactive
    # notification the agent should not react to.
    visibility: str = Visibility.ALL
    # Which intelligence channels this message asks to act (RFC §19.3), by
    # channel id. ``None`` addresses nobody in particular — every eligible
    # agent is solicited, or the router decides. How a caller *chooses* the
    # ids is its own business: a slash command, a picker, a mention syntax
    # parsed at the edge. RoomKit takes the decision, never the syntax.
    addressed_to: list[str] | None = None
    # Where the answer to this message may go, in the same vocabulary as
    # ``visibility``. ``None`` leaves it unrestricted. Set it when the reply
    # must stay as narrow as the question: a scope on the question alone
    # would hide what you asked and publish what you were told. Covers the
    # whole turn — text segments and tool activity alike.
    response_visibility: str | None = None


class DeliveryHandle:
    """A deferred caller's grip on its in-flight delivery (RFC §10.1 step 18).

    ``process_inbound(..., defer_delivery=True)`` returns at the commit; this
    handle, on :attr:`InboundResult.delivery`, is what remains of step 18:
    the event's delivery set, the reentry passes it transitively spawns (an
    AI reply included) and the consumption of streamed responses, all running
    in the room's delivery lane. ``wait()`` resolves once that whole tail has
    run — not merely the cascade, because a streamed reply is only generated
    while its stream is consumed, which starts after the cascade completes.

    The cascade is structurally typed (:class:`_CascadeLike`): models do not
    import from :mod:`roomkit.core`.
    """

    __slots__ = ("_cascade", "_consumer", "_result")

    def __init__(
        self, cascade: _CascadeLike, consumer: asyncio.Task[None], result: InboundResult
    ) -> None:
        self._cascade = cascade
        self._consumer = consumer
        self._result = result

    @property
    def done(self) -> bool:
        """Whether the deferred delivery will make no further progress.

        True once the turn finished — but also for a consumer cancelled by
        ``close()``, where the turn was abandoned mid-flight: this reports
        "nothing more will happen", not "everything ran". ``wait()``'s
        backfill is where the two read differently.
        """
        return self._consumer.done()

    async def wait(self) -> InboundResult:
        """Wait for the deferred delivery to complete, then report it.

        Backfills ``delivery_results`` and ``error`` on the result this
        handle belongs to — after this the result reads exactly like a
        non-deferred call's — and returns that result. Never raises: a
        consumer cancelled by ``close()`` resolves the wait too, with
        whatever the cascade recorded by then.

        Called from a context that must not wait on this room's delivery —
        the room's own lane executor (a tool handler) or under the room
        lock (a sync hook), where the lane cannot progress past the caller
        — it returns the result immediately, unwaited and un-backfilled:
        the same short-circuit the waiting path's step 18 applies, with
        delivery following in lane order.
        """
        if not self._consumer.done() and self._cascade.waiter_would_deadlock():
            return self._result
        await asyncio.wait({self._consumer})
        self._result.delivery_results = self._cascade.delivery_results
        if self._result.error is None:
            self._result.error = self._cascade.error
        return self._result


class InboundResult(BaseModel):
    """Result of processing an inbound message.

    ``error`` carries a generation/transport failure raised while consuming the
    intelligence channel's streaming response, so a headless caller (no
    streaming target to render an error card) can observe it and react —
    instead of the failure vanishing after ``ON_ERROR`` fires. ``None`` on
    success. Interactive callers ignore it; the ``ON_ERROR`` hooks still fire.

    ``response_metadata`` is the turn's own record, for the same reader and
    the same reason. It also rides each MESSAGE segment the turn persisted,
    but only segments that had text to carry: a turn ending on a tool call
    persists nothing after it, so the room cannot be asked how such a turn
    ended. The caller is handed the record instead of hunting for it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event: RoomEvent | None = None
    blocked: bool = False
    reason: str | None = None
    error: Exception | None = None
    response_metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)
    """The turn's response-metadata record; empty when no turn ran."""

    delivery_results: dict[str, DeliveryResult] = Field(default_factory=dict)
    """Per-channel outcome of this event's delivery set, keyed by channel id
    (RFC §10.1 step 18). ``process_inbound`` waits for that set to complete, so
    this is populated by the time it returns — for the caller's own event only,
    never for a reentry's, which is a separate event with its own result. A
    deferred call returns before the set executes: there it is backfilled by
    ``delivery.wait()`` instead."""

    delivery: DeliveryHandle | None = None
    """Set only by ``process_inbound(..., defer_delivery=True)``: the handle
    on the in-flight delivery (RFC §10.1 step 18 detached completion). ``None``
    on the waiting path, where the result already reports the completed set —
    and on a deferred call refused before the locked region (rate limited,
    pre-commit timeout, identity block), which has no delivery to follow.
    Whenever ``blocked`` is ``False`` the handle is there; a hook refusal,
    decided inside the locked region, gets one too (its near-empty cascade
    resolves at once)."""


class DeliveryError(BaseModel):
    """Why a delivery failed, in terms a caller can act on (RFC §5.13)."""

    code: str
    """Machine-readable code. The exception's own ``code`` when it carries one,
    otherwise its type name — enough to branch on without parsing prose."""

    message: str
    """Human-readable description."""

    retryable: bool = True
    """Whether a retry may succeed. Read from the exception's own ``retryable``
    when it declares one, matching what the delivery retry loop decides
    (§13.2); an error that says nothing about itself is reported as retryable,
    which is also how the loop treats it."""


class DeliveryResult(BaseModel):
    """The outcome of delivering one event to one channel (RFC §5.13)."""

    channel_id: str
    status: Literal["sent", "queued", "failed"]
    provider_message_id: str | None = None
    error: DeliveryError | None = None
    retry_after: datetime | None = None
    provider_result: ProviderResult | None = None


class DeliveryStatus(BaseModel):
    """Status update for an outbound message from a provider webhook.

    Providers send status webhooks when messages are sent, delivered, failed, etc.
    Use this with the ON_DELIVERY_STATUS hook to track outbound message delivery.

    Attributes:
        provider: Provider name (e.g., "telnyx", "twilio").
        message_id: Provider's unique message identifier.
        status: Status string (e.g., "sent", "delivered", "failed").
        recipient: Phone number/address the message was sent to.
        sender: Phone number/address the message was sent from.
        error_code: Provider-specific error code (if failed).
        error_message: Human-readable error message (if failed).
        timestamp: When the status was reported.
        raw: Original webhook payload for debugging.
    """

    room_id: str | None = None
    channel_id: str | None = None
    provider: str
    message_id: str
    status: str
    recipient: str = ""
    sender: str = ""
    error_code: str | None = None
    error_message: str | None = None
    timestamp: datetime | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_timestamp(cls, v: str | datetime | None) -> datetime | None:
        if v is None or isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        raise ValueError(f"Cannot parse timestamp from {type(v).__name__}")
