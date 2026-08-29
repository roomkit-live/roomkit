"""Abstract base class for channels."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from roomkit.core.task_utils import log_task_exception
from roomkit.models.channel import ChannelBinding, ChannelCapabilities, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
)
from roomkit.models.event import RoomEvent, TextContent

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.trace import ProtocolTrace

# Callback type for protocol trace observers
TraceCallback = Callable[["ProtocolTrace"], Any]

_trace_logger = logging.getLogger("roomkit.trace")


def _safe_invoke(cb: TraceCallback, trace: ProtocolTrace) -> None:
    """Invoke a trace callback, scheduling coroutines as tasks."""
    try:
        result = cb(trace)
        if inspect.iscoroutine(result):
            with contextlib.suppress(RuntimeError):
                task = asyncio.get_running_loop().create_task(result)
                task.add_done_callback(log_task_exception)
    except Exception:
        _trace_logger.exception("Trace callback error")


class FrameworkAwareChannel(ABC):
    """A channel that is handed the framework it is registered with.

    Session-based channels — voice, realtime voice, video, conference — route
    inbound media and fire hooks themselves, which needs a reference back to the
    :class:`~roomkit.core.framework.RoomKit` instance.  Registration passes it to
    every channel that inherits this class, and only to those.

    Inheriting is the opt-in, and the framework needs no edit to serve a channel
    that declares it.  It is also what makes the selection safe — a channel that
    merely happens to own a method named ``set_framework`` is not called with an
    argument it never expected, and a subclass whose override does not match the
    signature below is a type error.

    Example::

        class MyChannel(FrameworkAwareChannel, Channel):
            def set_framework(self, framework: RoomKit) -> None:
                self._framework = framework
    """

    @abstractmethod
    def set_framework(self, framework: RoomKit) -> None:
        """Receive the framework this channel was registered with."""


class Channel(ABC):
    """Base class for all channels."""

    channel_type: ChannelType
    category: ChannelCategory = ChannelCategory.TRANSPORT
    direction: ChannelDirection = ChannelDirection.BIDIRECTIONAL

    sender_is_participant: bool = False
    """Whether this channel's ``sender_id`` is a room ``Participant.id``.

    An identity resolver maps an *address* — a number, an email, a handle — to
    an Identity.  Most channels carry one: what arrives on ``sender_id`` is how
    the sender is reachable, and who that is remains to be looked up.  A channel
    that sets this declares the opposite: its senders are named by the room
    itself, so there is no address to look up and identity resolution (RFC §11)
    is skipped for its messages.

    Declaring it wrongly is not a small mistake: a channel that does carry
    addresses would stop resolving them, and every sender would stay
    unidentified.
    """

    def __init__(self, channel_id: str) -> None:
        self.channel_id = channel_id
        self._provider: Any = None
        self._trace_callbacks: list[tuple[TraceCallback, frozenset[str] | None]] = []
        self._trace_framework_handler: TraceCallback | None = None

    # -------------------------------------------------------------------------
    # Protocol trace
    # -------------------------------------------------------------------------

    @property
    def trace_enabled(self) -> bool:
        """Whether any trace observers are registered."""
        return bool(self._trace_callbacks) or self._trace_framework_handler is not None

    def on_trace(
        self,
        callback: TraceCallback,
        *,
        protocols: list[str] | None = None,
    ) -> None:
        """Register a protocol trace observer.

        Args:
            callback: Called with each :class:`ProtocolTrace`.  May be sync
                or async (coroutines are scheduled as tasks).
            protocols: Optional allowlist of protocol names (e.g.
                ``["sip"]``).  ``None`` means all protocols.
        """
        self._trace_callbacks.append((callback, frozenset(protocols) if protocols else None))

    def emit_trace(self, trace: ProtocolTrace) -> None:
        """Emit a protocol trace to all registered observers."""
        for cb, protocols in self._trace_callbacks:
            if protocols is None or trace.protocol in protocols:
                _safe_invoke(cb, trace)
        if self._trace_framework_handler is not None:
            _safe_invoke(self._trace_framework_handler, trace)

    def resolve_trace_room(self, session_id: str | None) -> str | None:  # noqa: B027
        """Resolve a room ID for a trace with the given session.

        Override in session-based channels (voice, realtime voice) to
        map session IDs to room IDs.  Returns ``None`` by default.
        """
        return None

    @property
    def provider_name(self) -> str | None:
        """Provider or backend name for event attribution."""
        p = self._provider
        return p.name if p is not None and hasattr(p, "name") else None

    # -------------------------------------------------------------------------
    # Channel metadata
    # -------------------------------------------------------------------------

    @property
    def info(self) -> dict[str, Any]:
        """Return channel metadata. Override in subclasses."""
        return {}

    @property
    def active_turns(self) -> int:
        """How many turns this channel is running right now.

        What a caller retiring the object needs to know: a channel taken out
        of the registry (displaced by a rebuild, or removed with the thing it
        served) may still be answering for a turn that captured it, and
        ``close`` on most intelligence channels cancels that turn rather than
        waiting for it. A channel that counts its turns answers here so the
        caller can wait until it is idle; one that does not count them
        answers 0 and is treated as idle, which is the behaviour it had
        before the property existed.
        """
        return 0

    @property
    def recent_events_window(self) -> int:
        """How many recent room events this channel reads per turn.

        Drives how many events the framework loads into ``RoomContext`` for a
        room. Transport-only channels (WebSocket, realtime voice) don't consume
        room history, so the default is 0; AI channels override it with their
        memory provider's window.
        """
        return 0

    @abstractmethod
    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        """Process an inbound message into a RoomEvent."""

    @abstractmethod
    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Deliver an event to this channel."""

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """React to an event. Default: no-op for transport channels."""
        return ChannelOutput.empty()

    @property
    def supports_streaming_delivery(self) -> bool:
        """Whether this channel can accept streaming text delivery."""
        return False

    def supports_streaming_delivery_for(self, room_id: str) -> bool:
        """Whether this channel can stream *into a specific room*.

        Defaults to the channel-wide answer, which is right for a channel whose
        capability does not vary by room — a voice session, a terminal. A
        channel that holds per-room client connections overrides this, so a
        room whose clients cannot stream does not take the streaming path only
        to fall back at the end.
        """
        return self.supports_streaming_delivery

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[str],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        """Deliver a streaming text response to this channel.

        Default: accumulate text, deliver as complete event.
        """
        chunks: list[str] = []
        async for chunk in text_stream:
            if isinstance(chunk, str):
                chunks.append(chunk)
        updated = event.model_copy(update={"content": TextContent(body="".join(chunks))})
        return await self.deliver(updated, binding, context)

    async def connect_session(  # noqa: B027
        self,
        session: Any,
        room_id: str,
        binding: ChannelBinding,
    ) -> None:
        """Accept a long-lived session after inbound processing.

        Called by ``process_inbound`` when ``message.session`` is present
        and hooks did not block.  Override in session-based channels
        (voice, persistent WebSocket, etc.).  Default: no-op.
        """

    async def disconnect_session(  # noqa: B027
        self,
        session: Any,
        room_id: str,
    ) -> None:
        """Clean up a session on remote disconnect.

        Override in session-based channels to release resources.
        Default: no-op.
        """

    def update_binding(  # noqa: B027
        self,
        room_id: str,
        binding: ChannelBinding,
    ) -> None:
        """Notify the channel that a room's binding has changed.

        Called by the framework after ``mute()``, ``unmute()``, or
        ``set_access()`` update the store.  Override in session-based
        channels (voice, realtime voice) to update cached binding state
        used for audio gating.  Default: no-op.
        """

    async def on_room_attached(  # noqa: B027
        self,
        room_id: str,
        binding: ChannelBinding,
    ) -> None:
        """Establish whatever the new binding claims exists.

        Awaited by ``attach_channel()`` after the binding is written and before
        anything has observed it — no system event, no hook.  This is where a
        channel does the outside-world work an attachment implies: a conference
        channel creates the SFU room here (RFC §12.10.4 step 1).

        Raising cancels the attachment.  The framework takes the binding back
        and re-raises, so the caller learns that the channel refused rather than
        receiving a binding to something that was never built.  Which is why
        this is not a hook: a lifecycle hook is observation, its errors are
        logged and never raised, and the room would go on believing it was
        attached.  Default: no-op.
        """

    async def on_room_detached(  # noqa: B027
        self,
        room_id: str,
    ) -> None:
        """Take down what :meth:`on_room_attached` established.

        Awaited by ``detach_channel()`` before the ``ON_CHANNEL_DETACHED``
        hooks, so an integrator's handler runs after the channel has finished
        letting go rather than alongside it.

        Nothing is rolled back if this raises — the binding is already gone and
        the detach already announced — but the error reaches the caller instead
        of disappearing into a log.  Default: no-op.
        """

    def capabilities(self) -> ChannelCapabilities:
        """Return channel capabilities."""
        return ChannelCapabilities(media_types=[ChannelMediaType.TEXT])

    async def close(self) -> None:
        """Close the channel and its provider."""
        if self._provider is not None:
            await self._provider.close()

    @staticmethod
    def extract_text(event: RoomEvent) -> str:
        """Extract plain text from an event's content."""
        if isinstance(event.content, TextContent):
            return event.content.body
        return ""
