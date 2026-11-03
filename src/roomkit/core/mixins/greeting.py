"""GreetingMixin — greeting delivery and gate management."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core._reentrance import RoomWork, chain_floor
from roomkit.core.exceptions import RoomClosedError
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.core.visibility import with_committed_visibility
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventDirection,
    EventStatus,
    EventType,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.models.channel import AgentIdentity
    from roomkit.store.base import ConversationStore
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.framework")


@runtime_checkable
class GreetingHost(Protocol):
    """Contract: capabilities a host class must provide for GreetingMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation store for bindings and event persistence.
        _channels: Channel registry for resolving Agent channels.
        _greeting_gates: Per-room gates that block inbound processing
            until greeting delivery completes.
        _greeting_gate_counts: Reference counts for multi-agent gates.

    Attributes also provided by the host's ``__init__``:
        _room_work: Tells the greeting what it is to the room's work.

    Cross-mixin methods (provided by other mixins in the MRO):
        _commit_and_deliver: From :class:`LaneExecutionMixin`.
        _store_entry_past_cap: From :class:`RoomKit`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _greeting_gates: dict[str, asyncio.Event]
    _greeting_gate_counts: dict[str, int]
    _room_work: RoomWork


class GreetingMixin(HelpersMixin):
    """Greeting delivery and gate management for the framework.

    Host contract: :class:`GreetingHost`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _greeting_gates: dict[str, asyncio.Event]
    _greeting_gate_counts: dict[str, int]

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    _commit_and_deliver: Any  # LaneExecutionMixin
    get_room: Any  # RoomLifecycleMixin
    _room_work: RoomWork  # RoomKit
    _store_entry_past_cap: Any  # RoomKit

    async def send_greeting(
        self,
        room_id: str,
        *,
        channel_id: str | None = None,
        agent_id: str | None = None,
        greeting: str | None = None,
        session: VoiceSession | None = None,
        channel_type: ChannelType | None = None,
        organization_id: str | None = None,
    ) -> None:
        """Deliver a greeting and store it as assistant history.

        Dispatches by session context rather than scanning room bindings:

        * **Realtime voice session** — injected through the channel as the
          model's own turn (:meth:`_greet_realtime`).
        * **Voice session** — speaks via ``say()`` (runs BEFORE_TTS /
          AFTER_TTS hooks).
        * **Everything else** (text channels, no session) — stores and
          broadcasts to transport channels.

        Args:
            room_id: The room to greet.
            channel_id: Optional channel ID to use for finding the agent.
            agent_id: Optional agent channel ID to use (alternative to
                ``channel_id``).
            greeting: Explicit greeting text.  Falls back to the agent's
                (``Agent.greeting``, or its declaration's) if not provided.
            session: The voice session to deliver into (passed by the
                auto-greet handler from the ``SessionStartedEvent``).
            channel_type: The channel type that started the session.
            organization_id: The organization the caller acts for (RFC
                §17.2): the room is read with it before an agent is looked
                up. Unset, no room is read for the scope.

        Raises:
            RoomNotFoundError: *organization_id* is set and the room is
                missing or another organization's; nothing is said or stored.
        """
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        found = await self._greeting_agent(room_id, agent_id or channel_id)
        if found is None:
            return  # No agent found — nothing to greet with
        agent_channel_id, identity = found

        # Resolve greeting text
        text = greeting or identity.greeting
        if not text:
            return  # No greeting configured

        # Dispatch by session type
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel
        from roomkit.channels.voice import VoiceChannel

        voice_ch = self._channels.get(session.channel_id) if session is not None else None
        if session is not None and isinstance(voice_ch, RealtimeVoiceChannel):
            # Every realtime channel, an audio-video one included (its session
            # start names its own type).
            await self._greet_realtime(voice_ch, session, room_id, agent_channel_id, text)
            return

        if session is not None and channel_type == ChannelType.VOICE:
            # Store first so AI sees greeting in history even if TTS fails
            await self._store_greeting_event(room_id, agent_channel_id, text)
            voice_ch = self._channels.get(session.channel_id)
            if isinstance(voice_ch, VoiceChannel) and voice_ch._tts and voice_ch._backend:
                voice = voice_ch._resolve_voice(agent_channel_id)
                await voice_ch.say(session, text, voice=voice)
            return

        # Text path: commit the greeting and deliver it through the room's
        # lane, which is what keeps its index and its delivery in the same
        # order as every other event's (RFC §10.2). Made in the room's own
        # work, a hook's body or a delivery, it is that work's write as
        # send_event's is: at its depth, on its budget (RFC §8.3, §9.5.1), so
        # that a hook greeting on every answer stops at the limit.
        entry = self._room_work.entry(room_id)
        event = self._make_greeting_event(
            room_id, agent_channel_id, text, chain_depth=chain_floor()
        )
        if entry.charged and not entry.draw():
            with contextlib.suppress(RoomClosedError):
                await self._store_entry_past_cap(room_id, event)
            return
        await self._commit_and_deliver(room_id, event, agent_channel_id, budget=entry.budget)

    async def _greeting_agent(
        self, room_id: str, channel_id: str | None
    ) -> tuple[str, AgentIdentity] | None:
        """The agent that greets *room_id* and its identity: *channel_id*'s
        when it is an agent, else the first agent bound to the room. A
        declared agent is one as a live one is (RFC §10.6.2)."""
        if channel_id:
            identity = self._agent_identity(channel_id)
            if identity is not None:
                return channel_id, identity
        for binding in await self._store.list_bindings(room_id):
            identity = self._agent_identity(binding.channel_id)
            if identity is not None:
                return binding.channel_id, identity
        return None

    def _agent_identity(self, channel_id: str) -> AgentIdentity | None:
        channel = self._channels.get(channel_id)
        if channel is None or channel.category is not ChannelCategory.INTELLIGENCE:
            return None
        return channel.declaration().identity

    async def _greet_realtime(
        self, channel: Any, session: VoiceSession, room_id: str, agent_id: str, text: str
    ) -> None:
        """Greet in a realtime session as the model's own turn: stored first,
        so the model sees it in its history even if the injection fails, then
        injected through the channel, which announces it to
        ON_REALTIME_TEXT_INJECTED as every injection (RFC §12.4)."""
        await self._store_greeting_event(room_id, agent_id, text)
        await channel.inject_text(session, text, role="assistant")

    @staticmethod
    def _make_greeting_event(
        room_id: str, agent_channel_id: str, text: str, *, chain_depth: int = 0
    ) -> RoomEvent:
        """Build the greeting as an assistant event."""
        return RoomEvent(
            room_id=room_id,
            type=EventType.MESSAGE,
            source=EventSource(
                channel_id=agent_channel_id,
                channel_type=ChannelType.AI,
                direction=EventDirection.OUTBOUND,
            ),
            content=TextContent(body=text),
            status=EventStatus.DELIVERED,
            chain_depth=chain_depth,
            metadata={"auto_greeting": True},
        )

    async def _store_greeting_event(
        self, room_id: str, agent_channel_id: str, text: str
    ) -> RoomEvent | None:
        """Store a greeting the voice paths speak themselves.

        Storage only: the greeting reaches the caller as TTS audio or as an
        injected realtime turn, never as a broadcast, so it carries no
        delivery set and its index is a plain cursor entry. The text path
        goes through :meth:`_commit_and_deliver` instead.
        """
        # Commit atomically (index + room counters, §14.3): the greeting is a
        # DELIVERED timeline event and must be reflected in the counters. Its
        # visibility is narrowed by the agent's binding (RFC §7.3), as any
        # event its agent writes.
        bindings = await self._store.list_bindings(room_id)
        agent_binding = next((b for b in bindings if b.channel_id == agent_channel_id), None)
        event = with_committed_visibility(
            self._make_greeting_event(room_id, agent_channel_id, text), agent_binding, bindings
        )
        # An agent's words to the room: its activity (RFC §5.1).
        return await self._persist_committed(room_id, event, activity=True)

    def _set_greeting_gate(self, room_id: str) -> None:
        """Increment the reference-counted gate for *room_id*.

        Creates the gate on the first call; subsequent calls for the same
        room (e.g. multi-agent) increment the refcount.
        """
        if room_id not in self._greeting_gates:
            self._greeting_gates[room_id] = asyncio.Event()
            self._greeting_gate_counts[room_id] = 1
        else:
            self._greeting_gate_counts[room_id] = self._greeting_gate_counts.get(room_id, 0) + 1

    def _clear_greeting_gate(self, room_id: str) -> None:
        """Decrement the gate refcount; release waiters when it reaches zero."""
        count = self._greeting_gate_counts.get(room_id, 0)
        if count <= 1:
            # Last agent done — release the gate
            gate = self._greeting_gates.pop(room_id, None)
            self._greeting_gate_counts.pop(room_id, None)
            if gate is not None:
                gate.set()
        else:
            self._greeting_gate_counts[room_id] = count - 1

    def _force_clear_greeting_gate(self, room_id: str) -> None:
        """Unconditionally release the gate (used on timeout and close)."""
        gate = self._greeting_gates.pop(room_id, None)
        self._greeting_gate_counts.pop(room_id, None)
        if gate is not None:
            gate.set()

    async def _wait_greeting_gate(self, room_id: str, timeout: float = 30.0) -> None:
        """Wait until the greeting gate for *room_id* is cleared.

        If no gate exists the call returns immediately.  On timeout the
        gate is forcibly cleared and a warning is logged.
        """
        gate = self._greeting_gates.get(room_id)
        if gate is None:
            return
        try:
            await asyncio.wait_for(gate.wait(), timeout=timeout)
        except TimeoutError:
            logger.warning(
                "Greeting gate timed out after %.1fs for room %s — force-clearing",
                timeout,
                room_id,
            )
            self._force_clear_greeting_gate(room_id)
