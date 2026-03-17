"""GreetingMixin — greeting delivery and gate management."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.enums import (
    ChannelDirection,
    ChannelType,
    EventStatus,
    EventType,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.store.base import ConversationStore
    from roomkit.voice.base import VoiceSession

logger = logging.getLogger("roomkit.framework")


class GreetingMixin(HelpersMixin):
    """Greeting delivery and gate management for the framework."""

    _store: ConversationStore
    _channels: dict[str, Channel]
    _greeting_gates: dict[str, asyncio.Event]
    _greeting_gate_counts: dict[str, int]

    async def send_greeting(
        self,
        room_id: str,
        *,
        channel_id: str | None = None,
        agent_id: str | None = None,
        greeting: str | None = None,
        session: VoiceSession | None = None,
        channel_type: ChannelType | None = None,
    ) -> None:
        """Deliver a greeting and store it as assistant history.

        Dispatches by session context rather than scanning room bindings:

        * **Realtime voice session** — injects text via the provider.
        * **Voice session** — speaks via ``say()`` (runs BEFORE_TTS /
          AFTER_TTS hooks).
        * **Everything else** (text channels, no session) — stores and
          broadcasts to transport channels.

        Args:
            room_id: The room to greet.
            channel_id: Optional channel ID to use for finding the agent.
            agent_id: Optional agent channel ID to use (alternative to
                ``channel_id``).
            greeting: Explicit greeting text.  Falls back to
                ``Agent.greeting`` if not provided.
            session: The voice session to deliver into (passed by the
                auto-greet handler from the ``SessionStartedEvent``).
            channel_type: The channel type that started the session.
        """
        from roomkit.channels.agent import Agent as AgentChannel

        # Resolve the agent channel
        agent: AgentChannel | None = None
        resolve_id = agent_id or channel_id
        if resolve_id:
            ch = self._channels.get(resolve_id)
            if isinstance(ch, AgentChannel):
                agent = ch

        # Fallback: scan room bindings for an Agent channel
        if agent is None:
            bindings = await self._store.list_bindings(room_id)
            for b in bindings:
                ch = self._channels.get(b.channel_id)
                if isinstance(ch, AgentChannel):
                    agent = ch
                    break

        if agent is None:
            return  # No agent found — nothing to greet with

        # Resolve greeting text
        text = greeting or agent.greeting
        if not text:
            return  # No greeting configured

        # Dispatch by session type
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel
        from roomkit.channels.voice import VoiceChannel

        if session is not None and channel_type == ChannelType.REALTIME_VOICE:
            # Store first so AI sees greeting in history even if inject fails
            await self._store_greeting_event(room_id, agent.channel_id, text)
            voice_ch = self._channels.get(session.channel_id)
            if isinstance(voice_ch, RealtimeVoiceChannel):
                await voice_ch.provider.inject_text(session, text, role="assistant")
            return

        if session is not None and channel_type == ChannelType.VOICE:
            # Store first so AI sees greeting in history even if TTS fails
            await self._store_greeting_event(room_id, agent.channel_id, text)
            voice_ch = self._channels.get(session.channel_id)
            if isinstance(voice_ch, VoiceChannel) and voice_ch._tts and voice_ch._backend:
                voice = voice_ch._resolve_voice(agent.channel_id)
                await voice_ch.say(session, text, voice=voice)
            return

        # Text path: store greeting and broadcast to transport channels
        greeting_event = await self._store_greeting_event(room_id, agent.channel_id, text)
        source_binding = await self._store.get_binding(room_id, agent.channel_id)
        if source_binding is not None:
            router = self._get_router()  # type: ignore[attr-defined]
            context = await self._build_context(room_id)
            await router.broadcast(greeting_event, source_binding, context)

    async def _store_greeting_event(
        self, room_id: str, agent_channel_id: str, text: str
    ) -> RoomEvent:
        """Store a greeting as an assistant event in conversation history."""
        event = RoomEvent(
            room_id=room_id,
            type=EventType.MESSAGE,
            source=EventSource(
                channel_id=agent_channel_id,
                channel_type=ChannelType.AI,
                direction=ChannelDirection.OUTBOUND,
            ),
            content=TextContent(body=text),
            status=EventStatus.DELIVERED,
            metadata={"auto_greeting": True},
        )
        return await self._store.add_event_auto_index(room_id, event)

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
