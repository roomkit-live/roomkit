"""Handoff protocol for multi-agent orchestration.

Agents trigger handoffs via the ``handoff_conversation`` tool call.
The framework intercepts the call, updates conversation state,
preserves context, and optionally escalates channels.
"""

from __future__ import annotations

import contextvars
import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from roomkit.memory.base import MemoryProvider, MemoryResult
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType, EventStatus, EventType, HookTrigger
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.orchestration.state import (
    get_conversation_state,
    set_conversation_state,
)
from roomkit.providers.ai.base import AIMessage, AITool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from roomkit.channels.ai import AIChannel
    from roomkit.core.framework import RoomKit
    from roomkit.orchestration.router import ConversationRouter

logger = logging.getLogger("roomkit.orchestration.handoff")

# ContextVar set by the routing hook, read by the handoff tool handler.
# Safe across concurrent asyncio tasks because each task inherits a copy
# of the parent context at creation time.
_room_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_orchestration_room_id", default=None
)


# -- Models ------------------------------------------------------------------


class HandoffRequest(BaseModel):
    """Parsed from agent's handoff tool call arguments."""

    target_agent_id: str
    reason: str = ""
    summary: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    channel_escalation: str | None = None
    urgent: bool = False


class HandoffResult(BaseModel):
    """Result returned to the calling agent."""

    accepted: bool = True
    new_agent_id: str | None = None
    new_phase: str | None = None
    message: str = ""
    reason: str = ""


# -- Tool definition ----------------------------------------------------------


HANDOFF_TOOL = AITool(
    name="handoff_conversation",
    description=(
        "Transfer this conversation to another agent or specialist. "
        "Use when: the user asks to speak with someone else, "
        "the user wants to be transferred or go back to a previous agent, "
        "the conversation needs expertise you don't have, "
        "or your task is complete and the next step requires a different agent. "
        "Always provide a clear summary of the conversation so far."
    ),
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "Target agent ID or alias (e.g., 'agent-advisor', 'agent-coder', 'human')"
                ),
            },
            "reason": {
                "type": "string",
                "description": "Why the handoff is needed",
            },
            "summary": {
                "type": "string",
                "description": (
                    "Summary of the conversation so far and what the next "
                    "agent needs to know to continue effectively"
                ),
            },
            "next_phase": {
                "type": "string",
                "description": (
                    "Optional: conversation phase to transition to "
                    "(e.g., 'handling', 'review', 'escalation')"
                ),
            },
            "channel_escalation": {
                "type": "string",
                "enum": ["same", "voice", "email", "sms"],
                "description": (
                    "Whether to escalate to a different channel. 'same' keeps the current channel."
                ),
            },
        },
        "required": ["target", "reason", "summary"],
    },
)


def build_handoff_tool(targets: list[tuple[str, str | None]]) -> AITool:
    """Build a handoff tool with constrained target enum.

    Args:
        targets: List of ``(agent_id, description_or_none)`` pairs.
            When empty, returns the generic :data:`HANDOFF_TOOL`.

    Returns:
        An :class:`AITool` whose ``target`` parameter has an ``enum``
        restricting the AI to only the listed agent IDs.
    """
    if not targets:
        return HANDOFF_TOOL

    target_ids = [t[0] for t in targets]
    target_lines = []
    for agent_id, desc in targets:
        if desc:
            target_lines.append(f"  - {agent_id}: {desc}")
        else:
            target_lines.append(f"  - {agent_id}")

    description = "Transfer this conversation to another agent.\nAvailable targets:\n" + "\n".join(
        target_lines
    )

    # Clone the base parameters and constrain the target field
    params = {
        "type": "object",
        "properties": {
            **HANDOFF_TOOL.parameters["properties"],
            "target": {
                "type": "string",
                "enum": target_ids,
                "description": "Target agent ID to transfer to",
            },
        },
        "required": HANDOFF_TOOL.parameters["required"],
    }

    return AITool(
        name="handoff_conversation",
        description=description,
        parameters=params,
    )


# -- HandoffHandler -----------------------------------------------------------


class HandoffHandler:
    """Processes handoff tool calls.

    Intercepts the ``handoff_conversation`` tool, updates conversation
    state, emits a system event, and returns the result to the agent.
    """

    def __init__(
        self,
        kit: RoomKit,
        router: ConversationRouter,
        agent_aliases: dict[str, str] | None = None,
        phase_map: dict[str, str] | None = None,
        allowed_transitions: dict[str, set[str]] | None = None,
        known_agents: set[str] | None = None,
        on_handoff_complete: (Callable[[str, HandoffResult], Awaitable[None]] | None) = None,
        event_channel_id: str | None = None,
    ) -> None:
        self._kit = kit
        self._router = router
        self._aliases = agent_aliases or {}
        self._phase_map = phase_map or {}
        self._allowed_transitions = allowed_transitions
        self._known_agents = known_agents or set()
        self._on_handoff_complete = on_handoff_complete
        self._event_channel_id = event_channel_id
        # Populated by pipeline.install() from Agent fields
        self._greeting_map: dict[str, str] = {}
        self._agents: dict[str, Any] = {}  # agent_id -> Agent
        self._agent_configs: dict[str, dict[str, Any]] = {}  # for realtime reconfigure

    @property
    def greeting_map(self) -> dict[str, str]:
        """Agent ID to greeting text mapping."""
        return self._greeting_map

    @greeting_map.setter
    def greeting_map(self, value: dict[str, str]) -> None:
        self._greeting_map = value

    @property
    def agents(self) -> dict[str, Any]:
        """Agent ID to Agent instance mapping."""
        return self._agents

    @agents.setter
    def agents(self, value: dict[str, Any]) -> None:
        self._agents = value

    @property
    def on_handoff_complete(
        self,
    ) -> Callable[[str, HandoffResult], Awaitable[None]] | None:
        """Callback invoked after a successful handoff."""
        return self._on_handoff_complete

    @on_handoff_complete.setter
    def on_handoff_complete(
        self,
        value: Callable[[str, HandoffResult], Awaitable[None]] | None,
    ) -> None:
        self._on_handoff_complete = value

    def get_room_language(self, room: Room, agent_id: str | None) -> str | None:
        """Get effective language: room override > agent default."""
        return self._get_room_language(room, agent_id)

    async def handle(
        self,
        room_id: str,
        calling_agent_id: str,
        arguments: dict[str, Any],
    ) -> HandoffResult:
        """Process a handoff request.

        Steps:
        1. Resolve target agent (alias -> channel_id)
        2. Validate target exists in room
        3. Update ConversationState
        4. Persist state to room metadata
        5. Emit system event in room
        6. Return result to calling agent
        """
        request = HandoffRequest(
            target_agent_id=self._resolve_alias(arguments.get("target", "")),
            reason=arguments.get("reason", ""),
            summary=arguments.get("summary", ""),
            channel_escalation=arguments.get("channel_escalation", "same"),
            context=arguments,
        )

        # Validate and update state under room lock to prevent lost updates.
        # Hooks and send_event() run AFTER the lock is released to avoid
        # blocking concurrent operations during slow I/O.
        rejected_result: HandoffResult | None = None
        new_phase: str = "handling"
        new_agent: str | None = None
        async with self._kit.lock_manager.locked(room_id):
            room = await self._kit.get_room(room_id)
            bindings = await self._kit.store.list_bindings(room_id)
            target_binding = None
            for b in bindings:
                if b.channel_id == request.target_agent_id:
                    target_binding = b
                    break

            if (
                target_binding is None
                and request.target_agent_id != "human"
                and request.target_agent_id not in self._known_agents
            ):
                logger.warning(
                    "Handoff rejected: agent %s not in room %s",
                    request.target_agent_id,
                    room_id,
                )
                rejected_result = HandoffResult(
                    accepted=False,
                    reason=f"Agent {request.target_agent_id} not found in room",
                )

            if rejected_result is None:
                # Determine new phase
                new_phase = (
                    arguments.get("next_phase")
                    or self._phase_map.get(request.target_agent_id)
                    or "handling"
                )

                # Validate transition if constraints are configured
                if self._allowed_transitions is not None:
                    state = get_conversation_state(room)
                    allowed = self._allowed_transitions.get(state.phase, set())
                    if new_phase not in allowed:
                        logger.warning(
                            "Handoff rejected: transition %s -> %s not allowed",
                            state.phase,
                            new_phase,
                        )
                        rejected_result = HandoffResult(
                            accepted=False,
                            reason=(
                                f"Transition from '{state.phase}' to '{new_phase}' is not allowed"
                            ),
                        )

            if rejected_result is None:
                # Handle "human" target — set active_agent to None
                new_agent = None if request.target_agent_id == "human" else request.target_agent_id

                # Update conversation state
                state = get_conversation_state(room)
                new_state = state.transition(
                    to_phase=new_phase,
                    to_agent=new_agent,
                    reason=request.reason,
                    metadata={
                        "summary": request.summary,
                        "channel_escalation": request.channel_escalation,
                    },
                )
                # Preserve handoff summary in state context for memory injection
                new_state = new_state.model_copy(
                    update={
                        "context": {
                            **new_state.context,
                            "handoff_summary": request.summary,
                            "handoff_from": calling_agent_id,
                            "handoff_reason": request.reason,
                        }
                    }
                )

                # Persist
                updated_room = set_conversation_state(room, new_state)
                await self._kit.store.update_room(updated_room)

        # Fire rejection hook outside the lock
        if rejected_result is not None:
            await self._fire_hook(
                room_id,
                room,
                bindings,
                HookTrigger.ON_HANDOFF_REJECTED,
                calling_agent_id,
                request,
                rejected_result,
            )
            return rejected_result

        # Emit system event in room timeline (outside lock — send_event acquires its own)
        # In speech-to-speech mode, config-only agents aren't bound to the room,
        # so use the event_channel_id (the RealtimeVoiceChannel) as source.
        await self._kit.send_event(
            room_id=room_id,
            channel_id=self._event_channel_id or calling_agent_id,
            content=TextContent(
                body=(
                    f"[Handoff: {calling_agent_id} -> {request.target_agent_id}] {request.reason}"
                )
            ),
            event_type=EventType.SYSTEM,
            visibility="all",
            metadata={
                "handoff": True,
                "from_agent": calling_agent_id,
                "to_agent": request.target_agent_id,
                "summary": request.summary,
                "_orchestration_internal": True,
            },
        )

        logger.info(
            "Handoff: %s -> %s (phase=%s, room=%s)",
            calling_agent_id,
            request.target_agent_id,
            new_phase,
            room_id,
        )

        result = HandoffResult(
            accepted=True,
            new_agent_id=new_agent,
            new_phase=new_phase,
            message=(
                f"Conversation transferred to {request.target_agent_id}. "
                "Do not respond — the receiving agent will greet the user."
            ),
        )

        # Fire orchestration hooks (async, fire-and-forget)
        updated_room = await self._kit.get_room(room_id)
        updated_bindings = await self._kit.store.list_bindings(room_id)
        await self._fire_hook(
            room_id,
            updated_room,
            updated_bindings,
            HookTrigger.ON_HANDOFF,
            calling_agent_id,
            request,
            result,
        )
        await self._fire_hook(
            room_id,
            updated_room,
            updated_bindings,
            HookTrigger.ON_PHASE_TRANSITION,
            calling_agent_id,
            request,
            result,
        )

        if self._on_handoff_complete:
            await self._on_handoff_complete(room_id, result)

        return result

    async def _fire_hook(
        self,
        room_id: str,
        room: Room,
        bindings: list[ChannelBinding],
        trigger: HookTrigger,
        calling_agent_id: str,
        request: HandoffRequest,
        result: HandoffResult,
    ) -> None:
        """Fire an orchestration hook (async, best-effort)."""
        event = RoomEvent(
            room_id=room_id,
            source=EventSource(
                channel_id=calling_agent_id,
                channel_type=ChannelType.AI,
            ),
            content=TextContent(body=request.reason),
            type=EventType.SYSTEM,
            status=EventStatus.DELIVERED,
            visibility="internal",
            metadata={
                "handoff": True,
                "from_agent": calling_agent_id,
                "to_agent": request.target_agent_id,
                "accepted": result.accepted,
                "new_phase": result.new_phase,
            },
        )
        context = RoomContext(room=room, bindings=bindings)
        await self._kit.hook_engine.run_async_hooks(room_id, trigger, event, context)

    async def send_greeting(
        self,
        room_id: str,
        *,
        channel_id: str | None = None,
    ) -> None:
        """Send the initial agent greeting for a room.

        Looks up the current active agent from conversation state and
        sends its ``greeting``.  For :class:`RealtimeVoiceChannel`,
        injects text directly into the provider session.  For traditional
        voice, sends a synthetic inbound message to trigger an AI response.

        Call this after setting the initial conversation state::

            room = set_conversation_state(room, ConversationState(...))
            await kit.store.update_room(room)
            await handler.send_greeting(room_id, channel_id="voice")

        Does nothing when the active agent has no greeting configured.
        """
        room = await self._kit.get_room(room_id)
        state = get_conversation_state(room)
        agent_id = state.active_agent_id
        if not agent_id:
            return

        greeting = self._greeting_map.get(agent_id)
        if not greeting:
            return

        # Prepend language instruction if set
        lang = self._get_room_language(room, agent_id)
        if lang:
            greeting = f"[Respond in {lang}] {greeting}"

        ch_id = channel_id or self._event_channel_id
        if not ch_id:
            return

        channel = self._kit.channels.get(ch_id)

        # RealtimeVoiceChannel: inject text directly into provider session
        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        if isinstance(channel, RealtimeVoiceChannel):
            for session in channel.get_room_sessions(room_id):
                await channel.provider.inject_text(session, greeting, role="user")
            return

        # Traditional voice: send synthetic inbound message
        from roomkit.models.delivery import InboundMessage

        await self._kit.process_inbound(
            InboundMessage(
                channel_id=ch_id,
                sender_id="system",
                content=TextContent(body=greeting),
            ),
            room_id=room_id,
        )

    async def set_language(
        self,
        room_id: str,
        language: str,
        *,
        channel_id: str | None = None,
    ) -> None:
        """Change the conversation language for a room.

        Stores the language in conversation state and, for realtime
        sessions, reconfigures the active agent's session with an
        updated system prompt that includes the language instruction.

        Args:
            room_id: The room to update.
            language: Language name or code (e.g. ``"French"``, ``"fr"``).
            channel_id: Voice channel ID. Falls back to the event channel
                configured by ``install()``.
        """
        # 1. Persist language in conversation state
        async with self._kit.lock_manager.locked(room_id):
            room = await self._kit.get_room(room_id)
            state = get_conversation_state(room)
            new_state = state.model_copy(
                update={"context": {**state.context, "language": language}}
            )
            room = set_conversation_state(room, new_state)
            await self._kit.store.update_room(room)

        agent_id = state.active_agent_id
        if not agent_id:
            return

        # 2. For realtime: reconfigure session with language in prompt
        ch_id = channel_id or self._event_channel_id
        if not ch_id:
            return

        channel = self._kit.channels.get(ch_id)

        from roomkit.channels.realtime_voice import RealtimeVoiceChannel

        if not isinstance(channel, RealtimeVoiceChannel):
            return

        agent = self._agents.get(agent_id)
        if agent is None:
            return

        # Rebuild system prompt with new language
        prompt = getattr(agent, "system_prompt", None) or ""
        identity = agent.build_identity_block(language=language)
        if identity:
            prompt = prompt + identity

        for session in channel.get_room_sessions(room_id):
            await channel.reconfigure_session(session, system_prompt=prompt)

        logger.info("Language set to %r for room %s (agent=%s)", language, room_id, agent_id)

    def _get_room_language(self, room: Room, agent_id: str | None) -> str | None:
        """Get effective language: room override > agent default."""
        state = get_conversation_state(room)
        lang: str | None = state.context.get("language") if state.context else None
        if lang:
            return lang
        if agent_id:
            agent = self._agents.get(agent_id)
            if agent is not None:
                return getattr(agent, "language", None)
        return None

    def _resolve_alias(self, target: str) -> str:
        return self._aliases.get(target, target)


# -- HandoffMemoryProvider ----------------------------------------------------


class HandoffMemoryProvider(MemoryProvider):
    """Injects handoff context (summary, reason) into agent prompts.

    Wraps an inner MemoryProvider and prepends handoff information
    when the conversation state indicates a recent handoff.
    """

    def __init__(self, inner: MemoryProvider) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        return f"Handoff({self._inner.name})"

    async def retrieve(
        self,
        room_id: str,
        current_event: RoomEvent,
        context: RoomContext,
        *,
        channel_id: str | None = None,
    ) -> MemoryResult:
        result = await self._inner.retrieve(room_id, current_event, context, channel_id=channel_id)

        state = get_conversation_state(context.room)
        summary = state.context.get("handoff_summary")
        handoff_from = state.context.get("handoff_from")

        if summary and handoff_from:
            handoff_msg = AIMessage(
                role="user",
                content=(f"[Context from previous agent ({handoff_from})]: {summary}"),
            )
            result.messages.insert(0, handoff_msg)

        return result

    async def ingest(
        self,
        room_id: str,
        event: RoomEvent,
        *,
        channel_id: str | None = None,
    ) -> None:
        await self._inner.ingest(room_id, event, channel_id=channel_id)

    async def clear(self, room_id: str) -> None:
        await self._inner.clear(room_id)

    async def close(self) -> None:
        await self._inner.close()


# -- Wiring -------------------------------------------------------------------


def setup_handoff(
    channel: AIChannel,
    handler: HandoffHandler,
    *,
    tool: AITool | None = None,
) -> None:
    """Wire handoff into an AIChannel's tool chain.

    - Injects the handoff tool into the channel's tool definitions
    - Wraps the tool handler to intercept ``handoff_conversation`` calls
    - Uses ``_room_id_var`` ContextVar for room_id (set by routing hook)

    Args:
        channel: The AI channel to wire handoff into.
        handler: The handoff handler that processes tool calls.
        tool: Optional custom handoff tool (e.g. from :func:`build_handoff_tool`).
            Defaults to the generic :data:`HANDOFF_TOOL`.
    """
    # Guard against double registration
    if any(t.name == "handoff_conversation" for t in channel.extra_tools):
        msg = f"setup_handoff() already called for channel '{channel.channel_id}'"
        raise RuntimeError(msg)

    # Inject the handoff tool definition
    channel.extra_tools.append(tool or HANDOFF_TOOL)

    # Wrap the tool handler chain
    original = channel.tool_handler

    async def handoff_aware_handler(name: str, arguments: dict[str, Any]) -> str:
        if name == "handoff_conversation":
            room_id = _room_id_var.get()
            if room_id is None:
                return json.dumps({"error": "No orchestration context (room_id unavailable)"})
            result = await handler.handle(
                room_id=room_id,
                calling_agent_id=channel.channel_id,
                arguments=arguments,
            )
            return result.model_dump_json()
        if original:
            return await original(name, arguments)
        return json.dumps({"error": f"Unknown tool: {name}"})

    channel.tool_handler = handoff_aware_handler
