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
from roomkit.models.context import RoomContext
from roomkit.models.enums import EventType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.orchestration.state import (
    get_conversation_state,
    set_conversation_state,
)
from roomkit.providers.ai.base import AIMessage, AITool

if TYPE_CHECKING:
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
        "Use when: the conversation needs expertise you don't have, "
        "the user requests escalation, or your task is complete "
        "and the next step requires a different agent. "
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
    ) -> None:
        self._kit = kit
        self._router = router
        self._aliases = agent_aliases or {}
        self._phase_map = phase_map or {}

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

        # Validate target agent is attached to room
        room = await self._kit.get_room(room_id)
        bindings = await self._kit.store.list_bindings(room_id)
        target_binding = None
        for b in bindings:
            if b.channel_id == request.target_agent_id:
                target_binding = b
                break

        if target_binding is None and request.target_agent_id != "human":
            logger.warning(
                "Handoff rejected: agent %s not in room %s",
                request.target_agent_id,
                room_id,
            )
            return HandoffResult(
                accepted=False,
                reason=f"Agent {request.target_agent_id} not found in room",
            )

        # Determine new phase
        new_phase = (
            arguments.get("next_phase")
            or self._phase_map.get(request.target_agent_id)
            or "handling"
        )

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

        # Emit system event in room timeline
        await self._kit.send_event(
            room_id=room_id,
            channel_id=calling_agent_id,
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
            },
        )

        logger.info(
            "Handoff: %s -> %s (phase=%s, room=%s)",
            calling_agent_id,
            request.target_agent_id,
            new_phase,
            room_id,
        )

        return HandoffResult(
            accepted=True,
            new_agent_id=new_agent,
            new_phase=new_phase,
            message=f"Conversation transferred to {request.target_agent_id}",
        )

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
) -> None:
    """Wire handoff into an AIChannel's tool chain.

    - Injects HANDOFF_TOOL into the channel's tool definitions
    - Wraps the tool handler to intercept ``handoff_conversation`` calls
    - Uses ``_room_id_var`` ContextVar for room_id (set by routing hook)
    """
    # Inject the handoff tool definition
    channel._extra_tools.append(HANDOFF_TOOL)

    # Wrap the tool handler chain
    original = channel._tool_handler

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

    channel._tool_handler = handoff_aware_handler
