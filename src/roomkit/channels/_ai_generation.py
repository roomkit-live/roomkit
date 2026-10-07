"""AIChannel mixin for what every turn's generation shares: the
BEFORE_AI_GENERATION hook and the telemetry provider."""

from __future__ import annotations

import logging
from collections.abc import Container
from typing import TYPE_CHECKING

from roomkit.channels._served_tools import CollisionLog
from roomkit.channels._turn_notes import turn_input
from roomkit.models.event import RoomEvent
from roomkit.models.tool_call import AIGenerationEvent, GenerationPurpose
from roomkit.providers.ai.base import AIContext
from roomkit.telemetry.base import TelemetryProvider
from roomkit.telemetry.noop import NoopTelemetryProvider
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext

if TYPE_CHECKING:
    from roomkit.channels._ai_callbacks import BeforeGenerationHook
    from roomkit.providers.ai.base import AITool

if TYPE_CHECKING:
    from roomkit.channels._ai_contract import _AIChannelContract
else:
    _AIChannelContract = object

logger = logging.getLogger("roomkit.channels.ai")


class AIGenerationMixin(_AIChannelContract):
    """The generation hook and telemetry every turn uses.

    What it calls on the other mixins is declared once, in
    :class:`~roomkit.channels._ai_contract._AIChannelContract`, which it
    derives from for the type checker only.
    """

    _before_generation_hook: BeforeGenerationHook | None
    _collisions: CollisionLog
    channel_id: str
    provider_name: str

    @property
    def _telemetry_provider(self) -> TelemetryProvider:
        """Access telemetry provider (set by register_channel)."""
        return getattr(self, "_telemetry", None) or NoopTelemetryProvider()

    async def _fire_before_generation_hook(
        self, ai_context: AIContext, event: RoomEvent, *, purpose: GenerationPurpose = "answer"
    ) -> tuple[AIContext, bool]:
        """Fire BEFORE_AI_GENERATION hook. Returns ``(context, blocked)``.

        *purpose* is ``answer`` for the agent's turn, ``thought`` for its thinker
        (RFC §6.4): every model call on the room's context passes the hook.
        """
        if not self._before_generation_hook:
            return ai_context, False
        gen_event = AIGenerationEvent(
            ai_context=ai_context,
            channel_id=self.channel_id,
            room_id=event.room_id,
            trigger=event,
            provider_name=self.provider_name,
            purpose=purpose,
        )
        # Read before the hook runs: it may edit the list in place.
        declared = {tool.name for tool in ai_context.tools or []}
        sync_result = await self._before_generation_hook(gen_event)
        if not sync_result.allowed:
            logger.info(
                "AI generation blocked by hook (reason=%s, blocked_by=%s)",
                sync_result.reason,
                sync_result.blocked_by,
            )
            return ai_context, True
        # A hook that REPLACES the context (rather than mutating it) brings a
        # record of its own. The turn has one record — the one the events are
        # built from — so the loop context adopts the hook's: a tool handler's
        # writes then land where the reply reads, whichever object the hook
        # returned.
        loop_ctx = _current_loop_ctx.get()
        if loop_ctx is not None:
            loop_ctx.response_metadata = gen_event.ai_context.response_metadata
            # The input the hook left is the one a compaction keeps whole.
            loop_ctx.turn_input = turn_input(gen_event.ai_context.messages)
            _adopt_hook_toolset(
                loop_ctx,
                declared,
                gen_event.ai_context.tools,
                served=self._served_tool_names(loop_ctx.room_id),
                collisions=self._collisions,
            )
        return gen_event.ai_context, False


def _adopt_hook_toolset(
    loop_ctx: _ToolLoopContext,
    declared: set[str],
    left: list[AITool] | None,
    *,
    served: Container[str],
    collisions: CollisionLog,
) -> None:
    """Make what BEFORE_AI_GENERATION left of the toolset it saw the turn's base.

    Every round re-filters from ``all_context_tools``: without this, a tool
    the hook withdrew would come back (and run), and one it added would
    vanish. The hook saw the toolset the policy and skill gating leave, Tool
    Search's catalogue included, so a tool it removes is gone from every
    round, reveal and call; one it never saw (gated by a skill, denied by the
    policy) stays in the base for those filters to decide. A tool it adds is
    pinned for the turn: Tool Search never defers it (RFC §6.4). A name the
    channel or orchestration serves keeps its definition: the hook may
    withdraw it, never redefine it, nor add a tool under it (RFC §21.1).
    """
    if loop_ctx.all_context_tools is None:
        return
    kept = {tool.name: tool for tool in left or []}
    withdrawn = declared - kept.keys()
    original = {tool.name: tool for tool in loop_ctx.all_context_tools}
    for name, tool in kept.items():
        # A tool added under a served name, or a served tool redefined.
        if name in served and original.get(name) != tool:
            collisions.served(name)
    base = [t if t.name in served else kept.get(t.name, t) for t in loop_ctx.all_context_tools]
    known = {tool.name for tool in base}
    added = {name for name in kept if name not in known and name not in served}
    base.extend(kept[name] for name in kept if name in added)
    loop_ctx.all_context_tools = base
    loop_ctx.withdraw(withdrawn)
    loop_ctx.hook_pinned = loop_ctx.hook_pinned | added
