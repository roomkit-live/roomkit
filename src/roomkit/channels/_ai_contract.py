"""What AIChannel's mixins need from each other, for the type checker only.

Every AIChannel mixin derives from :class:`_AIChannelContract` under
``TYPE_CHECKING`` and from ``object`` at runtime. For ``ty`` the contract is
then the last base in AIChannel's MRO, so a mixin's implementation of a member
overrides the declaration here and is checked against it: a signature that
drifts fails ``make all``. At runtime the class is no base, so nothing here
can stand in for an implementation; ``tests/test_ai_channel_contract.py``
checks that every member has one, of the declared kind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from roomkit.channels._ai_resilience import _StreamRetryBoundary
    from roomkit.models.channel import ChannelBinding
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.models.tool_call import GenerationPurpose
    from roomkit.providers.ai.base import (
        AIContext,
        AIImagePart,
        AIMessage,
        AITextPart,
        AITool,
        AIToolResultPart,
        StreamEvent,
    )
    from roomkit.realtime.base import EphemeralEventType
    from roomkit.speaking.base import SpeakDecision
    from roomkit.speaking.thought import Thought
    from roomkit.telemetry.base import TelemetryProvider
    from roomkit.tools.context import _ToolLoopContext
    from roomkit.tools.policy import ToolPolicy


class _AIChannelContract:
    """The members one AIChannel mixin calls on another, each implemented once."""

    def _orchestration_tools(self, room_id: str | None) -> list[AITool]: ...

    def _orchestration_tool_names(self, room_id: str | None) -> set[str]: ...

    def _skill_tools(self) -> list[AITool]: ...

    def _reachable_tools(self, tools: Iterable[AITool]) -> list[AITool]: ...

    def _hook_toolset(self, loop_ctx: _ToolLoopContext) -> list[str]: ...

    def _declared_once(self, tools: list[AITool], room_id: str | None) -> list[AITool]: ...

    def _policy_allows(self, name: str) -> bool: ...

    def _get_loop_ctx(self) -> _ToolLoopContext: ...

    async def _thinking_context(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> AIContext | None: ...

    async def _speak_decision(
        self,
        event: RoomEvent,
        context: RoomContext,
        thought: Thought | None = None,
        *,
        asked_again: bool = False,
    ) -> SpeakDecision | None: ...

    def _served_tool_names(self, room_id: str | None) -> set[str]: ...

    def _apply_tool_filters(self, tools: list[AITool]) -> list[AITool]: ...

    def _held_declaration(
        self, loop_ctx: _ToolLoopContext, shown: list[AITool]
    ) -> list[AITool]: ...

    def _open_turn_declaration(
        self, context: AIContext, loop_ctx: _ToolLoopContext, shown: list[AITool]
    ) -> None: ...

    async def _publish_tool_event(
        self,
        event_type: EphemeralEventType,
        room_id: str | None,
        tool_calls: list[Any],
        round_idx: int,
        *,
        duration_ms: int | None = None,
    ) -> None: ...

    def _bound_provider_result(self, name: str, result: str, tool_call_id: str) -> str: ...

    async def _execute_tools_parallel(
        self,
        tool_calls: list[Any],
        telemetry: TelemetryProvider,
        *,
        declared_tools: list[AITool] | None = None,
        parent_span_id: str | None = None,
    ) -> list[AIToolResultPart]: ...

    def _never_hidden(self, room_id: str | None) -> set[str]: ...

    async def _report_unreported_calls(self, loop_ctx: _ToolLoopContext) -> None: ...

    def _show_summarized_references(self, summarized: list[AIMessage]) -> None: ...

    async def _build_context(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> AIContext: ...

    async def _fire_before_generation_hook(
        self, ai_context: AIContext, event: RoomEvent, *, purpose: GenerationPurpose = "answer"
    ) -> tuple[AIContext, bool]: ...

    def _drain_steering_queue(
        self, context: AIContext, loop_ctx: _ToolLoopContext
    ) -> tuple[AIContext, bool]: ...

    def _generate_stream_with_retry(
        self, context: AIContext
    ) -> AsyncIterator[StreamEvent | _StreamRetryBoundary]: ...

    def _record_declared_tools(
        self, loop_ctx: _ToolLoopContext, tools: list[AITool] | None
    ) -> None: ...

    async def _publish_thinking_event(
        self, event_type: EphemeralEventType, room_id: str | None, thinking: str, round_idx: int
    ) -> None: ...

    @property
    def _telemetry_provider(self) -> TelemetryProvider: ...

    @property
    def _effective_tool_policy(self) -> ToolPolicy | None: ...

    @property
    def _gated_tool_names(self) -> set[str]: ...

    def _activated_skill_names(self) -> set[str]: ...

    @property
    def _exempt_tool_names(self) -> set[str]: ...

    def _maybe_truncate_result(
        self, result: str | list[AITextPart | AIImagePart], tool_call_id: str = ""
    ) -> str | list[AITextPart | AIImagePart]: ...

    def _never_deferred(self, loop_ctx: _ToolLoopContext) -> set[str]: ...

    def _gate_refusal(self, name: str) -> dict[str, str] | None: ...

    def _reference_shown(self, loop_ctx: _ToolLoopContext) -> list[str]: ...
