"""AI channel implementation."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roomkit.channels.base import Channel
from roomkit.memory.base import MemoryProvider
from roomkit.memory.sliding_window import SlidingWindowMemory
from roomkit.models.channel import (
    ChannelBinding,
    ChannelCapabilities,
    ChannelOutput,
    RetryPolicy,
)
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
)
from roomkit.models.event import (
    CompositeContent,
    EventSource,
    MediaContent,
    RoomEvent,
    TextContent,
)
from roomkit.models.steering import Cancel, InjectMessage, SteeringDirective, UpdateSystemPrompt
from roomkit.providers.ai.base import (
    AIContext,
    AIImagePart,
    AIMessage,
    AIProvider,
    AIResponse,
    AITextPart,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
    ProviderError,
    StreamDone,
    StreamTextDelta,
    StreamToolCall,
)
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.telemetry.noop import NoopTelemetryProvider
from roomkit.tools.policy import ToolPolicy

if TYPE_CHECKING:
    from roomkit.skills.executor import ScriptExecutor
    from roomkit.skills.registry import SkillRegistry

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]

logger = logging.getLogger("roomkit.channels.ai")

# Skill tool names
_TOOL_ACTIVATE_SKILL = "activate_skill"
_TOOL_READ_REFERENCE = "read_skill_reference"
_TOOL_RUN_SCRIPT = "run_skill_script"

_SKILLS_PREAMBLE = (
    "You have access to Agent Skills — specialized knowledge packages. "
    "Use the activate_skill tool to load a skill's full instructions before "
    "using it. Available skills are listed below."
)

_SKILLS_NO_SCRIPTS_NOTE = " Note: Script execution is not available in this environment."


@dataclass
class _ToolLoopContext:
    """Per-invocation state for a tool loop, scoped via contextvar."""

    activated_skills: set[str] = field(default_factory=set)
    all_context_tools: list[Any] = field(default_factory=list)
    current_participant_role: str | None = None
    steering_queue: asyncio.Queue[SteeringDirective] = field(default_factory=asyncio.Queue)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    loop_id: str = ""


_current_loop_ctx: contextvars.ContextVar[_ToolLoopContext | None] = contextvars.ContextVar(
    "_current_loop_ctx", default=None
)


class AIChannel(Channel):
    """AI intelligence channel that generates responses using an AI provider."""

    channel_type = ChannelType.AI
    category = ChannelCategory.INTELLIGENCE
    direction = ChannelDirection.BIDIRECTIONAL

    def __init__(
        self,
        channel_id: str,
        provider: AIProvider,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_context_events: int = 50,
        tool_handler: ToolHandler | None = None,
        max_tool_rounds: int = 200,
        tool_loop_timeout_seconds: float | None = 300.0,
        tool_loop_warn_after: int = 50,
        retry_policy: RetryPolicy | None = None,
        fallback_provider: AIProvider | None = None,
        skills: SkillRegistry | None = None,
        script_executor: ScriptExecutor | None = None,
        memory: MemoryProvider | None = None,
        tool_policy: ToolPolicy | None = None,
    ) -> None:
        super().__init__(channel_id)
        self._provider = provider
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_context_events = max_context_events
        self._max_tool_rounds = max_tool_rounds
        self._tool_loop_timeout_seconds = tool_loop_timeout_seconds
        self._tool_loop_warn_after = tool_loop_warn_after
        self._retry_policy = retry_policy
        self._fallback_provider = fallback_provider
        self._skills = skills
        self._script_executor = script_executor
        self._memory = memory or SlidingWindowMemory(max_events=max_context_events)
        self._tool_policy = tool_policy

        # Wrap the user's tool handler with skill-aware dispatch
        self._user_tool_handler = tool_handler
        self._tool_handler: ToolHandler | None
        if skills and skills.skill_count > 0:
            self._tool_handler = self._skill_aware_tool_handler
        else:
            self._tool_handler = tool_handler

        # Extra tools injected by orchestration (e.g. HANDOFF_TOOL)
        self._extra_tools: list[AITool] = []

        # Active tool loops for steering (loop_id -> context)
        self._active_loops: dict[str, _ToolLoopContext] = {}

    @property
    def tool_handler(self) -> ToolHandler | None:
        """The current tool handler (may be wrapped by orchestration)."""
        return self._tool_handler

    @tool_handler.setter
    def tool_handler(self, value: ToolHandler | None) -> None:
        self._tool_handler = value

    @property
    def extra_tools(self) -> list[AITool]:
        """Extra tools injected by orchestration (e.g. handoff tool)."""
        return self._extra_tools

    def _propagate_telemetry(self) -> None:
        """Propagate telemetry to AI provider."""
        telemetry = getattr(self, "_telemetry", None)
        if telemetry is not None:
            self._provider._telemetry = telemetry

    # -- Per-invocation context ------------------------------------------------

    def _get_loop_ctx(self) -> _ToolLoopContext:
        """Get the current tool loop context (from contextvar or create default)."""
        ctx = _current_loop_ctx.get()
        if ctx is None:
            # Fallback for code paths outside a tool loop
            ctx = _ToolLoopContext()
        return ctx

    # -- Steering (mid-run interaction) ----------------------------------------

    def steer(self, directive: SteeringDirective, *, loop_id: str | None = None) -> None:
        """Enqueue a steering directive for the active tool loop.

        Safe to call from any coroutine. Cancel directives also set the
        fast-path cancel event so the loop can exit without waiting for
        the next drain point.

        Args:
            directive: The steering directive to enqueue.
            loop_id: Optional loop ID to target. If ``None``, targets the
                most recently started active loop.
        """
        if loop_id is not None:
            ctx = self._active_loops.get(loop_id)
        elif self._active_loops:
            ctx = next(reversed(self._active_loops.values()))
        else:
            ctx = None
        if ctx is None:
            logger.warning("steer() called with no active tool loop")
            return
        ctx.steering_queue.put_nowait(directive)
        if isinstance(directive, Cancel):
            ctx.cancel_event.set()

    def _drain_steering_queue(
        self, context: AIContext, loop_ctx: _ToolLoopContext
    ) -> tuple[AIContext, bool]:
        """Drain all pending steering directives, applying them to *context*.

        Returns:
            (updated_context, should_cancel)
        """
        should_cancel = False
        while not loop_ctx.steering_queue.empty():
            try:
                directive = loop_ctx.steering_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if isinstance(directive, Cancel):
                logger.info("Steering: cancel received — %s", directive.reason)
                should_cancel = True
            elif isinstance(directive, InjectMessage):
                logger.info("Steering: injecting %s message", directive.role)
                context.messages.append(AIMessage(role=directive.role, content=directive.content))
            elif isinstance(directive, UpdateSystemPrompt):
                logger.info("Steering: appending to system prompt")
                context = context.model_copy(
                    update={"system_prompt": (context.system_prompt or "") + directive.append}
                )

        return context, should_cancel

    async def close(self) -> None:
        """Close the channel, its provider, and the memory provider."""
        await super().close()
        await self._memory.close()

    @property
    def info(self) -> dict[str, Any]:
        return {"provider": type(self._provider).__name__}

    def capabilities(self) -> ChannelCapabilities:
        media_types = [ChannelMediaType.TEXT, ChannelMediaType.RICH]
        if self._provider.supports_vision:
            media_types.append(ChannelMediaType.MEDIA)
        return ChannelCapabilities(
            media_types=media_types,
            supports_rich_text=True,
            supports_media=self._provider.supports_vision,
        )

    async def handle_inbound(self, message: InboundMessage, context: RoomContext) -> RoomEvent:
        raise NotImplementedError("AI channel does not accept inbound messages")

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """React to an event by generating an AI response.

        Skips events from this channel to prevent self-loops.
        When the provider supports streaming or structured streaming:
        - With tools: uses the streaming tool loop that executes tool calls
          between generation rounds while yielding text deltas progressively.
        - Without tools: returns a plain streaming response.
        Otherwise falls back to the non-streaming generate path.
        """
        if event.source.channel_id == self.channel_id:
            return ChannelOutput.empty()

        # Resolve participant role for role-based tool policy.
        # Set on a per-invocation _ToolLoopContext visible via contextvar so that
        # _build_context and the tool loop methods can read it.
        event_ctx = _ToolLoopContext()
        event_ctx.current_participant_role = self._resolve_participant_role(event, context)
        token = _current_loop_ctx.set(event_ctx)
        try:
            raw_tools = binding.metadata.get("tools", [])
            has_tools = (
                bool(raw_tools)
                or bool(self._extra_tools)
                or (self._skills is not None and self._skills.skill_count > 0)
            )

            if self._provider.supports_streaming or self._provider.supports_structured_streaming:
                if has_tools:
                    return await self._start_streaming_tool_response(event, binding, context)
                return await self._start_streaming_response(event, binding, context)

            return await self._generate_response(event, binding, context)
        finally:
            _current_loop_ctx.reset(token)

    async def deliver(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Intelligence channels are not called via deliver by the router."""
        return ChannelOutput.empty()

    async def _start_streaming_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response handle (generator starts on consumption)."""
        ai_context = await self._build_context(event, binding, context)
        return ChannelOutput(
            responded=True,
            response_stream=self._provider.generate_stream(ai_context),
        )

    async def _start_streaming_tool_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Return a streaming response that handles tool calls between rounds."""
        ai_context = await self._build_context(event, binding, context)
        return ChannelOutput(
            responded=True,
            response_stream=self._run_streaming_tool_loop(ai_context),
        )

    async def _run_streaming_tool_loop(self, context: AIContext) -> AsyncIterator[str]:
        """Stream text deltas, executing tool calls between generation rounds."""
        loop_ctx = _ToolLoopContext()
        loop_ctx.loop_id = str(id(loop_ctx))
        # Inherit participant role from the on_event-level context
        parent_ctx = _current_loop_ctx.get()
        if parent_ctx is not None:
            loop_ctx.current_participant_role = parent_ctx.current_participant_role
        _current_loop_ctx.set(loop_ctx)
        self._active_loops[loop_ctx.loop_id] = loop_ctx
        try:
            # Apply pre-queued directives (e.g. cancel enqueued before loop started)
            context, should_cancel = self._drain_steering_queue(context, loop_ctx)
            if should_cancel:
                return
            telemetry = self._telemetry_provider
            deadline = (
                asyncio.get_event_loop().time() + self._tool_loop_timeout_seconds
                if self._tool_loop_timeout_seconds
                else None
            )

            for _round_idx in range(self._max_tool_rounds + 1):
                # Steering checkpoint 1: fast-path cancel before generate
                if loop_ctx.cancel_event.is_set():
                    logger.info("Streaming tool loop cancelled before round %d", _round_idx)
                    return

                # Re-apply gating so newly-activated skills expose their tools
                if loop_ctx.all_context_tools:
                    context = context.model_copy(
                        update={"tools": self._apply_tool_filters(loop_ctx.all_context_tools)}
                    )

                text_parts: list[str] = []
                tool_calls: list[StreamToolCall] = []

                async for event in self._generate_stream_with_retry(context):
                    if isinstance(event, StreamTextDelta):
                        text_parts.append(event.text)
                        yield event.text
                    elif isinstance(event, StreamToolCall):
                        tool_calls.append(event)

                if not tool_calls or self._tool_handler is None:
                    return

                # Don't execute tools on the final iteration — no generation follows
                if _round_idx >= self._max_tool_rounds:
                    logger.warning(
                        "Streaming tool loop reached max_tool_rounds=%d",
                        self._max_tool_rounds,
                    )
                    return

                # Timeout check
                if deadline and asyncio.get_event_loop().time() >= deadline:
                    logger.warning(
                        "Streaming tool loop timeout after %d rounds (%.0fs)",
                        _round_idx,
                        self._tool_loop_timeout_seconds,
                    )
                    return

                # Soft warning
                if _round_idx == self._tool_loop_warn_after:
                    logger.warning(
                        "Streaming tool loop reached %d rounds, still running", _round_idx
                    )

                logger.info(
                    "Streaming tool round %d: %d call(s)",
                    _round_idx + 1,
                    len(tool_calls),
                )

                # Append assistant message with text + tool calls
                parts: list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart] = []
                accumulated_text = "".join(text_parts)
                if accumulated_text:
                    parts.append(AITextPart(text=accumulated_text))
                for tc in tool_calls:
                    parts.append(AIToolCallPart(id=tc.id, name=tc.name, arguments=tc.arguments))
                context.messages.append(AIMessage(role="assistant", content=parts))

                # Execute tools concurrently
                result_parts = await self._execute_tools_parallel(tool_calls, telemetry)
                context.messages.append(AIMessage(role="tool", content=result_parts))

                # Yield inline XML so streaming consumers can render tool calls.
                # Format: <invoke name="..."></invoke><result>output</result>
                # Arguments are omitted to avoid large JSON breaking markdown rendering.
                for tc, rp in zip(tool_calls, result_parts, strict=False):
                    result_text = rp.result if isinstance(rp, AIToolResultPart) else ""
                    yield (
                        f'\n<invoke name="{tc.name}"></invoke>\n<result>{result_text}</result>\n'
                    )

                # Steering checkpoint 2: drain queue after tool execution
                context, should_cancel = self._drain_steering_queue(context, loop_ctx)
                if should_cancel:
                    logger.info("Streaming tool loop cancelled after round %d", _round_idx)
                    return
        finally:
            self._active_loops.pop(loop_ctx.loop_id, None)
            _current_loop_ctx.set(None)

    @property
    def _telemetry_provider(self) -> NoopTelemetryProvider:
        """Access telemetry provider (set by register_channel)."""
        return getattr(self, "_telemetry", None) or NoopTelemetryProvider()

    async def _generate_response(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        """Generate an AI response, executing tool calls if needed."""
        from roomkit.telemetry.context import get_current_span

        ai_context = await self._build_context(event, binding, context)
        telemetry = self._telemetry_provider
        span_id = telemetry.start_span(
            SpanKind.LLM_GENERATE,
            "llm.generate",
            parent_id=get_current_span(),
            room_id=event.room_id,
            channel_id=self.channel_id,
            attributes={
                Attr.PROVIDER: type(self._provider).__name__,
                Attr.LLM_STREAMING: False,
            },
        )
        try:
            response = await self._run_tool_loop(ai_context, parent_span_id=span_id)
        except ProviderError as exc:
            telemetry.end_span(span_id, status="error", error_message=str(exc))
            if exc.status_code == 404:
                logger.error(
                    "AI model not found (channel=%s, provider=%s): %s",
                    self.channel_id,
                    exc.provider,
                    exc,
                )
            elif exc.status_code and exc.status_code >= 500:
                logger.error(
                    "AI provider server error (channel=%s, provider=%s, status=%s): %s",
                    self.channel_id,
                    exc.provider,
                    exc.status_code,
                    exc,
                )
            else:
                logger.exception(
                    "AI provider error for channel %s",
                    self.channel_id,
                    extra={
                        "provider": exc.provider,
                        "retryable": exc.retryable,
                        "status_code": exc.status_code,
                    },
                )
            return ChannelOutput.empty()
        except Exception:
            telemetry.end_span(span_id, status="error", error_message="AI provider failed")
            logger.exception("AI provider failed for channel %s", self.channel_id)
            return ChannelOutput.empty()

        # End LLM span with usage attributes
        usage = response.usage or {}
        telemetry.end_span(
            span_id,
            attributes={
                Attr.LLM_INPUT_TOKENS: usage.get("input_tokens", 0),
                Attr.LLM_OUTPUT_TOKENS: usage.get("output_tokens", 0),
                Attr.LLM_TOOL_COUNT: len(response.tool_calls) if response.tool_calls else 0,
            },
        )

        response_event = RoomEvent(
            room_id=event.room_id,
            source=EventSource(
                channel_id=self.channel_id,
                channel_type=self.channel_type,
                provider=self.provider_name,
            ),
            content=TextContent(body=response.content),
            chain_depth=event.chain_depth + 1,
            metadata={"ai_usage": response.usage},
        )

        return ChannelOutput(
            responded=True,
            response_events=[response_event],
        )

    async def _run_tool_loop(
        self, context: AIContext, *, parent_span_id: str | None = None
    ) -> AIResponse:
        """Generate -> execute tools -> re-generate until a text response."""
        loop_ctx = _ToolLoopContext()
        loop_ctx.loop_id = str(id(loop_ctx))
        # Inherit participant role from the on_event-level context
        parent_lctx = _current_loop_ctx.get()
        if parent_lctx is not None:
            loop_ctx.current_participant_role = parent_lctx.current_participant_role
        _current_loop_ctx.set(loop_ctx)
        self._active_loops[loop_ctx.loop_id] = loop_ctx
        try:
            # Apply pre-queued directives (e.g. cancel enqueued before loop started)
            context, should_cancel = self._drain_steering_queue(context, loop_ctx)
            if should_cancel:
                return AIResponse(content="", tool_calls=[])
            response: AIResponse = await self._generate_with_retry(context)
            telemetry = self._telemetry_provider
            deadline = (
                asyncio.get_event_loop().time() + self._tool_loop_timeout_seconds
                if self._tool_loop_timeout_seconds
                else None
            )

            for round_idx in range(self._max_tool_rounds):
                if not response.tool_calls or self._tool_handler is None:
                    break

                # Steering checkpoint 1: fast-path cancel before generate
                if loop_ctx.cancel_event.is_set():
                    logger.info("Tool loop cancelled before round %d", round_idx)
                    break

                # Timeout check (after current round finishes)
                if deadline and asyncio.get_event_loop().time() >= deadline:
                    logger.warning(
                        "Tool loop timeout after %d rounds (%.0fs)",
                        round_idx,
                        self._tool_loop_timeout_seconds,
                    )
                    break

                # Soft warning
                if round_idx == self._tool_loop_warn_after:
                    logger.warning("Tool loop reached %d rounds, still running", round_idx)

                logger.info(
                    "Tool round %d: %d call(s)",
                    round_idx + 1,
                    len(response.tool_calls),
                )

                # Append assistant message with tool calls
                parts: list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart] = []
                if response.content:
                    parts.append(AITextPart(text=response.content))
                for tc in response.tool_calls:
                    parts.append(AIToolCallPart(id=tc.id, name=tc.name, arguments=tc.arguments))
                context.messages.append(AIMessage(role="assistant", content=parts))

                # Execute tools concurrently
                result_parts = await self._execute_tools_parallel(
                    response.tool_calls, telemetry, parent_span_id=parent_span_id
                )
                context.messages.append(AIMessage(role="tool", content=result_parts))

                # Steering checkpoint 2: drain queue after tool execution
                context, should_cancel = self._drain_steering_queue(context, loop_ctx)
                if should_cancel:
                    logger.info("Tool loop cancelled after round %d", round_idx)
                    break

                # Re-apply gating so newly-activated skills expose their tools
                if loop_ctx.all_context_tools:
                    context = context.model_copy(
                        update={"tools": self._apply_tool_filters(loop_ctx.all_context_tools)}
                    )

                # Re-generate with tool results (with retry)
                try:
                    response = await self._generate_with_retry(context)
                except ProviderError as exc:
                    if self._is_context_overflow(exc):
                        logger.warning("Context overflow at round %d. Compacting.", round_idx)
                        context = await self._compact_context(context)
                        response = await self._generate_with_retry(context)
                    else:
                        # Return partial result if we have accumulated text
                        accumulated = self._extract_accumulated_text(context.messages)
                        if accumulated:
                            return AIResponse(
                                content=accumulated + f"\n\n[Agent interrupted: {exc}]",
                                tool_calls=[],
                            )
                        raise

            return response
        finally:
            self._active_loops.pop(loop_ctx.loop_id, None)
            _current_loop_ctx.set(None)

    async def _execute_tools_parallel(
        self,
        tool_calls: list[Any],
        telemetry: Any,
        *,
        parent_span_id: str | None = None,
    ) -> list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart]:
        """Execute tool calls concurrently and return result parts."""
        assert self._tool_handler is not None
        handler = self._tool_handler

        async def _run_one(tc: Any) -> AIToolResultPart:
            logger.info("Executing tool: %s(%s)", tc.name, tc.id)

            # Execution guard: policy deny (defense-in-depth, role-aware)
            effective_policy = self._effective_tool_policy
            if (
                tc.name not in self._SKILL_INFRA_TOOLS
                and effective_policy
                and not effective_policy.is_allowed(tc.name)
            ):
                logger.warning("Tool %s blocked by policy", tc.name)
                return AIToolResultPart(
                    tool_call_id=tc.id,
                    name=tc.name,
                    result=json.dumps(
                        {"error": f"Tool '{tc.name}' is not permitted by the agent's tool policy."}
                    ),
                )

            # Execution guard: skill gating
            if tc.name not in self._SKILL_INFRA_TOOLS and tc.name in self._gated_tool_names:
                logger.warning("Tool %s blocked by skill gating", tc.name)
                return AIToolResultPart(
                    tool_call_id=tc.id,
                    name=tc.name,
                    result=json.dumps(
                        {
                            "error": (
                                f"Tool '{tc.name}' is gated by a skill. "
                                "Activate the skill first using activate_skill."
                            ),
                        }
                    ),
                )

            tool_span_id = telemetry.start_span(
                SpanKind.LLM_TOOL_CALL,
                f"tool.{tc.name}",
                parent_id=parent_span_id,
                attributes={"tool.name": tc.name, "tool.id": tc.id},
            )
            try:
                result = await handler(tc.name, tc.arguments)
                result = self._maybe_truncate_result(result)
                telemetry.end_span(tool_span_id)
            except Exception as exc:
                telemetry.end_span(tool_span_id, status="error", error_message=str(exc))
                logger.warning("Tool %s raised %s: %s", tc.name, type(exc).__name__, exc)
                result = f"Error executing tool '{tc.name}': {exc}"
            return AIToolResultPart(
                tool_call_id=tc.id,
                name=tc.name,
                result=result,
            )

        results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls])
        return list(results)

    # -- Retry / fallback / context management --------------------------------

    _MAX_TOOL_RESULT_TOKENS = 30_000  # ~120K characters

    async def _generate_with_retry(self, context: AIContext) -> AIResponse:
        """Call provider.generate() with retry and optional fallback."""
        policy = self._retry_policy or RetryPolicy(max_retries=0)
        last_error: ProviderError | None = None

        provider: AIProvider = self._provider
        for attempt in range(policy.max_retries + 1):
            try:
                return await provider.generate(context)
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable:
                    raise
                if attempt >= policy.max_retries:
                    break
                delay = min(
                    policy.base_delay_seconds * (policy.exponential_base**attempt),
                    policy.max_delay_seconds,
                )
                logger.warning(
                    "Provider error (attempt %d/%d, status=%s): %s. Retrying in %.1fs",
                    attempt + 1,
                    policy.max_retries,
                    exc.status_code,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted — try fallback provider
        if self._fallback_provider and last_error:
            logger.warning(
                "Primary provider failed after %d attempts. Trying fallback.",
                policy.max_retries + 1,
            )
            try:
                return await self._fallback_provider.generate(context)
            except ProviderError as fallback_exc:
                logger.error("Fallback provider also failed: %s", fallback_exc)
                raise last_error from fallback_exc

        if last_error:
            raise last_error
        raise RuntimeError("_generate_with_retry completed without result or exception")

    async def _generate_stream_with_retry(
        self, context: AIContext
    ) -> AsyncIterator[StreamTextDelta | StreamToolCall | StreamDone]:
        """Stream with retry on provider errors."""
        policy = self._retry_policy or RetryPolicy(max_retries=0)
        last_error: ProviderError | None = None

        for attempt in range(policy.max_retries + 1):
            try:
                async for event in self._provider.generate_structured_stream(context):
                    yield event
                return  # Stream completed successfully
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable:
                    raise
                if attempt >= policy.max_retries:
                    break
                delay = min(
                    policy.base_delay_seconds * (policy.exponential_base**attempt),
                    policy.max_delay_seconds,
                )
                logger.warning(
                    "Stream error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    policy.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        # Fallback
        if self._fallback_provider and last_error:
            logger.warning("Trying fallback provider for stream.")
            async for event in self._fallback_provider.generate_structured_stream(context):
                yield event
            return

        if last_error:
            raise last_error

    @staticmethod
    def _is_context_overflow(exc: ProviderError) -> bool:
        """Check if a provider error indicates context window overflow."""
        msg = str(exc).lower()
        return any(
            phrase in msg
            for phrase in [
                "context length exceeded",
                "maximum context length",
                "token limit",
                "too many tokens",
                "request too large",
                "prompt is too long",
            ]
        )

    async def _compact_context(self, context: AIContext) -> AIContext:
        """Emergency compaction: summarize the first half of messages."""
        messages = context.messages
        if len(messages) <= 4:
            raise ProviderError(
                "Context too large but cannot compact further (<=4 messages)",
                retryable=False,
            )

        split = len(messages) // 2
        old_messages = messages[:split]
        recent_messages = messages[split:]

        # Build a quick summary of old messages
        summary_parts: list[str] = []
        for msg in old_messages:
            role = msg.role
            if isinstance(msg.content, str):
                text = msg.content[:500]
            elif isinstance(msg.content, list):
                text = " ".join(
                    p.text[:200] if hasattr(p, "text") else f"[{p.type}]" for p in msg.content
                )[:500]
            else:
                text = str(msg.content)[:500]
            summary_parts.append(f"[{role}]: {text}")

        summary_text = "\n".join(summary_parts)
        summary_msg = AIMessage(
            role="user",
            content=(f"[Context compacted — earlier conversation summary]\n{summary_text}"),
        )

        return context.model_copy(update={"messages": [summary_msg] + recent_messages})

    @staticmethod
    def _extract_accumulated_text(messages: list[AIMessage]) -> str:
        """Extract accumulated assistant text from message history."""
        parts: list[str] = []
        for msg in messages:
            if msg.role != "assistant":
                continue
            if isinstance(msg.content, str):
                parts.append(msg.content)
            elif isinstance(msg.content, list):
                for p in msg.content:
                    if isinstance(p, AITextPart) and p.text:
                        parts.append(p.text)
        return "\n".join(parts)

    def _maybe_truncate_result(self, result: str) -> str:
        """Truncate oversized tool results, keeping start and end."""
        estimated = len(result) // 4 + 1
        if estimated <= self._MAX_TOOL_RESULT_TOKENS:
            return result
        max_chars = self._MAX_TOOL_RESULT_TOKENS * 4
        half = max_chars // 2
        truncated = estimated - self._MAX_TOOL_RESULT_TOKENS
        return result[:half] + f"\n\n[... truncated ~{truncated} tokens ...]\n\n" + result[-half:]

    # -- Role-based tool policy -------------------------------------------------

    def _resolve_participant_role(self, event: RoomEvent, context: RoomContext) -> str | None:
        """Look up the participant role for the event source."""
        pid = event.source.participant_id
        if not pid:
            return None
        for p in context.participants:
            if p.id == pid:
                return p.role
        return None

    @property
    def _effective_tool_policy(self) -> ToolPolicy | None:
        """Return the tool policy resolved for the current participant role."""
        if self._tool_policy is None:
            return None
        return self._tool_policy.resolve(self._get_loop_ctx().current_participant_role)

    # -- Tool policy / skill gating helpers ------------------------------------

    # Skill infrastructure tool names — never filtered by policy or gating
    _SKILL_INFRA_TOOLS = frozenset(
        {
            _TOOL_ACTIVATE_SKILL,
            _TOOL_READ_REFERENCE,
            _TOOL_RUN_SCRIPT,
        }
    )

    @property
    def _gated_tool_names(self) -> set[str]:
        """Collect tool names gated by skills that have NOT been activated yet."""
        if not self._skills:
            return set()
        activated = self._get_loop_ctx().activated_skills
        gated: set[str] = set()
        for meta in self._skills.all_metadata():
            if meta.name in activated:
                continue
            gated.update(meta.gated_tool_names)
        return gated

    def _apply_tool_filters(self, tools: list[AITool]) -> list[AITool]:
        """Apply tool policy and skill gating to a list of tools.

        Skill infrastructure tools (activate_skill, read_reference, run_script)
        are *never* filtered — they must always remain visible.

        Uses ``_effective_tool_policy`` which incorporates role-based overrides.
        """
        gated = self._gated_tool_names
        policy = self._effective_tool_policy
        result: list[AITool] = []
        for tool in tools:
            # Skill infra tools always pass
            if tool.name in self._SKILL_INFRA_TOOLS:
                result.append(tool)
                continue
            # Tool policy filter (role-aware)
            if policy and not policy.is_allowed(tool.name):
                continue
            # Skill gating filter
            if tool.name in gated:
                continue
            result.append(tool)
        return result

    # -- Context building -------------------------------------------------------

    async def _build_context(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> AIContext:
        """Build AI context from room events.

        Per-room overrides can be set via binding.metadata:
        - system_prompt: Override the channel's default system prompt
        - temperature: Override the channel's default temperature
        - max_tokens: Override the channel's default max_tokens
        - tools: List of tool definitions for function calling
        """
        # Per-room overrides from binding metadata
        system_prompt = binding.metadata.get("system_prompt", self._system_prompt)
        temperature = binding.metadata.get("temperature", self._temperature)
        max_tokens = binding.metadata.get("max_tokens", self._max_tokens)
        raw_tools = binding.metadata.get("tools", [])

        # Convert raw tool dicts to AITool instances
        tools = [
            AITool(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("parameters", {}),
            )
            for t in raw_tools
        ]

        # Inject extra tools (orchestration handoff, etc.)
        tools.extend(self._extra_tools)

        # Inject skill tools and prompt (infra tools added here, gated tools later)
        if self._skills and self._skills.skill_count > 0:
            tools.extend(self._skill_tools())
            preamble = _SKILLS_PREAMBLE
            if not self._script_executor:
                preamble += _SKILLS_NO_SCRIPTS_NOTE
            skills_xml = self._skills.to_prompt_xml()
            skill_block = f"\n\n{preamble}\n\n{skills_xml}"
            system_prompt = (system_prompt or "") + skill_block

        # Store unfiltered tool list for re-application after skill activation
        loop_ctx = self._get_loop_ctx()
        loop_ctx.all_context_tools = list(tools)

        # Apply tool policy + skill gating visibility filters
        tools = self._apply_tool_filters(tools)

        # Retrieve memory
        memory_result = await self._memory.retrieve(
            event.room_id,
            event,
            context,
            channel_id=self.channel_id,
        )

        messages: list[AIMessage] = []

        # Pre-built messages from memory (e.g. summaries)
        messages.extend(memory_result.messages)

        # Convert memory events using AIChannel content extraction
        for past_event in memory_result.events:
            role = self._determine_role(past_event)
            content = self._extract_content(past_event)
            if content:
                messages.append(AIMessage(role=role, content=content))

        # Add current event
        content = self._extract_content(event)
        if content:
            messages.append(AIMessage(role="user", content=content))

        # Determine target channel capabilities for capability-aware generation
        # Use intersection of all transport bindings' media types (weakest common)
        transport_bindings = [
            b
            for b in context.bindings
            if b.category == ChannelCategory.TRANSPORT and b.channel_id != self.channel_id
        ]
        if transport_bindings:
            common_types = set(transport_bindings[0].capabilities.media_types)
            for b in transport_bindings[1:]:
                common_types &= set(b.capabilities.media_types)
            target_media = list(common_types)
            # Intersect capabilities: AND for booleans, MIN for numeric limits
            caps0 = transport_bindings[0].capabilities
            merged = caps0.model_dump()
            merged["media_types"] = target_media
            for b in transport_bindings[1:]:
                other = b.capabilities
                for field_name in (
                    "supports_threading",
                    "supports_reactions",
                    "supports_edit",
                    "supports_delete",
                    "supports_read_receipts",
                    "supports_typing",
                    "supports_templates",
                    "supports_rich_text",
                    "supports_buttons",
                    "supports_cards",
                    "supports_quick_replies",
                    "supports_media",
                    "supports_audio",
                    "supports_video",
                ):
                    merged[field_name] = merged[field_name] and getattr(other, field_name)
                for field_name in (
                    "max_length",
                    "max_buttons",
                    "max_media_size_bytes",
                    "max_audio_duration_seconds",
                    "max_video_duration_seconds",
                ):
                    a, b_val = merged[field_name], getattr(other, field_name)
                    if a is not None and b_val is not None:
                        merged[field_name] = min(a, b_val)
                    elif b_val is not None:
                        merged[field_name] = b_val
            from roomkit.models.channel import ChannelCapabilities

            target_caps = ChannelCapabilities(**merged)
        else:
            target_media = []
            target_caps = None

        return AIContext(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            room=context,
            target_capabilities=target_caps,
            target_media_types=target_media,
        )

    def _determine_role(self, event: RoomEvent) -> str:
        if event.source.channel_id == self.channel_id:
            return "assistant"
        return "user"

    def _extract_content(
        self,
        event: RoomEvent,
    ) -> str | list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart]:
        """Extract content, including images if provider supports vision."""
        content = event.content

        if not self._provider.supports_vision:
            # Text-only fallback (existing behavior)
            return self._extract_text(event)

        # Build multimodal content
        if isinstance(content, TextContent):
            return content.body  # Simple case: just text

        if isinstance(content, MediaContent):
            parts: list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart] = []
            if content.caption:
                parts.append(AITextPart(text=content.caption))
            parts.append(AIImagePart(url=content.url, mime_type=content.mime_type))
            return parts

        if isinstance(content, CompositeContent):
            cparts: list[AITextPart | AIImagePart | AIToolCallPart | AIToolResultPart] = []
            for part in content.parts:
                if isinstance(part, TextContent):
                    cparts.append(AITextPart(text=part.body))
                elif isinstance(part, MediaContent):
                    if part.caption:
                        cparts.append(AITextPart(text=part.caption))
                    cparts.append(AIImagePart(url=part.url, mime_type=part.mime_type))
            return cparts if cparts else ""

        # Fallback for other types
        return self._extract_text(event)

    def _extract_text(self, event: RoomEvent) -> str:
        if isinstance(event.content, TextContent):
            return event.content.body
        return ""

    # -- Skill integration --------------------------------------------------

    def _skill_tools(self) -> list[AITool]:
        """Build the list of AITool definitions for skill operations."""
        tools = [
            AITool(
                name=_TOOL_ACTIVATE_SKILL,
                description=(
                    "Activate a skill to get its full instructions, "
                    "available scripts, and reference files."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the skill to activate.",
                        },
                    },
                    "required": ["name"],
                },
            ),
            AITool(
                name=_TOOL_READ_REFERENCE,
                description="Read a reference file from a skill.",
                parameters={
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Name of the skill.",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Reference filename to read.",
                        },
                    },
                    "required": ["skill_name", "filename"],
                },
            ),
        ]
        if self._script_executor:
            tools.append(
                AITool(
                    name=_TOOL_RUN_SCRIPT,
                    description="Run a script from a skill's scripts/ directory.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "description": "Name of the skill.",
                            },
                            "script_name": {
                                "type": "string",
                                "description": "Script filename to run.",
                            },
                            "arguments": {
                                "type": "object",
                                "description": "Optional key-value arguments.",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "required": ["skill_name", "script_name"],
                    },
                )
            )
        return tools

    async def _skill_aware_tool_handler(self, name: str, arguments: dict[str, Any]) -> str:
        """Intercept skill tool calls; delegate the rest to user handler."""
        if name == _TOOL_ACTIVATE_SKILL:
            return await self._handle_activate_skill(arguments)
        if name == _TOOL_READ_REFERENCE:
            return await self._handle_read_reference(arguments)
        if name == _TOOL_RUN_SCRIPT:
            return await self._handle_run_script(arguments)
        if self._user_tool_handler:
            return await self._user_tool_handler(name, arguments)
        return json.dumps({"error": f"Unknown tool: {name}"})

    async def _handle_activate_skill(self, arguments: dict[str, Any]) -> str:
        """Load and return full skill instructions, tracking activation for gating."""
        skill_name = arguments.get("name", "")
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})

        skill = self._skills.get_skill(skill_name)
        if skill is None:
            available = self._skills.skill_names
            return json.dumps(
                {
                    "error": f"Skill {skill_name!r} not found",
                    "available_skills": available,
                }
            )

        # Track activation so gated tools become visible on next round
        self._get_loop_ctx().activated_skills.add(skill_name)

        result: dict[str, Any] = {
            "name": skill.name,
            "description": skill.description,
            "instructions": skill.instructions,
        }
        scripts = skill.list_scripts()
        if scripts:
            result["scripts"] = scripts
        refs = skill.list_references()
        if refs:
            result["references"] = refs
        return json.dumps(result)

    async def _handle_read_reference(self, arguments: dict[str, Any]) -> str:
        """Read a reference file from a skill."""
        skill_name = arguments.get("skill_name", "")
        filename = arguments.get("filename", "")
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})

        skill = self._skills.get_skill(skill_name)
        if skill is None:
            return json.dumps({"error": f"Skill {skill_name!r} not found"})

        try:
            content = skill.read_reference(filename)
            return json.dumps({"filename": filename, "content": content})
        except (ValueError, FileNotFoundError) as exc:
            return json.dumps({"error": str(exc)})

    async def _handle_run_script(self, arguments: dict[str, Any]) -> str:
        """Execute a script via the configured ScriptExecutor."""
        skill_name = arguments.get("skill_name", "")
        script_name = arguments.get("script_name", "")
        script_args = arguments.get("arguments")
        if not self._skills:
            return json.dumps({"error": "No skills registry configured"})
        if not self._script_executor:
            return json.dumps({"error": "Script execution is not available"})

        skill = self._skills.get_skill(skill_name)
        if skill is None:
            return json.dumps({"error": f"Skill {skill_name!r} not found"})

        try:
            result = await self._script_executor.execute(skill, script_name, arguments=script_args)
            return result.model_dump_json()
        except Exception as exc:
            logger.exception("Script execution failed: %s/%s", skill_name, script_name)
            return json.dumps({"error": f"Script execution failed: {exc}"})
