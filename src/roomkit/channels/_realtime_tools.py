"""Tool call handling for RealtimeVoiceChannel."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.channels._skill_constants import TOOL_ACTIVATE_SKILL
from roomkit.models.enums import ChannelType, HookTrigger
from roomkit.providers.ai.base import AIImagePart, AITextPart
from roomkit.telemetry.base import Attr, SpanKind
from roomkit.tools.validation import fold_hoisted_arguments, validate_tool_arguments

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.voice.backends.base import VoiceBackend
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_voice")

_LOOP_SEGMENT_BUDGET_S = 0.050
"""Sync work on the event loop past this delays realtime pacing.

The SIP pacer's jitter headroom is 60ms — one fused stretch beyond it is
an audible drop-out on a concurrent call.  Tool-call segments are timed
individually so the culprit is named in the logs without an asyncio
set_debug hunt."""


def result_text(raw: Any) -> str:
    """Flatten a tool handler result for a voice provider.

    A handler shared with an ``AIChannel`` may answer with a content-part
    list (text + images); a speech provider cannot consume an image, so the
    list flattens the way ``AIToolResultPart.as_text()`` does — text joined,
    ``[image]`` placeholders. ``json.dumps`` on such a list would raise on
    the pydantic parts instead. Anything else keeps the JSON coercion.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list) and all(isinstance(p, AITextPart | AIImagePart) for p in raw):
        return "\n".join(p.text if isinstance(p, AITextPart) else "[image]" for p in raw)
    return json.dumps(raw)


@runtime_checkable
class RealtimeToolsHost(Protocol):
    """Contract: capabilities a host class must provide for RealtimeToolsMixin.

    Attributes provided by the host's ``__init__``:
        _state_lock: Guards mutable per-session state from concurrent access.
        _session_rooms: Maps session IDs to room IDs.
        _session_spans: Active telemetry session span per session.
        _turn_spans: Active telemetry turn span per session.
        _session_tools: Per-session tool definitions.
        _tool_handler: User-provided tool handler callback.
        _tools: Default tool definitions.
        _mute_on_tool_call: Whether to mute mic during tool execution.
        _tool_result_max_length: Max characters for tool result.
        _skill_support: Skill infrastructure support.
        _provider: The realtime voice provider.
        _transport: The voice backend transport.
        _framework: The RoomKit framework instance (or None).
        channel_id: Channel identifier.
        _telemetry_provider: Telemetry provider for spans.

    Cross-mixin methods (implemented elsewhere in the MRO):
        _track_task: Schedule an async task with exception handling.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_spans: dict[str, Any]
    _turn_spans: dict[str, Any]
    _session_tools: dict[str, Any]
    _tool_handler: Any
    _tools: Any
    _system_prompt: str | None
    _mute_on_tool_call: bool
    _tool_result_max_length: int
    _skill_support: Any
    _tool_search_support: Any
    _provider: RealtimeVoiceProvider
    _transport: VoiceBackend
    _framework: RoomKit | None
    _transcription_order_locks: dict[str, asyncio.Lock]
    channel_id: str
    _telemetry_provider: Any

    def _track_task(self, loop: Any, coro: Any, *, name: str) -> Any: ...


class RealtimeToolsMixin:
    """Tool call execution for RealtimeVoiceChannel.

    Host contract: :class:`RealtimeToolsHost`.
    """

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_spans: dict[str, Any]
    _turn_spans: dict[str, Any]
    _session_tools: dict[str, Any]
    _tool_handler: Any
    _tools: Any
    _system_prompt: str | None
    _mute_on_tool_call: bool
    _tool_result_max_length: int
    _skill_support: Any
    _tool_search_support: Any
    _provider: RealtimeVoiceProvider
    _transport: VoiceBackend
    _framework: RoomKit | None
    _transcription_order_locks: dict[str, asyncio.Lock]
    channel_id: str
    _telemetry_provider: Any

    _track_task: Any  # see RealtimeToolsHost — cross-mixin

    def _on_provider_tool_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Handle tool call from provider."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._track_task(
            loop,
            self._handle_tool_call(session, call_id, name, arguments),
            name=f"rt_tool_call:{session.id}:{call_id}",
        )

    async def _handle_tool_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Execute a tool call and submit the result to the provider.

        If a ``tool_handler`` was provided, it is called directly.
        The ``ON_TOOL_CALL`` hook is then fired (handler result, if any,
        is passed as ``event.result`` so the hook can observe or override).
        """
        # Order barrier: a tool call must not overtake the transcriptions the
        # provider emitted before it. The user final that closes the current
        # utterance travels the serialised transcription queue, while tool
        # calls run in their own task — unbarriered, the tool reaches the
        # application first and the late final reads as new user speech.
        # Pass through the same FIFO lock, then release: tool execution
        # itself must not hold transcriptions back.
        with self._state_lock:
            order_lock = self._transcription_order_locks.setdefault(session.id, asyncio.Lock())
        async with order_lock:
            pass

        with self._state_lock:
            room_id = self._session_rooms.get(session.id)
            _rt_parent = self._session_spans.get(session.id)
            parent = self._turn_spans.get(session.id) or _rt_parent

        from roomkit.telemetry.context import reset_span, set_current_span

        _rt_tok = set_current_span(_rt_parent) if _rt_parent else None

        telemetry = self._telemetry_provider
        tool_span_id = telemetry.start_span(
            SpanKind.REALTIME_TOOL_CALL,
            f"realtime_tool:{name}",
            parent_id=parent,
            attributes={Attr.REALTIME_TOOL_NAME: name},
            room_id=room_id,
            session_id=session.id,
            channel_id=self.channel_id,
        )

        if self._mute_on_tool_call and self._transport is not None:
            self._transport.set_input_muted(session, True)

        try:
            result_str: str

            # Pre-execution gate (parity with the classic AI path): validate
            # arguments and run BEFORE_TOOL_USE BEFORE the call is routed, so a
            # block prevents the side effect instead of only hiding the result.
            # Its scope is every call: hook-only mode serves the tool from
            # ON_TOOL_CALL, and an infrastructure tool serves itself, but a
            # host auditing or denying tool use must see both.
            arguments, denial, gate_context = await self._authorize_realtime_tool(
                name, arguments, call_id, room_id, session
            )
            if denial is not None:
                await self._provider.submit_tool_result(session, call_id, denial)
                telemetry.end_span(tool_span_id)
                logger.info(
                    "Realtime tool %s(%s) denied before execution for session %s",
                    name,
                    call_id,
                    session.id,
                )
                return

            # Tool Search infrastructure tools — handle internally
            if self._tool_search_support and self._tool_search_support.is_search_tool(name):
                await self._dispatch_tool_search_call(
                    session, call_id, name, arguments, room_id, tool_span_id
                )
                return

            # Skill infrastructure tools — handle internally
            if self._skill_support and self._skill_support.is_skill_tool(name):
                result_str = await self._skill_support.handle_tool_call(
                    name, arguments, session.id
                )

                # Fire ON_TOOL_CALL hook observationally — the skill tool
                # has already executed, but audit + UI-broadcast hooks
                # need visibility into it the same as any other tool.
                if self._framework and room_id:
                    from roomkit.models.tool_call import ToolCallEvent

                    skill_event = ToolCallEvent(
                        channel_id=self.channel_id,
                        channel_type=ChannelType.REALTIME_VOICE,
                        tool_call_id=call_id,
                        name=name,
                        arguments=arguments,
                        result=result_str,
                        room_id=room_id,
                        session=session,
                    )
                    try:
                        skill_ctx = await self._framework._build_context(room_id)
                        await self._framework.hook_engine.run_sync_hooks(
                            room_id,
                            HookTrigger.ON_TOOL_CALL,
                            skill_event,
                            skill_ctx,
                            skip_event_filter=True,
                        )
                    except Exception:
                        logger.debug(
                            "ON_TOOL_CALL observation failed for skill tool %s",
                            name,
                            exc_info=True,
                        )

                # Submit the tool result FIRST — the model's pending
                # function call is bound to the live WebSocket. A
                # subsequent reconfigure tears that connection down and
                # replaces it with a fresh ``live_session`` that has no
                # record of ``call_id``, so the tool response would be
                # lost and the model would hang forever waiting on it.
                await self._provider.submit_tool_result(session, call_id, result_str)

                if name == TOOL_ACTIVATE_SKILL and self._provider.supports_mid_session_reconfigure:
                    # On providers that can safely reconfigure mid-session
                    # (e.g. Gemini 2.5, OpenAI Realtime), push the activated
                    # skill's body into ``system_instruction`` so it lives as
                    # binding rules. Skip when the provider cannot reconfigure
                    # — the body must reach the model some other way (e.g. the
                    # ``inline_full`` skill_delivery_mode that bakes every
                    # skill into the initial system_instruction).
                    #
                    # CRITICAL: ``reconfigure`` rebuilds the provider's
                    # session config from scratch — passing ``tools=None``
                    # erases the tool surface. Always pass the current
                    # visible tool list, even when it is unchanged.
                    with self._state_lock:
                        base_tools = self._session_tools.get(session.id, self._tools or [])
                    all_tools = self._skill_support.skill_tool_dicts() + base_tools
                    current_visible = self._skill_support.get_visible_tools(all_tools, session.id)
                    addendum = self._skill_support.activated_skills_prompt(session.id)
                    if addendum and self._system_prompt:
                        new_prompt: str | None = f"{self._system_prompt}\n\n{addendum}"
                    elif addendum:
                        new_prompt = addendum
                    else:
                        new_prompt = None

                    if (
                        addendum is not None
                        or self._skill_support.newly_visible_after_activation(
                            all_tools, session.id, arguments.get("name", "")
                        )
                        is not None
                    ):
                        await self._provider.reconfigure(
                            session,
                            tools=current_visible,
                            system_prompt=new_prompt,
                        )

                telemetry.end_span(tool_span_id)
                logger.info(
                    "Skill tool %s(%s) handled for session %s",
                    name,
                    call_id,
                    session.id,
                )
                return

            # Run tool_handler (if exists).
            handler_result: str | None = None
            if self._tool_handler is not None:
                logger.info(
                    "Executing tool %s(%s) via handler for session %s",
                    name,
                    call_id,
                    session.id,
                )
                from roomkit.channels._realtime_context import _current_voice_session

                t_seg = time.perf_counter()
                token = _current_voice_session.set(session)
                try:
                    raw = await self._tool_handler(name, arguments)
                finally:
                    _current_voice_session.reset(token)
                logger.debug(
                    "tool %s handler segment: %.0fms wall",
                    name,
                    (time.perf_counter() - t_seg) * 1000,
                )

                t_seg = time.perf_counter()
                handler_result = result_text(raw)
                ser_s = time.perf_counter() - t_seg
                if ser_s > _LOOP_SEGMENT_BUDGET_S:
                    # Pure sync CPU (wall == loop hold), and it runs on the
                    # FULL result before truncation caps it.
                    logger.warning(
                        "Tool %s result serialization held the event loop for "
                        "%.0fms (%d chars, budget ~%.0fms) — concurrent "
                        "realtime audio may underrun; return a string or a "
                        "compact reference instead of a large object",
                        name,
                        ser_s * 1000,
                        len(handler_result),
                        _LOOP_SEGMENT_BUDGET_S * 1000,
                    )
                # Yield so realtime pacing gets a slot between the handler
                # segment and hook dispatch — sync hooks run inline next and
                # would otherwise fuse with this segment into one loop step.
                await asyncio.sleep(0)

            # Run ON_TOOL_CALL hook (if framework + room).
            from roomkit.models.tool_call import ToolCallEvent

            tool_event = ToolCallEvent(
                channel_id=self.channel_id,
                channel_type=ChannelType.REALTIME_VOICE,
                tool_call_id=call_id,
                name=name,
                arguments=arguments,
                result=handler_result,
                room_id=room_id,
                session=session,
            )

            if self._framework and room_id:
                result_str = await self._fire_tool_hook(
                    tool_event, room_id, handler_result, name, call_id, session, gate_context
                )
                # Same reason as the post-handler yield: don't fuse hook
                # dispatch with submission into one loop step.
                await asyncio.sleep(0)
            elif handler_result is not None:
                result_str = handler_result
            else:
                result_str = json.dumps({"error": f"No handler for tool {name}"})

            if len(result_str) > self._tool_result_max_length:
                result_str = self._truncate_tool_result(result_str, name, call_id, session.id)

            await self._provider.submit_tool_result(session, call_id, result_str)

            telemetry.end_span(tool_span_id)
            logger.info(
                "Tool call %s(%s) handled for session %s",
                name,
                call_id,
                session.id,
            )

        except Exception:
            telemetry.end_span(tool_span_id, status="error", error_message=f"tool {name} failed")
            logger.exception("Error handling tool call %s for session %s", call_id, session.id)
            try:
                await self._provider.submit_tool_result(
                    session,
                    call_id,
                    json.dumps({"error": "Internal error handling tool call"}),
                )
            except Exception:
                logger.exception("Error submitting fallback tool result")
        finally:
            if self._mute_on_tool_call and self._transport is not None:
                self._transport.set_input_muted(session, False)
            if _rt_tok is not None:
                reset_span(_rt_tok)

    def _tool_parameters(self, name: str, session: VoiceSession) -> dict[str, Any] | None:
        """Return the declared ``parameters`` schema for realtime tool *name*.

        ``None`` when the tool's schema is unknown (skips argument validation).
        """
        with self._state_lock:
            tools = self._session_tools.get(session.id) or self._tools or []
        for t in tools:
            if isinstance(t, dict) and t.get("name") == name:
                params = t.get("parameters")
                return params if isinstance(params, dict) else None
        return None

    def _is_declared_realtime_tool(self, name: str, session: VoiceSession) -> bool:
        """Return whether *name* is in a non-empty session tool catalogue.

        An empty catalogue retains the historical hook-only/dynamic-handler
        mode. Once declarations exist, however, a provider cannot invent an
        undeclared name and reach a generic dispatcher.

        Infrastructure tools (Tool Search, skills) are declared by the channel
        rather than by the caller's catalogue, so they answer ``True`` without
        appearing in it.
        """
        if self._is_infrastructure_tool(name):
            return True
        with self._state_lock:
            tools = self._session_tools.get(session.id) or self._tools or []
        if not tools:
            return True
        return any(isinstance(tool, dict) and tool.get("name") == name for tool in tools)

    def _is_infrastructure_tool(self, name: str) -> bool:
        """Whether *name* is served by the channel itself, not by the host."""
        if self._tool_search_support and self._tool_search_support.is_search_tool(name):
            return True
        return bool(self._skill_support and self._skill_support.is_skill_tool(name))

    async def _authorize_realtime_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        call_id: str,
        room_id: str | None,
        session: VoiceSession,
    ) -> tuple[dict[str, Any], str | None, RoomContext | None]:
        """Pre-execution gate for realtime tool calls (parity with the classic
        AI path).

        Folds a flattened hub-tool call back into ``params``, validates the
        arguments against the declared schema, and runs BEFORE_TOOL_USE
        so a block prevents the side effect rather than only hiding the result.
        Hooks may replace the arguments through ``metadata["arguments"]``; the
        replacement is validated before it can reach the handler.

        Returns the effective arguments, an optional denial result, and the
        room context this gate built — ``None`` when it built none. The caller
        hands that context to :meth:`_fire_tool_hook` as ``carrying`` so one
        tool call deserialises the room history once instead of twice.
        """
        if not self._is_declared_realtime_tool(name, session):
            logger.warning("Realtime provider requested undeclared tool %s", name)
            return arguments, json.dumps({"error": f"Tool '{name}' is not declared"}), None

        # Argument validation against the declared schema (fail-closed), after
        # repairing a hub tool's flattened ``params`` — same gate, same order as
        # the classic AI path.
        params = self._tool_parameters(name, session)
        if params is not None:
            folded, fold_error = fold_hoisted_arguments(params, arguments)
            if fold_error is not None:
                logger.warning("Realtime tool %s arguments ambiguous: %s", name, fold_error)
                return (
                    arguments,
                    json.dumps({"error": f"Invalid arguments for '{name}': {fold_error}"}),
                    None,
                )
            if folded is not None:
                logger.info(
                    "Realtime tool %s: folded hoisted arguments %s into its container "
                    "(provider=%s, model=%s)",
                    name,
                    sorted(set(arguments) - set(folded)),
                    self._provider.name,
                    self._provider.model_name,
                )
                arguments = folded
            arg_error = validate_tool_arguments(params, arguments)
            if arg_error is not None:
                logger.warning("Realtime tool %s arguments rejected: %s", name, arg_error)
                return (
                    arguments,
                    json.dumps({"error": f"Invalid arguments for '{name}': {arg_error}"}),
                    None,
                )

        # Execution guard: skill gating (parity with the classic AI path).
        # Hiding a gated tool from the catalogue is not enforcement — the model
        # may still name one it saw before the skill was deactivated.
        if self._skill_support is not None and self._skill_support.is_gated(name, session.id):
            logger.warning("Realtime tool %s blocked by skill gating", name)
            return (
                arguments,
                json.dumps(
                    {
                        "error": (
                            f"Tool '{name}' is gated by a skill. "
                            "Activate the skill first using activate_skill."
                        )
                    }
                ),
                None,
            )

        # BEFORE_TOOL_USE gate (needs a framework + room to run room hooks).
        if self._framework is None or not room_id:
            return arguments, None, None
        # Building a context costs two store reads; skip it when nothing listens.
        # Schema validation above stays unconditional — it needs no context.
        if not self._framework.hook_engine.has_hooks(HookTrigger.BEFORE_TOOL_USE):
            return arguments, None, None
        from roomkit.models.tool_call import ToolCallEvent

        pre_event = ToolCallEvent(
            channel_id=self.channel_id,
            channel_type=ChannelType.REALTIME_VOICE,
            tool_call_id=call_id,
            name=name,
            arguments=arguments,
            result=None,
            room_id=room_id,
            session=session,
        )
        context = await self._framework._build_context(room_id)
        hook_result = await self._framework.hook_engine.run_sync_hooks(
            room_id,
            HookTrigger.BEFORE_TOOL_USE,
            pre_event,
            context,
            skip_event_filter=True,
        )
        await self._framework._emit_framework_event(
            "before_tool_use",
            room_id=room_id,
            channel_id=self.channel_id,
            data={
                "tool_name": name,
                "tool_call_id": call_id,
                "allowed": hook_result.allowed,
                "reason": hook_result.reason,
            },
        )

        if not hook_result.allowed:
            logger.info("Realtime tool %s denied by BEFORE_TOOL_USE hook", name)
            return (
                arguments,
                json.dumps(
                    {"error": hook_result.reason or f"Tool '{name}' denied by pre-execution hook."}
                ),
                context,
            )

        rewritten = hook_result.metadata.get("arguments")
        if "arguments" in hook_result.metadata and not isinstance(rewritten, dict):
            logger.error(
                "BEFORE_TOOL_USE hook returned non-object arguments for realtime tool %s "
                "— denying tool call",
                name,
            )
            return (
                arguments,
                json.dumps(
                    {"error": f"Invalid rewritten arguments for '{name}': expected an object"}
                ),
                context,
            )

        effective_arguments = rewritten if isinstance(rewritten, dict) else arguments
        # No fold here, deliberately: a hook's rewritten arguments are user code,
        # and repairing them would hide the hook's bug. The model's own call was
        # already folded above.
        if params is not None:
            arg_error = validate_tool_arguments(params, effective_arguments)
            if arg_error is not None:
                logger.warning(
                    "Realtime tool %s post-hook arguments rejected: %s", name, arg_error
                )
                return (
                    effective_arguments,
                    json.dumps(
                        {"error": f"Invalid rewritten arguments for '{name}': {arg_error}"}
                    ),
                    context,
                )
        return effective_arguments, None, context

    async def _fire_tool_hook(
        self,
        tool_event: Any,
        room_id: str,
        handler_result: str | None,
        name: str,
        call_id: str,
        session: VoiceSession,
        carrying: RoomContext | None = None,
    ) -> str:
        """Fire ON_TOOL_CALL hook and determine final result.

        ``carrying`` is the context the pre-execution gate already built for
        this same call, when it built one: handing it over spares the room
        history a second deserialisation per tool call.
        """
        assert self._framework is not None  # guarded by caller  # noqa: S101
        t_seg = time.perf_counter()
        context = await self._framework._build_context(room_id, carrying=carrying)
        hook_result = await self._framework.hook_engine.run_sync_hooks(
            room_id,
            HookTrigger.ON_TOOL_CALL,
            tool_event,
            context,
            skip_event_filter=True,
        )
        # Wall time, not loop hold — sync hooks may legitimately await I/O.
        logger.debug(
            "tool %s ON_TOOL_CALL segment: %.0fms wall",
            name,
            (time.perf_counter() - t_seg) * 1000,
        )

        if not hook_result.allowed:
            result_str = json.dumps({"error": hook_result.reason or "Tool call blocked by hook"})
        elif "result" in hook_result.metadata:
            hook_val = hook_result.metadata["result"]
            result_str = hook_val if isinstance(hook_val, str) else json.dumps(hook_val)
        elif handler_result is not None:
            result_str = handler_result
        elif hook_result.hook_errors:
            errors = "; ".join(f"{e['hook']}: {e['error']}" for e in hook_result.hook_errors)
            result_str = json.dumps(
                {
                    "error": f"Tool call failed: {errors}",
                }
            )
        else:
            result_str = json.dumps({"status": "ok"})

        await self._framework._emit_framework_event(
            "tool_call",
            room_id=room_id,
            channel_id=self.channel_id,
            data={
                "tool_name": name,
                "tool_call_id": call_id,
                "channel_type": str(ChannelType.REALTIME_VOICE),
            },
        )
        return result_str

    async def _dispatch_tool_search_call(
        self,
        session: VoiceSession,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        room_id: str | None,
        tool_span_id: Any,
    ) -> None:
        """Handle find_tools / list_tools and reconfigure on a successful match.

        ``find_tools`` returns ``(json_result, updated_tool_list_or_None)`` —
        when the second element is non-None the matched tools became
        invocable for this session and we must push them via
        ``provider.reconfigure(tools=...)``. ``reconfigure`` rebuilds
        the provider config from scratch, so we also pass the current
        ``system_prompt`` (with any active skill addendum) to avoid
        wiping it. The skills layer is composed back on top so its
        infra tools (activate_skill, …) stay live.
        """
        telemetry = self._telemetry_provider
        result_str, updated = await self._tool_search_support.handle_tool_call(
            name, arguments, session.id
        )

        # Fire ON_TOOL_CALL hook so audit + UI-broadcast hooks see search calls.
        if self._framework and room_id:
            from roomkit.models.tool_call import ToolCallEvent

            search_event = ToolCallEvent(
                channel_id=self.channel_id,
                channel_type=ChannelType.REALTIME_VOICE,
                tool_call_id=call_id,
                name=name,
                arguments=arguments,
                result=result_str,
                room_id=room_id,
                session=session,
            )
            try:
                ctx = await self._framework._build_context(room_id)
                await self._framework.hook_engine.run_sync_hooks(
                    room_id,
                    HookTrigger.ON_TOOL_CALL,
                    search_event,
                    ctx,
                    skip_event_filter=True,
                )
            except Exception:
                logger.debug(
                    "ON_TOOL_CALL observation failed for tool-search tool %s",
                    name,
                    exc_info=True,
                )

        # Submit the tool result FIRST: the model's pending call is bound
        # to the live WebSocket. Reconfigure would tear that connection
        # down and the response would be lost.
        await self._provider.submit_tool_result(session, call_id, result_str)

        if updated is not None and self._provider.supports_mid_session_reconfigure:
            # Recompose: skill tools (if any) sit alongside the search-tool
            # output, then preserve any active skill bodies in the prompt.
            full_tools = updated
            if self._skill_support:
                skill_defs = self._skill_support.skill_tool_dicts()
                # Avoid duplicating any skill tool already present in updated.
                seen = {t.get("name") for t in updated}
                full_tools = [t for t in skill_defs if t.get("name") not in seen] + updated
                full_tools = self._skill_support.get_visible_tools(full_tools, session.id)

            new_prompt: str | None = self._system_prompt
            if self._skill_support:
                addendum = self._skill_support.activated_skills_prompt(session.id)
                if addendum and new_prompt:
                    new_prompt = f"{new_prompt}\n\n{addendum}"
                elif addendum:
                    new_prompt = addendum

            await self._provider.reconfigure(
                session,
                tools=full_tools,
                system_prompt=new_prompt,
            )
        elif updated is not None:
            logger.debug(
                "Tool-search match for %s but provider %s cannot reconfigure "
                "mid-session — newly matched tools will not be exposed this turn",
                name,
                self._provider.name,
            )

        telemetry.end_span(tool_span_id)
        logger.info(
            "Tool-search %s(%s) handled for session %s (%d tools now visible)",
            name,
            call_id,
            session.id,
            len(updated) if updated is not None else 0,
        )

    def _truncate_tool_result(
        self,
        result_str: str,
        name: str,
        call_id: str,
        session_id: str,
    ) -> str:
        """Truncate an oversized tool result with a notice."""
        original_len = len(result_str)
        logger.warning(
            "Tool result for %s(%s) truncated from %d to %d chars (session %s)",
            name,
            call_id,
            original_len,
            self._tool_result_max_length,
            session_id,
        )
        notice = (
            f"\n... [truncated — original result was {original_len} chars. "
            "The full content has been delivered to the client.]"
        )
        return result_str[: self._tool_result_max_length - len(notice)] + notice
