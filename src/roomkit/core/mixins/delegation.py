"""DelegationMixin — task delegation to child rooms."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.core.exceptions import ChannelNotRegisteredError
from roomkit.core.mixins.helpers import HelpersMixin
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    ChannelCategory,
    ChannelType,
    EventStatus,
    EventType,
    HookTrigger,
    TaskStatus,
    Visibility,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker
from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

if TYPE_CHECKING:
    from roomkit.channels.base import Channel
    from roomkit.core.framework import RoomKit
    from roomkit.core.hooks import HookEngine
    from roomkit.store.base import ConversationStore
    from roomkit.tasks.base import TaskRunner
    from roomkit.telemetry.base import TelemetryProvider


_tasks_logger = logging.getLogger("roomkit.tasks")


# ---------------------------------------------------------------------------
# Shared agent execution — single code path for both sync and background
# ---------------------------------------------------------------------------


async def _persist_child_stream(
    kit: RoomKit,
    child_room_id: str,
    sr: Any,
    chain_depth: int,
) -> str:
    """Drain a streaming response into the child room, persisting tool calls.

    Mirrors the main inbound streaming path: text deltas accumulate into
    MESSAGE segments split at tool-call boundaries, and each
    ``ToolCall{Start,End}Marker`` is persisted as a TOOL_CALL_{START,END}
    event — so the child room holds the worker's full trace (what it
    searched/ran, with arguments and results), not just its final answer.
    Returns the full concatenated text (the worker's output for the caller).
    """
    source = EventSource(channel_id=sr.source_channel_id, channel_type=sr.source_channel_type)
    text_parts: list[str] = []
    segment: list[str] = []

    async def _flush_segment() -> None:
        if not segment:
            return
        body = "".join(segment)
        segment.clear()
        await kit.store.add_event_auto_index(
            child_room_id,
            RoomEvent(
                room_id=child_room_id,
                source=source,
                type=EventType.MESSAGE,
                content=TextContent(body=body),
                chain_depth=chain_depth,
            ),
        )

    async for delta in sr.stream:
        if isinstance(delta, str):
            text_parts.append(delta)
            segment.append(delta)
        elif isinstance(delta, ToolCallStartMarker):
            await _flush_segment()
            await kit.store.add_event_auto_index(
                child_room_id,
                RoomEvent(
                    room_id=child_room_id,
                    source=source,
                    type=EventType.TOOL_CALL_START,
                    content=ToolCallContent(
                        tool_name=delta.tool_name,
                        tool_id=delta.tool_id,
                        arguments=delta.arguments,
                        status="pending",
                    ),
                    chain_depth=chain_depth,
                ),
            )
        elif isinstance(delta, ToolCallEndMarker):
            await kit.store.add_event_auto_index(
                child_room_id,
                RoomEvent(
                    room_id=child_room_id,
                    source=source,
                    type=EventType.TOOL_CALL_END,
                    content=ToolCallContent(
                        tool_name=delta.tool_name,
                        tool_id=delta.tool_id,
                        arguments=delta.arguments,
                        result=delta.result,
                        status=delta.status,
                        duration_ms=delta.duration_ms,
                        error=delta.error,
                    ),
                    chain_depth=chain_depth,
                ),
            )
        # ThinkingDeltaMarker (and any other marker): transient, not persisted.
    await _flush_segment()
    return "".join(text_parts)


async def _broadcast_and_collect(
    kit: RoomKit, child_room_id: str, message_body: str
) -> str | None:
    """One delegated turn: store *message_body* as a system message, broadcast it,
    persist the agent's full trace (tool calls + messages), and return its text."""
    room = await kit.get_room(child_room_id)
    bindings = await kit.store.list_bindings(child_room_id)

    msg_event = RoomEvent(
        room_id=child_room_id,
        type=EventType.MESSAGE,
        source=EventSource(channel_id="system", channel_type=ChannelType.SYSTEM),
        content=TextContent(body=message_body),
    )
    msg_event = await kit.store.add_event_auto_index(child_room_id, msg_event)

    # Build context AFTER storing so the agent's memory provider
    # can see the message in recent_events.
    recent = await kit.store.list_events(child_room_id, offset=0, limit=50)
    context = RoomContext(room=room, bindings=bindings, recent_events=recent)

    router = kit._get_router()
    source_binding = ChannelBinding(
        channel_id="system",
        room_id=child_room_id,
        channel_type=ChannelType.SYSTEM,
    )
    result = await router.broadcast(msg_event, source_binding, context)
    child_depth = msg_event.chain_depth + 1

    # Non-streaming: response_events already include the tool-call events —
    # persist them all (not just the final text) so the trace survives.
    for output in result.outputs.values():
        if output.responded and output.response_events:
            final_text: str | None = None
            for resp in output.response_events:
                await kit.store.add_event_auto_index(child_room_id, resp)
                if isinstance(resp.content, TextContent) and resp.content.body:
                    final_text = resp.content.body
            if final_text is not None:
                return final_text

    # Streaming: drain the marker stream, persisting tool calls + text segments.
    for sr in result.streaming_responses:
        text = await _persist_child_stream(kit, child_room_id, sr, child_depth)
        if text:
            return text

    return None


#: A cursor larger than any real event index, so ``before_index`` returns the
#: tail (most recent events) — where a worker's final submit_result call lives.
#: Capped at max int32: the postgres store binds ``before_index`` as int4.
_LATEST_TAIL_CURSOR = 2**31 - 1


async def _scan_for_submitted_result(kit: RoomKit, child_room_id: str) -> dict[str, Any] | None:
    """Find a ``submit_result`` call in the worker's persisted trace.

    The function-calling path captures the payload through the wrapped
    ``tool_handler``; a claude_code worker instead calls the gateway-exposed
    tool, which never reaches that handler — but the call IS persisted as a
    TOOL_CALL event (with an ``mcp__…`` prefix). Scanning the trace tail makes
    the capture delivery-agnostic. Returns the normalized payload, or None.
    """
    from roomkit.orchestration.result import is_submit_result, normalize_result

    events = await kit.store.list_events(
        child_room_id, before_index=_LATEST_TAIL_CURSOR, limit=100
    )
    for ev in reversed(events):
        if ev.type in (EventType.TOOL_CALL_END, EventType.TOOL_CALL_START):
            name = getattr(ev.content, "tool_name", "") or ""
            if is_submit_result(name):
                return normalize_result(getattr(ev.content, "arguments", None) or {})
    return None


async def _run_with_structured_result(
    kit: RoomKit,
    child_room_id: str,
    task_desc: str,
    max_result_retries: int,
) -> str:
    """Run a delegated agent that must hand its work back via the ``submit_result``
    tool. Injects the tool (for function-calling providers), runs the agent, and a
    deterministic completion guard: if the agent ends a turn without calling
    ``submit_result``, it is re-prompted to use the tool (up to *max_result_retries*
    times); if it still hasn't, the orchestration submits a fail on its behalf.

    Capture is delivery-agnostic: a function-calling provider's call is caught by
    the wrapped ``tool_handler``; a claude_code worker calls the gateway-exposed
    tool, which is caught by scanning its persisted trace. Returns the structured
    payload as a JSON string (an orchestration fail when exhausted)."""
    from roomkit.orchestration.result import (
        SUBMIT_RESULT_TOOL,
        SUBMIT_RESULT_TOOL_NAME,
        normalize_result,
        orchestration_fail,
    )

    room = await kit.get_room(child_room_id)
    agent_id = (room.metadata or {}).get("task_agent_id")
    channel = kit.channels.get(agent_id) if agent_id else None
    if channel is None or not hasattr(channel, "_injected_tools"):
        # No injectable agent channel — fall back to plain text collection.
        text = await _broadcast_and_collect(kit, child_room_id, task_desc)
        return text or ""
    role = getattr(channel, "role", None) or getattr(channel, "description", None) or str(agent_id)

    captured: dict[str, Any] = {}
    original_handler = channel.tool_handler

    async def _capture(name: str, arguments: dict[str, Any]) -> str:
        if name == SUBMIT_RESULT_TOOL_NAME:
            captured["payload"] = normalize_result(arguments or {})
            return json.dumps({"status": "received"})
        if original_handler:
            return await original_handler(name, arguments)
        return json.dumps({"error": f"unknown tool {name}"})

    channel._injected_tools.append(SUBMIT_RESULT_TOOL)
    channel.tool_handler = _capture
    try:
        message = task_desc
        last_text = ""
        for _attempt in range(max_result_retries + 1):
            text = await _broadcast_and_collect(kit, child_room_id, message)
            if "payload" in captured:
                return json.dumps(captured["payload"])
            scanned = await _scan_for_submitted_result(kit, child_room_id)
            if scanned is not None:
                return json.dumps(scanned)
            last_text = text or last_text
            message = (
                "You did not submit a result. You MUST now call the `submit_result` "
                "tool with your final structured result. Do NOT reply with plain text "
                "or a question — call submit_result."
            )
        _tasks_logger.warning(
            "Delegated agent %s never called submit_result after %d attempts; failing.",
            agent_id,
            max_result_retries + 1,
        )
        return json.dumps(
            orchestration_fail(role=role, last_output=last_text, attempts=max_result_retries + 1)
        )
    finally:
        with contextlib.suppress(ValueError):
            channel._injected_tools.remove(SUBMIT_RESULT_TOOL)
        channel.tool_handler = original_handler


async def run_agent_in_child_room(
    kit: RoomKit,
    child_room_id: str,
    task_desc: str,
    *,
    require_structured_result: bool = False,
    max_result_retries: int = 3,
) -> str | None:
    """Send a task to a child room and collect the attached agent's response.

    This is the **single code path** for executing a delegated agent, used by
    both ``delegate(wait=True)`` (inline) and the background task runner.

    By default the agent's free-text response is collected and returned. When
    *require_structured_result* is set, the agent must instead hand its work back
    via the ``submit_result`` tool (forced structure + a guaranteed result); the
    returned string is then the JSON-encoded structured payload (see
    :func:`_run_with_structured_result`).

    Either way the agent's full trace (tool calls + messages) is persisted in the
    child room, which records its parent via ``metadata.parent_room_id`` (set at
    creation in :meth:`delegate`) so the parent↔child link is rebuildable.
    """
    if require_structured_result:
        return await _run_with_structured_result(kit, child_room_id, task_desc, max_result_retries)
    return await _broadcast_and_collect(kit, child_room_id, task_desc)


# ---------------------------------------------------------------------------
# Hook metadata builder
# ---------------------------------------------------------------------------


def _delegation_metadata(
    *,
    task_id: str,
    child_room_id: str,
    parent_room_id: str,
    agent_id: str,
    task_input: str | None = None,
    task_status: TaskStatus | str | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build consistent metadata for delegation hooks."""
    meta: dict[str, Any] = {
        "task_id": task_id,
        "child_room_id": child_room_id,
        "parent_room_id": parent_room_id,
        "agent_id": agent_id,
    }
    if task_input is not None:
        meta["task_input"] = task_input
    if task_status is not None:
        meta["task_status"] = task_status
    if duration_ms is not None:
        meta["duration_ms"] = duration_ms
    if error is not None:
        meta["error"] = error
    return meta


# ---------------------------------------------------------------------------
# DelegationMixin
# ---------------------------------------------------------------------------


@runtime_checkable
class DelegationHost(Protocol):
    """Contract: capabilities a host class must provide for DelegationMixin.

    Attributes provided by the host's ``__init__``:
        _store: Conversation persistence backend.
        _channels: Registry of channel-id to :class:`Channel` instances.
        _task_runner: Background task execution backend.
        _hook_engine: Engine for hook execution (via :class:`HelpersMixin`).
        _telemetry: Telemetry / tracing provider (optional — mixin
            falls back to ``NoopTelemetryProvider`` when absent).

    Cross-mixin methods (provided by other mixins in the MRO):
        get_room: From :class:`RoomLifecycleMixin`.
        create_room: From :class:`RoomLifecycleMixin`.
        attach_channel: From :class:`ChannelOpsMixin`.
        deliver: From :class:`DeliverMixin`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _task_runner: TaskRunner
    _hook_engine: HookEngine
    _telemetry: TelemetryProvider | None


class DelegationMixin(HelpersMixin):
    """Task delegation to child rooms — sync and background.

    Host contract: :class:`DelegationHost`.
    """

    _store: ConversationStore
    _channels: dict[str, Channel]
    _task_runner: TaskRunner

    # Cross-mixin methods — attribute annotations avoid MRO shadowing
    get_room: Any  # see DelegationHost
    create_room: Any  # see DelegationHost
    attach_channel: Any  # see DelegationHost
    deliver: Any  # see DelegationHost

    async def delegate(
        self,
        room_id: str,
        agent_id: str,
        task: str,
        *,
        wait: bool = False,
        context: dict[str, Any] | None = None,
        share_channels: list[str] | None = None,
        notify: str | None = None,
        on_complete: Any | None = None,
        require_structured_result: bool = False,
        max_result_retries: int = 3,
    ) -> DelegatedTask:
        """Delegate a task to an agent in a child room.

        Creates a child room linked to *room_id*, attaches the agent and
        any shared channels, then either runs the agent inline or submits
        the task for background execution.

        Args:
            room_id: Parent room ID.
            agent_id: Channel ID of the agent to run the task.
            task: Description of what the agent should do.
            wait: If ``True``, run the agent inline and return a
                pre-completed :class:`DelegatedTask`.  If ``False``
                (default), submit as a background task.
            context: Optional context dict passed to the agent.
            share_channels: Channel IDs from the parent to share.
            notify: Channel ID to update when the task completes
                (system prompt injection). Defaults to *agent_id*.
            on_complete: Optional async callback ``(DelegatedTaskResult) -> None``.

        Returns:
            A :class:`DelegatedTask` handle. When *wait* is ``True``,
            the result is already set.  When ``False``, call ``.wait()``
            to block for the result, or let it run fire-and-forget.

        Raises:
            RoomNotFoundError: If the parent room doesn't exist.
            ChannelNotRegisteredError: If the agent channel isn't registered.
        """
        from uuid import uuid4

        from roomkit.telemetry.base import Attr, SpanKind
        from roomkit.telemetry.context import get_current_span
        from roomkit.telemetry.noop import NoopTelemetryProvider

        # Validate
        await self.get_room(room_id)
        if agent_id not in self._channels:
            raise ChannelNotRegisteredError(f"Agent channel '{agent_id}' not registered")

        child_room_id = f"{room_id}::task-{uuid4().hex[:12]}"
        task_id = f"task-{uuid4().hex[:12]}"

        # Start telemetry span
        telemetry = getattr(self, "_telemetry", None) or NoopTelemetryProvider()
        mode = "inline" if wait else "background"
        span_id = telemetry.start_span(
            SpanKind.DELEGATION,
            f"delegation.{mode}",
            parent_id=get_current_span(),
            room_id=room_id,
            channel_id=agent_id,
            attributes={
                Attr.DELEGATION_TASK_ID: task_id,
                Attr.DELEGATION_WORKER_ID: agent_id,
                Attr.DELEGATION_CHILD_ROOM_ID: child_room_id,
                Attr.DELEGATION_PARENT_ROOM_ID: room_id,
                Attr.DELEGATION_MODE: mode,
            },
        )

        # Create child room — no orchestration so the parent's strategy
        # doesn't leak (e.g. Supervisor attaching itself to the child).
        await self.create_room(
            room_id=child_room_id,
            metadata={
                "parent_room_id": room_id,
                "task_agent_id": agent_id,
                "task_input": task,
                "task_context": context or {},
                "task_status": "pending",
            },
            orchestration=None,
        )

        # Attach agent as intelligence
        await self.attach_channel(
            child_room_id,
            agent_id,
            category=ChannelCategory.INTELLIGENCE,
        )

        # Share channels from parent
        for ch_id in share_channels or []:
            parent_binding = await self._store.get_binding(room_id, ch_id)
            if parent_binding:
                await self.attach_channel(
                    child_room_id,
                    ch_id,
                    category=parent_binding.category,
                    metadata=parent_binding.metadata,
                )

        # Create task handle
        handle = DelegatedTask(
            id=task_id,
            child_room_id=child_room_id,
            parent_room_id=room_id,
            agent_id=agent_id,
            task=task,
        )

        # Fire ON_TASK_DELEGATED hook
        hook_meta = _delegation_metadata(
            task_id=handle.id,
            child_room_id=child_room_id,
            parent_room_id=room_id,
            agent_id=agent_id,
            task_input=task,
        )
        hook_event = RoomEvent(
            room_id=room_id,
            source=EventSource(
                channel_id=agent_id,
                channel_type=ChannelType.AI,
            ),
            content=TextContent(body=f"[Task delegated to {agent_id}] {task}"),
            type=EventType.TASK_DELEGATED,
            status=EventStatus.DELIVERED,
            visibility=Visibility.INTERNAL,
            metadata=hook_meta,
        )
        room_context = await self._build_context(room_id)
        await self._hook_engine.run_async_hooks(
            room_id, HookTrigger.ON_TASK_DELEGATED, hook_event, room_context
        )

        if wait:
            result_handle = await self._run_inline(
                handle,
                context,
                notify,
                on_complete,
                require_structured_result=require_structured_result,
                max_result_retries=max_result_retries,
            )
            telemetry.end_span(
                span_id,
                attributes={
                    Attr.DELEGATION_STATUS: result_handle.status,
                    Attr.DURATION_MS: result_handle.result.duration_ms
                    if result_handle.result
                    else 0,
                },
            )
            return result_handle

        # Background — span ends when task completes (via callback)
        return await self._run_background(handle, context, notify, on_complete, span_id, telemetry)

    async def _run_inline(
        self,
        handle: DelegatedTask,
        context: dict[str, Any] | None,
        notify: str | None,
        on_complete: Any | None,
        *,
        require_structured_result: bool = False,
        max_result_retries: int = 3,
    ) -> DelegatedTask:
        """Run the agent inline and return a pre-completed task."""
        start = time.monotonic()
        handle.status = TaskStatus.IN_PROGRESS
        agent_response: str | None = None
        error: str | None = None

        try:
            agent_response = await run_agent_in_child_room(
                self,  # ty: ignore[invalid-argument-type]
                handle.child_room_id,
                handle.task,
                require_structured_result=require_structured_result,
                max_result_retries=max_result_retries,
            )
        except Exception as exc:
            _tasks_logger.exception("Inline task %s failed: %s", handle.id, exc)
            error = str(exc)

        elapsed = (time.monotonic() - start) * 1000
        status = TaskStatus.COMPLETED if agent_response else TaskStatus.FAILED

        result = DelegatedTaskResult(
            task_id=handle.id,
            child_room_id=handle.child_room_id,
            parent_room_id=handle.parent_room_id,
            agent_id=handle.agent_id,
            status=status,
            output=agent_response,
            error=error,
            duration_ms=elapsed,
            metadata=context or {},
        )

        # Fire completion hooks + callbacks (skip proactive delivery for inline —
        # the caller handles presenting results directly)
        notify_channel = notify or handle.agent_id
        await self._on_delegation_complete(result, notify_channel, deliver=False)
        if on_complete:
            try:
                await on_complete(result)
            except Exception:
                _tasks_logger.exception("on_complete failed for task %s", handle.id)

        handle._set_result(result)
        return handle

    async def _run_background(
        self,
        handle: DelegatedTask,
        context: dict[str, Any] | None,
        notify: str | None,
        on_complete: Any | None,
        span_id: str,
        telemetry: Any,
    ) -> DelegatedTask:
        """Submit the task to the background task runner."""
        from roomkit.telemetry.base import Attr

        notify_channel = notify or handle.agent_id

        async def _on_bg_complete(result: DelegatedTaskResult) -> None:
            telemetry.end_span(
                span_id,
                attributes={
                    Attr.DELEGATION_STATUS: result.status,
                    Attr.DURATION_MS: result.duration_ms,
                },
            )
            await self._on_delegation_complete(result, notify_channel)
            if on_complete:
                await on_complete(result)

        await self._task_runner.submit(
            self,  # ty: ignore[invalid-argument-type]
            handle,
            context=context,
            on_complete=_on_bg_complete,
        )
        return handle

    async def _on_delegation_complete(
        self,
        result: DelegatedTaskResult,
        notify_channel_id: str,
        *,
        deliver: bool = True,
    ) -> None:
        """Handle delegation completion: inject result + fire hook + deliver."""
        # Inject result into the notified agent's system prompt
        max_delegation_prompt = 4000
        binding = await self._store.get_binding(result.parent_room_id, notify_channel_id)
        if binding:
            current_prompt = binding.metadata.get("system_prompt", "")
            appendix = (
                "\n\n--- BACKGROUND TASK COMPLETED ---\n"
                + f"Task ID: {result.task_id}\n"
                + f"Agent: {result.agent_id}\n"
                + f"Status: {result.status}\n"
                + f"Result:\n{result.output or result.error or 'No output'}\n"
                + "--- END ---\n"
            )
            new_prompt = current_prompt + appendix
            if len(new_prompt) > max_delegation_prompt:
                new_prompt = "...\n" + new_prompt[-max_delegation_prompt:]
            updated = binding.model_copy(
                update={"metadata": {**binding.metadata, "system_prompt": new_prompt}}
            )
            await self._store.update_binding(updated)

        # Fire ON_TASK_COMPLETED hook with enriched metadata
        hook_meta = _delegation_metadata(
            task_id=result.task_id,
            child_room_id=result.child_room_id,
            parent_room_id=result.parent_room_id,
            agent_id=result.agent_id,
            task_status=result.status,
            duration_ms=result.duration_ms,
            error=result.error,
        )
        hook_event = RoomEvent(
            room_id=result.parent_room_id,
            source=EventSource(
                channel_id=result.agent_id,
                channel_type=ChannelType.AI,
            ),
            content=TextContent(body=result.output or result.error or ""),
            type=EventType.TASK_COMPLETED,
            status=EventStatus.DELIVERED,
            visibility=Visibility.INTERNAL,
            metadata=hook_meta,
        )
        try:
            room_context = await self._build_context(result.parent_room_id)
            await self._hook_engine.run_async_hooks(
                result.parent_room_id, HookTrigger.ON_TASK_COMPLETED, hook_event, room_context
            )
        except Exception:
            _tasks_logger.exception(
                "Failed to fire ON_TASK_COMPLETED hook for task %s", result.task_id
            )

        # Deliver result via kit.deliver() (background path only)
        if not deliver:
            return
        content = result.output or result.error
        if content:
            try:
                prompt = (
                    f"[Background task from {result.agent_id} completed. "
                    f"Share the result with the user.]"
                )
                await self.deliver(
                    result.parent_room_id,
                    prompt,
                    channel_id=notify_channel_id,
                )
            except Exception:
                _tasks_logger.exception("Delivery failed for task %s", result.task_id)
