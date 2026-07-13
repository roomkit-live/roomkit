"""Execute one delegated turn in a child room and capture its result.

The single code path for running a delegated agent — used by both
``delegate(wait=True)`` (inline) and the background task runner via
:meth:`DelegationMixin`. Persists the worker's full trace (tool calls +
messages) into the child room and returns its output, either as free text or
as a structured ``submit_result`` payload.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType, EventStatus, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.streaming import ToolCallEndMarker, ToolCallStartMarker

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit


_tasks_logger = logging.getLogger("roomkit.tasks")


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
        await kit.store.commit_event(
            child_room_id,
            RoomEvent(
                room_id=child_room_id,
                source=source,
                type=EventType.MESSAGE,
                content=TextContent(body=body),
                status=EventStatus.DELIVERED,
                chain_depth=chain_depth,
            ),
        )

    async for delta in sr.stream:
        if isinstance(delta, str):
            text_parts.append(delta)
            segment.append(delta)
        elif isinstance(delta, ToolCallStartMarker):
            await _flush_segment()
            await kit.store.commit_event(
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
                    status=EventStatus.DELIVERED,
                    chain_depth=chain_depth,
                ),
            )
        elif isinstance(delta, ToolCallEndMarker):
            await kit.store.commit_event(
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
                    status=EventStatus.DELIVERED,
                    chain_depth=chain_depth,
                ),
            )
        # ThinkingDeltaMarker (and any other marker): transient, not persisted.
    await _flush_segment()
    return "".join(text_parts)


async def _persist_response_events(
    kit: RoomKit, child_room_id: str, response_events: list[RoomEvent]
) -> str | None:
    """Persist a non-streaming response's events (tool calls + messages) and
    return the last message text, so the child room keeps the full trace."""
    final_text: str | None = None
    for resp in response_events:
        await kit.store.commit_event(
            child_room_id, resp.model_copy(update={"status": EventStatus.DELIVERED})
        )
        if isinstance(resp.content, TextContent) and resp.content.body:
            final_text = resp.content.body
    return final_text


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
        status=EventStatus.DELIVERED,
    )
    msg_event = await kit.store.commit_event(child_room_id, msg_event)

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
        if not (output.responded and output.response_events):
            continue
        final_text = await _persist_response_events(kit, child_room_id, output.response_events)
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


@runtime_checkable
class _InjectableToolChannel(Protocol):
    """A channel that can host an injected tool and a swappable tool handler.

    ``_run_with_structured_result`` matches by structure, not by class, so
    duck-typed agent channels (and test doubles) qualify as long as they expose
    both members.
    """

    _injected_tools: list[Any]
    tool_handler: Any


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
    if not isinstance(channel, _InjectableToolChannel):
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
