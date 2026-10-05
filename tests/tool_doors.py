"""One tool call through each door that serves one, and what was reported.

The doors: an AIChannel turn (streaming or not), a realtime session's
provider call, a recovered call and a backend's raw gate call, an agent
reasoning backend's loop, and a conference. :func:`run_door` returns what
ON_TOOL_CALL's SYNC chain, its ASYNC observers, BEFORE_TOOL_USE and the
``tool_call`` framework event saw, the stored TOOL_CALL_END rows, and what
the model read. *channel* passes options every door's channel takes under
the same name (``human_input_handler``, ``tool_timeout_seconds``,
``tool_timeouts``): an AIChannel's, a realtime channel's, or a conference's
realtime configuration.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from roomkit import (
    ChannelCategory,
    ConferenceRealtimeConfig,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomKit,
    TextContent,
    ToolCallEvent,
)
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.enums import EventType
from roomkit.models.event import ToolCallContent
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import AgentReasoningBackend
from tests.conference.test_conference_realtime import ROOM, realtime_kit
from tests.test_framework import SimpleChannel

SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}}
TOOL_DICT = {"name": "lookup", "description": "Look up", "parameters": SCHEMA}
TOOL = AITool(name="lookup", description="Look up", parameters=SCHEMA)


def _tool_dict(schema: dict[str, Any]) -> dict[str, Any]:
    return {**TOOL_DICT, "parameters": schema}


DOORS = (
    "text-stream",
    "text-nostream",
    "rt-provider",
    "rt-recovered",
    "rt-backend-gate",
    "rt-agent-backend",
    "conference",
)

Hook = Callable[[Any, Any], Awaitable[HookResult]]


@dataclass(frozen=True)
class Hooks:
    """The kit's hooks around the call: ON_TOOL_CALL's SYNC *judge* (none
    without *sync_hook*), BEFORE_TOOL_USE's *before*, and any *setup* of the
    kit before the call."""

    judge: Hook | None = None
    before: Hook | None = None
    sync_hook: bool = True
    setup: Callable[[RoomKit], None] | None = None


@dataclass
class Seen:
    """What one call left: its reports, what the hooks saw, its framework
    events and rows, and what the model read."""

    reports: list[ToolCallEvent] = field(default_factory=list)
    sync: list[ToolCallEvent] = field(default_factory=list)
    before: list[ToolCallEvent] = field(default_factory=list)
    framework: list[dict[str, Any]] = field(default_factory=list)
    rows: list[ToolCallContent] = field(default_factory=list)
    model_read: Any = None


def _reported(seen: Seen) -> bool:
    """Whether the call was reported: to its observers, or by its framework
    event alone when they could not hear it."""
    return bool(seen.reports or seen.framework)


async def _until(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate() and loop.time() < deadline:
        await asyncio.sleep(0.01)


def _install(kit: RoomKit, seen: Seen, hooks: Hooks) -> None:
    judge, before = hooks.judge, hooks.before
    if hooks.sync_hook:

        @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="judge")
        async def _judge(event: ToolCallEvent, ctx: Any) -> HookResult:
            seen.sync.append(event)
            return await judge(event, ctx) if judge is not None else HookResult.allow()

    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.ASYNC, name="audit")
    async def _audit(event: ToolCallEvent, ctx: Any) -> None:
        seen.reports.append(event)

    if before is not None:

        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="gate")
        async def _gate(event: Any, ctx: Any) -> HookResult:
            seen.before.append(event)
            return await before(event, ctx)

    @kit.on("tool_call")
    async def _framework(event: Any) -> None:
        seen.framework.append(dict(event.data))

    if hooks.setup is not None:
        hooks.setup(kit)


async def _text_door(
    handler: Any,
    streaming: bool,
    call: AIToolCall,
    hooks: Hooks,
    schema: dict[str, Any],
    channel: dict[str, Any],
) -> Seen:
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="done"),
        ],
        streaming=streaming,
    )
    tool = TOOL.model_copy(update={"parameters": schema})
    ai = AIChannel("ai1", provider=provider, tool_handler=handler, tools=[tool], **channel)
    kit = RoomKit()
    kit.register_channel(SimpleChannel("sms1"))
    kit.register_channel(ai)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "sms1")
    await kit.attach_channel("r1", "ai1", category=ChannelCategory.INTELLIGENCE)
    seen = Seen()
    _install(kit, seen, hooks)
    message = InboundMessage(channel_id="sms1", sender_id="u1", content=TextContent(body="go"))
    await kit.process_inbound(message)
    await _until(lambda: _reported(seen))
    if len(provider.calls) > 1:
        parts = [p for m in provider.calls[1].messages if m.role == "tool" for p in m.content]
        seen.model_read = parts[0].result if parts else None
    for event in await kit.store.list_events("r1"):
        if event.type == EventType.TOOL_CALL_END and isinstance(event.content, ToolCallContent):
            seen.rows.append(event.content)
    await kit.close()
    return seen


async def _realtime(
    handler: Any, schema: dict[str, Any], **kwargs: Any
) -> tuple[RoomKit, Any, Any, Any]:
    provider = MockRealtimeProvider(full_duplex="reasoning_backend" in kwargs)
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=handler,
        tools=[_tool_dict(schema)],
        **kwargs,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u1", "ws")
    return kit, channel, provider, session


async def _realtime_door(
    door: str,
    handler: Any,
    call: AIToolCall,
    hooks: Hooks,
    schema: dict[str, Any],
    options: dict[str, Any],
) -> Seen:
    kit, channel, provider, session = await _realtime(handler, schema, **options)
    seen = Seen()
    _install(kit, seen, hooks)
    arguments = dict(call.arguments)
    if door == "rt-provider":
        await provider.simulate_tool_call(session, call.id, call.name, arguments)
        await _until(lambda: bool(provider.tool_results) and _reported(seen))
        seen.model_read = provider.tool_results[0][2] if provider.tool_results else None
    elif door == "rt-recovered":
        await serve_recovered(channel, session, call.name, arguments)
        await _until(lambda: _reported(seen))
        seen.model_read = provider.injected_texts[-1][1] if provider.injected_texts else None
    else:
        done = await channel._execute_backend_tool_call(session, "d1", call.name, arguments)
        await _until(lambda: _reported(seen))
        seen.model_read = (done.text, done.is_error, done.refused)
    await kit.close()
    return seen


async def _agent_backend_door(
    handler: Any, call: AIToolCall, hooks: Hooks, schema: dict[str, Any], channel: dict[str, Any]
) -> Seen:
    ai_provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", finish_reason="tool_calls", tool_calls=[call]),
            AIResponse(content="done"),
        ]
    )
    backend = AgentReasoningBackend(Agent("reasoner", provider=ai_provider))
    kit, _, provider, session = await _realtime(
        handler, schema, reasoning_backend=backend, **channel
    )
    seen = Seen()
    _install(kit, seen, hooks)
    await provider.simulate_delegation(session, "d1", "integrator")
    await _until(lambda: _reported(seen))
    # A call the provider served may end the loop without a round after it.
    await _until(lambda: len(ai_provider.calls) > 1, timeout=0.2)
    if len(ai_provider.calls) > 1:
        parts = [p for m in ai_provider.calls[1].messages if m.role == "tool" for p in m.content]
        seen.model_read = parts[0].result if parts else None
    await kit.close()
    return seen


async def _conference_door(
    handler: Any, call: AIToolCall, hooks: Hooks, schema: dict[str, Any], channel: dict[str, Any]
) -> Seen:
    provider = MockRealtimeProvider()

    async def served(room_id: str, name: str, arguments: dict[str, Any]) -> Any:
        return await handler(name, arguments)

    config = ConferenceRealtimeConfig(
        provider=provider, tools=[_tool_dict(schema)], tool_handler=served, **channel
    )
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    seen = Seen()
    _install(kit, seen, hooks)
    session = await channel._realtime.ensure_session(ROOM)
    await provider.simulate_tool_call(session, call.id, call.name, dict(call.arguments))
    await _until(lambda: bool(provider.tool_results) and _reported(seen))
    seen.model_read = provider.tool_results[0][2] if provider.tool_results else None
    await kit.close()
    return seen


async def run_door(
    door: str,
    handler: Any,
    *,
    arguments: dict[str, Any] | None = None,
    hooks: Hooks | None = None,
    call: AIToolCall | None = None,
    schema: dict[str, Any] = SCHEMA,
    channel: dict[str, Any] | None = None,
) -> Seen:
    """Run one ``lookup`` call with *arguments* through *door*, served by
    *handler* ``(name, arguments)``, around the kit's *hooks*; or the model's
    *call* as it stands, when given; the tool declared with *schema*, on a
    channel given the options in *channel*."""
    call = call or AIToolCall(id="c1", name="lookup", arguments=dict(arguments or {}))
    hooks = hooks or Hooks()
    options = dict(channel or {})
    if door.startswith("text-"):
        return await _text_door(handler, door == "text-stream", call, hooks, schema, options)
    if door == "rt-agent-backend":
        return await _agent_backend_door(handler, call, hooks, schema, options)
    if door == "conference":
        return await _conference_door(handler, call, hooks, schema, options)
    return await _realtime_door(door, handler, call, hooks, schema, options)


async def serve_recovered(
    channel: RealtimeVoiceChannel, session: Any, name: str, arguments: dict[str, Any]
) -> None:
    """Serve a call recovered from speech in this task, as the channel's
    recovery does in a task of its own."""
    call = channel._book_recovered_call(session, name, arguments)
    call.task = asyncio.current_task()
    await channel._serve_recovered_call(call)
