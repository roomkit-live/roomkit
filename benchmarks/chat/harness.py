"""Full inbound pipeline, AI channel, event bus, store and WebSocket callbacks."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.chat.measurement import MeasuredProvider, covered_seconds
from roomkit import (
    AIChannel,
    ChannelOutput,
    HookExecution,
    HookResult,
    HookTrigger,
    InboundMessage,
    RoomContext,
    RoomEvent,
    RoomKit,
    SQLiteStore,
    TextContent,
    ToolCallEvent,
    WebSocketChannel,
)
from roomkit.channels.websocket import StreamChunk, StreamEnd, StreamError, StreamMessage
from roomkit.models.channel import ChannelBinding
from roomkit.models.streaming import LoopEndMarker
from roomkit.models.tool_call import AIResponseEvent
from roomkit.providers.ai.base import AIProvider
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType


class ObservedAIChannel(AIChannel):
    """Read the public terminal stream marker, keyed by room."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ends: list[dict[str, Any]] = []

    async def on_event(
        self, event: RoomEvent, binding: ChannelBinding, context: RoomContext
    ) -> ChannelOutput:
        result = await super().on_event(event, binding, context)
        if result.response_stream is None:
            return result

        async def observe() -> AsyncIterator[Any]:
            async for item in result.response_stream:
                if isinstance(item, LoopEndMarker):
                    self.ends.append({"room": event.room_id, **asdict(item)})
                yield item

        return result.model_copy(update={"response_stream": observe()})


class Harness:
    def __init__(
        self,
        inner: AIProvider,
        *,
        streaming: bool = True,
        sqlite: Path | None = None,
        **ai_options: Any,
    ) -> None:
        self.provider = MeasuredProvider(inner, streaming=streaming)
        self.kit = RoomKit(store=SQLiteStore(sqlite) if sqlite else None)
        self.ws = WebSocketChannel("user")
        self.handler: Callable[[str, dict[str, Any]], Awaitable[str]] | None = None

        async def dispatch(name: str, arguments: dict[str, Any]) -> str:
            if self.handler is None:
                raise RuntimeError("Scenario did not configure its tool handler")
            return await self.handler(name, arguments)

        self.ai = ObservedAIChannel(
            "assistant",
            self.provider,
            **{
                "temperature": 0.0,
                "max_tool_rounds": 12,
                "tool_loop_timeout_seconds": 45,
                "system_prompt": (
                    "Follow the user's instructions precisely. Keep answers concise. "
                    "Never invent tool results."
                ),
                **({"tool_handler": dispatch} if ai_options.get("tools") else {}),
                **ai_options,
            },
        )
        self.kit.register_channel(self.ws)
        self.kit.register_channel(self.ai)
        self.delivered: list[RoomEvent] = []
        self.streams: list[tuple[float, StreamMessage]] = []
        self.ephemeral: list[EphemeralEvent] = []
        self.responses: list[AIResponseEvent] = []
        self.tool_events: list[ToolCallEvent] = []
        self.executions: list[dict[str, Any]] = []
        self.checks: dict[str, bool] = {}
        self.details: dict[str, Any] = {}
        self.turns: list[dict[str, Any]] = []
        self.start = self.end = 0.0

        @self.kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC)
        async def on_tool(event: ToolCallEvent, context: RoomContext) -> HookResult:
            self.tool_events.append(event)
            return HookResult.allow()

        @self.kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
        async def on_response(event: AIResponseEvent, context: RoomContext) -> HookResult:
            self.responses.append(event)
            return HookResult.allow()

    async def add_room(self, room_id: str) -> None:
        await self.kit.create_room(room_id=room_id)
        await self.kit.attach_channel(room_id, "user")
        await self.kit.attach_channel(room_id, "assistant")

        async def deliver(connection: str, event: RoomEvent) -> None:
            self.check("socket_room_isolation", event.room_id == room_id)
            self.delivered.append(event)

        async def stream(connection: str, message: StreamMessage) -> None:
            self.check("socket_room_isolation", message.room_id == room_id)
            self.streams.append((time.perf_counter(), message))

        async def ephemeral(event: EphemeralEvent) -> None:
            self.ephemeral.append(event)

        self.ws.register_connection(room_id, deliver, room_id=room_id, stream_send_fn=stream)
        await self.kit.subscribe_room(room_id, ephemeral)

    def check(self, name: str, passed: bool) -> None:
        self.checks[name] = self.checks.get(name, True) and bool(passed)

    async def ask(
        self,
        text: str,
        *,
        room: str = "main",
        actor: str = "alice",
        idempotency_key: str | None = None,
    ) -> Any:
        start = time.perf_counter()
        result = await self.kit.process_inbound(
            InboundMessage(
                channel_id="user",
                sender_id=actor,
                content=TextContent(body=text),
                idempotency_key=idempotency_key,
            ),
            room_id=room,
        )
        end = time.perf_counter()
        chunks = [
            t
            for t, msg in self.streams
            if t >= start and msg.room_id == room and isinstance(msg, StreamChunk) and msg.delta
        ]
        self.turns.append(
            {
                "room": room,
                "actor": actor,
                "elapsed_ms": (end - start) * 1000,
                "first_text_ms": (min(chunks) - start) * 1000 if chunks else None,
                "blocked": result.blocked,
                "error": type(result.error).__name__ if result.error else None,
            }
        )
        return result

    def answer(self, room: str = "main") -> str:
        # Segmented streams persist/deliver MESSAGEs; their terminal StreamEnd
        # intentionally has an empty body. Deduplicate by persisted event id.
        events = {
            event.id: event
            for event in self.delivered
            if event.room_id == room and event.source.channel_id == "assistant"
        }
        for _, msg in self.streams:
            if isinstance(msg, StreamEnd) and msg.room_id == room:
                events.setdefault(msg.event.id, msg.event)
        return "\n".join(
            event.content.body
            for event in events.values()
            if isinstance(event.content, TextContent) and event.content.body
        )

    async def validate(self) -> None:
        # Subscribers dispatch asynchronously. This drain is outside measured latency.
        await asyncio.sleep(0.02)
        rooms = {t["room"] for t in self.turns}
        for room in rooms:
            events = await self.kit.store.list_events(room)
            indices = [event.index for event in events]
            self.check("event_indices_sequential", indices == list(range(len(indices))))
        self.check(
            "no_stream_errors", not any(isinstance(msg, StreamError) for _, msg in self.streams)
        )
        self.check(
            "no_thinking_tags_in_visible_text",
            "<think>" not in "\n".join(self.answer(room) for room in rooms),
        )
        starts: list[tuple[str, str]] = []
        ends: list[tuple[str, str]] = []
        composing: set[str] = set()
        for event in self.ephemeral:
            calls = event.data.get("tool_calls", [])
            if event.type == EphemeralEventType.TOOL_CALL_START:
                starts.extend((event.room_id, call["id"]) for call in calls)
            elif event.type == EphemeralEventType.TOOL_CALL_END:
                ends.extend((event.room_id, call["id"]) for call in calls)
                self.check(
                    "tool_result_preview_bounded",
                    all(len(str(call.get("result", ""))) <= 503 for call in calls),
                )
            elif event.type == EphemeralEventType.TOOL_CALL_DELTA:
                self.check(
                    "composition_hides_arguments",
                    all(
                        "arguments" not in call and "arguments_delta" not in call for call in calls
                    ),
                )
                if calls:
                    composing.add(event.room_id)
                else:
                    composing.discard(event.room_id)
        if starts or ends:
            self.check("tool_events_paired", sorted(starts) == sorted(ends))
            self.check(
                "tool_counts_in_response_hook",
                sum(response.tool_calls_count for response in self.responses) == len(starts),
            )
        self.check("composition_terminated", not composing)
        for response in self.responses:
            self.check(
                "response_transcript_consistent",
                response.response_content == "\n\n".join(response.segments),
            )

    def measurements(self) -> dict[str, Any]:
        calls = self.provider.calls
        intervals = [interval for call in calls for interval in call.waits]
        tool_intervals = [(e["start"], e["end"]) for e in self.executions]
        usage: dict[str, int] = {}
        for call in calls:
            for key, value in call.usage.items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
        firsts = [t["first_text_ms"] for t in self.turns if t["first_text_ms"] is not None]
        measured = self.end - self.start
        return {
            "elapsed_ms": measured * 1000,
            "first_text_ms": (
                firsts[0]
                if len(self.turns) == 1 and firsts
                else (calls[0].first_text - self.start) * 1000
                if not self.turns and calls and calls[0].first_text
                else None
            ),
            "provider_wall_ms": covered_seconds(intervals) * 1000,
            "provider_sum_ms": sum(right - left for left, right in intervals) * 1000,
            "residual_ms": max(0, measured - covered_seconds(intervals + tool_intervals)) * 1000,
            "provider_calls": len(calls),
            "tool_calls": len(self.tool_events),
            "usage": usage,
            "turns": self.turns,
            "calls": [
                {
                    **asdict(c),
                    "start": (c.start - self.start) * 1000,
                    "end": (c.end - self.start) * 1000,
                    "first_text": (c.first_text - c.start) * 1000 if c.first_text else None,
                    "waits": [
                        ((left - self.start) * 1000, (right - self.start) * 1000)
                        for left, right in c.waits
                    ],
                }
                for c in calls
            ],
            "loop_ends": self.ai.ends,
            "ephemeral_counts": {
                kind: sum(e.type.value == kind for e in self.ephemeral)
                for kind in sorted({e.type.value for e in self.ephemeral})
            },
            "response_hook_count": len(self.responses),
            "response_usage": [response.usage for response in self.responses],
            "response_tool_counts": [response.tool_calls_count for response in self.responses],
            "checks": self.checks,
            "details": self.details,
        }

    async def close(self) -> None:
        await self.kit.close()
