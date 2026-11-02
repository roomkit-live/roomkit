"""State and text handling for one streamed AI generation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, Protocol

from roomkit.channels._ai_coalescers import (
    _TextCoalescer,
    _ThinkingCoalescer,
    _ToolCallDeltaCoalescer,
)
from roomkit.channels._ai_resilience import _StreamRetryBoundary
from roomkit.channels._ai_stream_external_tools import _ExternalStreamTools
from roomkit.models.streaming import StreamDelta, ThinkingDeltaMarker, ToolCallEndMarker
from roomkit.providers.ai.base import (
    AIToolResultPart,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCall,
    StreamToolCallDelta,
)
from roomkit.providers.ai.round_parts import RoundTranscript
from roomkit.providers.utils import _aclose_stream
from roomkit.realtime.base import EphemeralEventType
from roomkit.tools._outcome import OutcomeKind, ToolOutcome
from roomkit.tools.result import as_tool_result


@dataclass
class _StreamRoundState:
    """A generation's raw transcript and the fragments actually delivered."""

    # The reasoning text as it streamed, for the live windows and the turn's
    # report; the blocks a round replays live in the transcript.
    thinking_chunks: list[str] = field(default_factory=list)
    thinking_started: bool = False
    thinking_published: int = 0
    transcript: RoundTranscript = field(default_factory=RoundTranscript)
    reported: list[str] = field(default_factory=list)
    # The calls the loop serves; the provider's own are reported inline, then
    # replayed with their results beside the loop's (RFC §9.3).
    tool_calls: list[StreamToolCall] = field(default_factory=list)
    provider_calls: list[StreamToolCall] = field(default_factory=list)
    provider_results: list[AIToolResultPart] = field(default_factory=list)
    finish_reason: str | None = None
    refusal: str | None = None  # the provider's reason, when it refused
    cancelled: bool = False
    # Whether the provider handed any event, a call the loop serves or text
    # a response schema holds included: its answer began, so it took the
    # request (RFC §6.4).
    received: bool = False

    def note_received(self) -> None:
        """The provider handed an event."""
        self.received = True

    @property
    def text(self) -> str:
        return self.transcript.text

    @property
    def thinking(self) -> str:
        return "".join(self.thinking_chunks)


class _PrefixDeduplicator:
    """Withhold a replayed prefix until new text or a mismatch settles it.

    A generation ending inside the prefix still gets its buffered text back:
    it may be the entire final answer. Only a prefix followed by new text is
    suppressed. The consumer decides when returned fragments are delivered.
    """

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._active = bool(prefix)
        self._offset = 0
        self._buffer: list[str] = []

    def add(self, text: str) -> list[str]:
        """Return the fragments ready to deliver after this delta."""
        if not self._active:
            return [text]
        end = self._offset + len(text)
        if end <= len(self._prefix):
            if self._prefix[self._offset : end] == text:
                self._offset = end
                self._buffer.append(text)
                return []
            result = [*self._buffer, text]
        else:
            tail = self._prefix[self._offset :]
            if text[: len(tail)] == tail:
                remaining = text[len(tail) :]
                result = [remaining] if remaining else []
            else:
                result = [*self._buffer, text]
        self._active = False
        self._buffer.clear()
        return result

    def finish(self) -> list[str]:
        """Deliver a partial or exact prefix when no new text followed it."""
        result = self._buffer
        self._buffer = []
        return result


class _ThinkingWindowCloser(Protocol):
    async def __call__(
        self,
        coalescer: _ThinkingCoalescer,
        room_id: str,
        thinking_chunks: list[str],
        round_idx: int,
        *,
        published: int,
    ) -> int: ...


@dataclass
class _StreamRound:
    """Consume one generation and own its stream and observable windows.

    The turn chooses whether another round can run. This component only
    projects provider events and keeps the raw and delivered transcripts.
    A retry boundary starts a new composition attempt within the same round.
    """

    index: int
    room_id: str | None
    cancel_event: asyncio.Event
    thinking_coalescer: _ThinkingCoalescer
    new_composition: Callable[[], _ToolCallDeltaCoalescer]
    publish_thinking: Callable[[EphemeralEventType, str, str, int], Awaitable[None]]
    close_thinking: _ThinkingWindowCloser
    record_usage: Callable[[dict[str, Any]], None]
    text_coalescer: _TextCoalescer | None = None
    prefix: str = ""
    external_tools: _ExternalStreamTools | None = None
    # A call the loop serves, its intent taken out of its arguments, when
    # the agent asks its model for one (RFC §6.4).
    take_intent: Callable[[StreamToolCall], StreamToolCall] | None = None
    state: _StreamRoundState = field(default_factory=_StreamRoundState, init=False)
    _composition: _ToolCallDeltaCoalescer = field(init=False)
    _dedup: _PrefixDeduplicator = field(init=False)

    def __post_init__(self) -> None:
        self._composition = self.new_composition()
        self._dedup = _PrefixDeduplicator(self.prefix)

    async def _end_thinking(self) -> None:
        state = self.state
        if not (state.thinking_started and state.thinking_chunks and self.room_id):
            return
        # Disarm before awaiting a publish: cleanup can itself be cancelled.
        state.thinking_started = False
        state.thinking_published = await self.close_thinking(
            self.thinking_coalescer,
            self.room_id,
            state.thinking_chunks,
            self.index,
            published=state.thinking_published,
        )

    async def _add_thinking(self, event: StreamThinkingDelta) -> None:
        state = self.state
        state.transcript.add_thinking(event)
        if not event.thinking:
            return
        if not state.thinking_started and self.room_id:
            state.thinking_started = True
            await self.publish_thinking(
                EphemeralEventType.THINKING_START, self.room_id, "", self.index
            )
        state.thinking_chunks.append(event.thinking)
        await self.thinking_coalescer.add(event.thinking)

    async def _add_text(self, event: StreamTextDelta) -> list[str]:
        """Record a text delta; the text it releases for delivery."""
        await self._end_thinking()
        self.state.transcript.add_text(event.text)
        released = list(self._dedup.add(event.text))
        self.state.reported.extend(released)
        await self._publish_text(released)
        return released

    async def _publish_text(self, texts: list[str]) -> None:
        if self.text_coalescer is None or not self.room_id:
            return
        for text in texts:
            await self.text_coalescer.add(text)

    async def _flush_text(self) -> None:
        if self.text_coalescer is not None and self.room_id:
            await self.text_coalescer.flush()

    async def close(self) -> None:
        """Close only opened windows, including when another close is cancelled."""
        try:
            await self._end_thinking()
            await self._flush_text()
        finally:
            if self.room_id:
                await self._composition.close()

    async def _take_call(self, call: StreamToolCall) -> AsyncGenerator[StreamDelta, None]:
        """Keep a call for the loop to serve, or report the provider's own
        inline and keep it with its result for the next round's transcript."""
        self.state.transcript.add_call(call.id)
        external = self.external_tools
        if external is None or not external.takes(call):
            # Only a call the loop serves itself: the provider's own were
            # never shown the intent's property (RFC §6.4).
            taken = call if self.take_intent is None else self.take_intent(call)
            self.state.tool_calls.append(taken)
            return
        async with aclosing(external.stream_call(call, self.index)) as deltas:
            async for delta in deltas:
                if isinstance(delta, ToolCallEndMarker):
                    self.state.provider_calls.append(
                        call.model_copy(update={"arguments": delta.arguments})
                    )
                    self.state.provider_results.append(provider_result(call, delta))
                yield delta

    async def stream(
        self, source: AsyncIterator[StreamEvent | _StreamRetryBoundary]
    ) -> AsyncGenerator[StreamDelta, None]:
        """Yield on demand; the direct consumer must explicitly close this iterator."""
        state = self.state
        try:
            async for event in source:
                if self.cancel_event.is_set():
                    state.cancelled = True
                    await self.close()
                    return
                if isinstance(event, _StreamRetryBoundary):
                    if self.room_id:
                        await self._composition.close()
                        self._composition = self.new_composition()
                elif isinstance(event, StreamThinkingDelta):
                    await self._add_thinking(event)
                    if event.thinking:
                        yield ThinkingDeltaMarker(thinking=event.thinking)
                elif isinstance(event, StreamTextDelta):
                    for text in await self._add_text(event):
                        yield text
                elif isinstance(event, StreamToolCallDelta):
                    await self._end_thinking()
                    await self._flush_text()
                    if self.room_id:
                        await self._composition.add(
                            event.index, event.id, event.name, len(event.arguments_delta)
                        )
                elif isinstance(event, StreamToolCall):
                    await self._flush_text()
                    async with aclosing(self._take_call(event)) as deltas:
                        async for delta in deltas:
                            yield delta
                elif isinstance(event, StreamDone):
                    state.finish_reason = event.finish_reason
                    state.refusal = event.metadata.get("refusal")
                    if event.usage:
                        self.record_usage(event.usage)

            finished = list(self._dedup.finish())
            state.reported.extend(finished)
            await self._publish_text(finished)
            for text in finished:
                yield text
            await self.close()
        finally:
            try:
                await _aclose_stream(source)
            finally:
                await self.close()


def provider_result(call: StreamToolCall, end: ToolCallEndMarker) -> AIToolResultPart:
    """The part the model reads of a call the provider served: its result, as
    the end row states it."""
    outcome = end.outcome or ("failed" if end.status == "failed" else "served")
    return ToolOutcome(OutcomeKind(outcome), as_tool_result(end.result or "")).as_part(
        call.id, call.name
    )
