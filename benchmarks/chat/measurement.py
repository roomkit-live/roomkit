"""Measure provider calls without changing their request or response semantics."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from roomkit.providers.ai.base import (
    AIContext,
    AIProvider,
    AIResponse,
    ModelInfo,
    ProviderError,
    StreamDone,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
)


@dataclass
class Call:
    start: float
    end: float = 0
    first_text: float | None = None
    streaming: bool = False
    messages: int = 0
    prompt_chars: int = 0
    tools: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    error: str | None = None
    injected: bool = False
    thinking_chars: int = 0
    waits: list[tuple[float, float]] = field(default_factory=list)


class MeasuredProvider(AIProvider):
    """Observe real calls; optional faults are explicitly tagged in the report.

    Ownership of the underlying client remains with the suite so independent
    rooms and samples reuse its connection pool. Context snapshots stay local
    to assertions and are not included in exported measurements.
    """

    def __init__(self, inner: AIProvider, *, streaming: bool = True) -> None:
        self.inner = inner
        self.streaming = streaming
        self.calls: list[Call] = []
        self.contexts: list[AIContext] = []
        self.fail_next = 0

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    @property
    def supports_streaming(self) -> bool:
        return self.streaming and self.inner.supports_streaming

    @property
    def supports_structured_streaming(self) -> bool:
        return self.supports_streaming and self.inner.supports_structured_streaming

    @property
    def supports_vision(self) -> bool:
        return self.inner.supports_vision

    def catalog_entry(self) -> ModelInfo | None:
        return self.inner.catalog_entry()

    def _begin(self, context: AIContext, streaming: bool) -> Call:
        self.contexts.append(context.model_copy(deep=True))
        call = Call(
            start=time.perf_counter(),
            streaming=streaming,
            messages=len(context.messages),
            prompt_chars=len(context.system_prompt or "")
            + sum(len(str(m.content)) for m in context.messages),
            tools=[t.name for t in context.tools],
        )
        self.calls.append(call)
        return call

    def _fault(self, call: Call) -> None:
        if self.fail_next:
            self.fail_next -= 1
            call.injected = True
            raise ProviderError(
                "Benchmark injected transient failure",
                provider=self.name,
                retryable=True,
                status_code=503,
            )

    async def generate(self, context: AIContext) -> AIResponse:
        call = self._begin(context, False)
        try:
            self._fault(call)
            response = await self.inner.generate(context)
            call.usage = dict(response.usage)
            call.finish_reason = response.finish_reason
            call.thinking_chars = len(response.thinking or "")
            return response
        except BaseException as exc:
            call.error = type(exc).__name__
            raise
        finally:
            call.end = time.perf_counter()
            call.waits.append((call.start, call.end))

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        call = self._begin(context, True)
        try:
            self._fault(call)
            stream = self.inner.generate_structured_stream(context)
            try:
                while True:
                    waiting_at = time.perf_counter()
                    try:
                        event = await anext(stream)
                    except StopAsyncIteration:
                        break
                    finally:
                        call.waits.append((waiting_at, time.perf_counter()))
                    if (
                        isinstance(event, StreamTextDelta)
                        and event.text
                        and call.first_text is None
                    ):
                        call.first_text = time.perf_counter()
                    if isinstance(event, StreamDone):
                        call.usage = dict(event.usage)
                        call.finish_reason = event.finish_reason
                    if isinstance(event, StreamThinkingDelta):
                        call.thinking_chars += len(event.thinking)
                    yield event
            finally:
                await stream.aclose()
        except BaseException as exc:
            call.error = type(exc).__name__
            raise
        finally:
            call.end = time.perf_counter()


def covered_seconds(intervals: list[tuple[float, float]]) -> float:
    """Union duration; overlapping provider/tool calls are never double-counted."""
    end = 0.0
    covered = 0.0
    for left, right in sorted(intervals):
        if right > left:
            covered += max(0.0, right - max(left, end))
            end = max(end, right)
    return covered
