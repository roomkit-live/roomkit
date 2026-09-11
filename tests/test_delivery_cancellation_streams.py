"""Every layer must join the provider stream's asynchronous finalizer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from roomkit import Agent, ChannelCategory, InboundMessage, RoomKit, TextContent
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.event import RoomEvent
from roomkit.providers.ai.base import (
    AIContext,
    AITool,
    ProviderError,
    StreamEvent,
    StreamTextDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_framework import SimpleChannel


class FinalizingProvider(MockAIProvider):
    def __init__(self, *, structured: bool) -> None:
        super().__init__(streaming=True)
        self.structured = structured
        self.unwinding = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    @property
    def supports_structured_streaming(self) -> bool:
        return self.structured

    async def generate_stream(self, context: AIContext) -> AsyncIterator[str]:
        try:
            yield "partial"
            await asyncio.Future()
        finally:
            self.unwinding.set()
            await self.release.wait()
            self.finished.set()

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        try:
            yield StreamTextDelta(text="partial")
            await asyncio.Future()
        finally:
            self.unwinding.set()
            await self.release.wait()
            self.finished.set()


class PausedTransport(SimpleChannel):
    def __init__(self, *, fail: bool) -> None:
        super().__init__("input")
        self.fail = fail
        self.rendering = asyncio.Event()

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self,
        stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        await anext(stream)
        self.rendering.set()
        if self.fail:
            raise RuntimeError("renderer failed")
        await asyncio.Future()
        return ChannelOutput.empty()


class FailingProvider(MockAIProvider):
    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        raise ProviderError("unavailable", retryable=True)
        yield  # pragma: no cover


@pytest.mark.parametrize("kind", ["plain", "structured", "tools", "fallback", "fallback_tools"])
@pytest.mark.parametrize("fail", [False, True])
async def test_stream_finalizer_finishes_before_delivery_completion(kind: str, fail: bool) -> None:
    provider = FinalizingProvider(structured=kind != "plain")
    transport = PausedTransport(fail=fail)
    kwargs: dict[str, Any] = {}
    if "tools" in kind:
        kwargs["tools"] = [
            AITool(name="lookup", description="lookup", parameters={"type": "object"})
        ]
    if kind.startswith("fallback"):
        kwargs["fallback_provider"] = provider
        primary = FailingProvider(streaming=True)
    else:
        primary = provider
    ai = Agent("ai", provider=primary, **kwargs)
    kit = RoomKit()
    kit.register_channel(transport)
    kit.register_channel(ai)
    await kit.create_room(room_id="room")
    await kit.attach_channel("room", "input")
    await kit.attach_channel("room", "ai", category=ChannelCategory.INTELLIGENCE)
    try:
        result = await kit.process_inbound(
            InboundMessage(channel_id="input", sender_id="user", content=TextContent(body="go")),
            room_id="room",
            defer_delivery=True,
        )
        assert result.delivery is not None
        await asyncio.wait_for(transport.rendering.wait(), 2)
        operation = asyncio.create_task(
            result.delivery.wait() if fail else result.delivery.cancel()
        )
        await asyncio.wait_for(provider.unwinding.wait(), 2)
        assert not operation.done()
        assert not result.delivery.done
        assert not provider.finished.is_set()
        provider.release.set()
        await asyncio.wait_for(operation, 2)
        assert provider.finished.is_set()
        assert ai.active_turns == 0
        assert result.delivery.done
        if fail:
            assert str(result.error) == "renderer failed"
        else:
            assert result.cancellation_reason == "caller_cancelled"
    finally:
        provider.release.set()
        await kit.close()
