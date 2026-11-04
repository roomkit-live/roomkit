"""A response its transport stopped reading (a barge-in) ends ``cancelled``,
never read as an answer that completed (RMK-479, RFC §6.4, §12.2 step 13s).

Delegated with that transport shared, the worker's task fails as a turn a
stop cut does; as a room turn, the caller reads its end under ``turns``.
Whether the stop came after text, while the model was still thinking, or
after an ACP agent had already finished its prompt: the reader did not read
the turn to its end.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import acp
import pytest
from acp.schema import PromptResponse

from roomkit import RoomKit
from roomkit.channels.agent import Agent
from roomkit.models.channel import ChannelBinding, ChannelOutput
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType, TaskStatus
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.streaming import ThinkingDeltaMarker
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
    AITool,
    AIToolCall,
    StreamEvent,
    StreamTextDelta,
    StreamThinkingDelta,
)
from roomkit.providers.ai.mock import MockAIProvider
from tests.test_channels.test_acp import _channel
from tests.test_framework import SimpleChannel

_WORDS = ("The ", "partial ", "rest ", "of the answer.")


class _Slow(MockAIProvider):
    """Streams the start of an answer, then the rest slowly."""

    def __init__(self) -> None:
        super().__init__(streaming=True)

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        for word in ("The ", "partial "):
            yield StreamTextDelta(text=word)
        await asyncio.sleep(0.3)
        for word in ("rest ", "of the answer."):
            yield StreamTextDelta(text=word)


class _Thinks(MockAIProvider):
    """Thinks first, then answers a moment later."""

    def __init__(self) -> None:
        super().__init__(streaming=True)

    async def generate_structured_stream(self, context: AIContext) -> AsyncIterator[StreamEvent]:
        for word in ("Let me ", "think "):
            yield StreamThinkingDelta(thinking=word)
        await asyncio.sleep(0.3)
        yield StreamTextDelta(text="The answer.")


class _StopsReading(SimpleChannel):
    """Renders a stream and stops, as a barge-in does: after two words of
    text (*pause* later), however its chunks cut them, or at the first
    thinking delta."""

    def __init__(self, channel_id: str, *, on_thinking: bool = False, pause: float = 0) -> None:
        super().__init__(channel_id)
        self._on_thinking = on_thinking
        self._pause = pause

    @property
    def supports_streaming_delivery(self) -> bool:
        return True

    async def deliver_stream(
        self,
        text_stream: AsyncIterator[Any],
        event: RoomEvent,
        binding: ChannelBinding,
        context: RoomContext,
    ) -> ChannelOutput:
        seen = 0
        async for chunk in text_stream:
            if isinstance(chunk, ThinkingDeltaMarker) and self._on_thinking:
                return ChannelOutput.empty()
            if isinstance(chunk, str):
                seen += len(chunk.split())
                if seen >= 2:
                    await asyncio.sleep(self._pause)
                    return ChannelOutput.empty()
        return ChannelOutput.empty()


async def _ai_agent(kit: RoomKit, tmp_path: Path, provider: MockAIProvider) -> str:
    kit.register_channel(Agent("worker", provider=provider))
    return "worker"


async def _acp_agent(kit: RoomKit, tmp_path: Path) -> str:
    """An ACP agent whose prompt returns at once, its whole reply queued
    before the transport reads it."""
    channel, connection, _ = _channel(tmp_path, emit_updates=False)

    async def speaks_fast(session_id: str, prompt: list[Any], **kw: Any) -> PromptResponse:
        for word in _WORDS:
            await connection.client.session_update(session_id, acp.update_agent_message_text(word))
        return PromptResponse(stop_reason="end_turn")

    connection.prompt = speaks_fast  # type: ignore[method-assign]
    kit.register_channel(channel)
    return channel.channel_id


STOPS: dict[str, tuple[Callable[[], _StopsReading], Any]] = {
    "after-text": (lambda: _StopsReading("tx"), lambda k, t: _ai_agent(k, t, _Slow())),
    "while-thinking": (
        lambda: _StopsReading("tx", on_thinking=True),
        lambda k, t: _ai_agent(k, t, _Thinks()),
    ),
    # The agent finished its prompt meanwhile: its record says completed.
    "acp-finished": (lambda: _StopsReading("tx", pause=0.1), _acp_agent),
}
EVERY_STOP = pytest.mark.parametrize("stop", list(STOPS))


async def _kit(stop: str, tmp_path: Path) -> tuple[RoomKit, str]:
    transport, agent = STOPS[stop]
    kit = RoomKit()
    kit.register_channel(transport())
    agent_id = await agent(kit, tmp_path)
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "tx")
    return kit, agent_id


@EVERY_STOP
async def test_a_delegated_turn_its_transport_stopped_fails_cancelled(
    stop: str, tmp_path: Path
) -> None:
    kit, agent_id = await _kit(stop, tmp_path)

    task = await kit.delegate("r", agent_id, "Answer.", wait=True, share_channels=["tx"])
    await kit.close()

    assert task.result is not None
    assert task.result.status == TaskStatus.FAILED
    assert task.result.metadata.get("loop_end_reason") == "cancelled"


@EVERY_STOP
async def test_a_room_turn_its_transport_stopped_tells_its_caller_cancelled(
    stop: str, tmp_path: Path
) -> None:
    kit, agent_id = await _kit(stop, tmp_path)
    await kit.attach_channel("r", agent_id, category=ChannelCategory.INTELLIGENCE)

    result = await kit.process_inbound(
        InboundMessage(channel_id="tx", sender_id="u", content=TextContent(body="Answer."))
    )
    await kit.close()

    assert dict(result.response_metadata).get("turns") == {
        agent_id: {"loop_end_reason": "cancelled"}
    }


async def test_a_reader_gone_while_a_call_runs_leaves_no_generation_in_flight() -> None:
    """The round that runs ends, its end stored, and no model is asked again:
    the record reads complete (RFC §6.4)."""
    running = asyncio.Event()
    release = asyncio.Event()

    class Leaves(SimpleChannel):
        @property
        def supports_streaming_delivery(self) -> bool:
            return True

        async def deliver_stream(
            self,
            text_stream: AsyncIterator[Any],
            event: RoomEvent,
            binding: ChannelBinding,
            context: RoomContext,
        ) -> ChannelOutput:
            async for item in text_stream:
                if isinstance(item, RoomEvent) and item.type == EventType.TOOL_CALL_START:
                    break
            # Its next read runs the call, and its target leaves meanwhile.
            read = asyncio.ensure_future(anext(text_stream))
            await running.wait()
            release.set()
            read.cancel()
            return ChannelOutput.empty()

    async def wait(name: str, arguments: dict[str, Any]) -> str:
        running.set()
        await release.wait()
        return "found"

    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="Checking.",
                tool_calls=[AIToolCall(id="c1", name="lookup", arguments={})],
                usage={"input_tokens": 1000, "output_tokens": 10},
            )
        ],
        streaming=True,
    )
    kit = RoomKit()
    kit.register_channel(Leaves("ws", ChannelType.WEBSOCKET))
    kit.register_channel(
        Agent(
            "ai",
            provider=provider,
            tools=[AITool(name="lookup", description="d", parameters={"type": "object"})],
            tool_handler=wait,
        )
    )
    await kit.create_room(room_id="r")
    await kit.attach_channel("r", "ws")
    await kit.attach_channel("r", "ai", category=ChannelCategory.INTELLIGENCE)
    asked = InboundMessage(channel_id="ws", sender_id="u", content=TextContent(body="Go."))
    await asyncio.wait_for(kit.process_inbound(asked, room_id="r"), 5)

    rows = await kit.store.list_events("r", limit=20)
    said = next(
        row for row in rows if row.type == EventType.MESSAGE and row.source.channel_id == "ai"
    )
    assert said.metadata["loop_end_reason"] == "cancelled"
    assert said.metadata["ai_usage_complete"] is True
    assert len(provider.calls) == 1
    await kit.close()
