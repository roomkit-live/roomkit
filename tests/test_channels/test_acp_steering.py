"""Steering addresses a returned prompt until its response stream ends."""

from __future__ import annotations

import asyncio
import contextvars
from pathlib import Path
from typing import Any

import acp
from acp.schema import PromptResponse

from roomkit import ACPChannel
from roomkit.core._turn_scope import enter_turn
from roomkit.models import Cancel
from tests.conftest import make_event
from tests.test_channels.test_acp import (
    _binding,
    _context,
    _FakeACPConnection,
    _FakeTransport,
)


class _CompletedPrompt(_FakeACPConnection):
    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        for text in ("first", "second", "third"):
            await self.client.session_update(session_id, acp.update_agent_message_text(text))
        return PromptResponse(stop_reason="end_turn")


async def test_steering_reaches_a_finished_prompt_until_its_stream_ends(tmp_path: Path) -> None:
    connection = _CompletedPrompt(None, emit_updates=False)
    channel = ACPChannel("acp-agent", transport=_FakeTransport(connection), cwd=tmp_path)
    connection.client = channel._client
    event = make_event(room_id="room-1", body="hello", index=0)

    async def read() -> None:
        output = await channel.on_event(event, _binding(), _context(event))
        stream = output.response_stream
        assert stream is not None
        try:
            first = await anext(stream)
            async with asyncio.timeout(5):
                while not output.response_metadata["acp"].get("prompt_returned"):
                    await asyncio.sleep(0)
            second = await anext(stream)
            assert (first, second) == ("first", "second")
            assert await channel.steer(Cancel(), turn_id="turn-1") == 1
            assert connection.cancelled_sessions == ["session-1"]
            assert [chunk async for chunk in stream] == ["third"]
            assert await channel.steer(Cancel(), turn_id="turn-1") is None
        finally:
            await stream.aclose()

    context = contextvars.copy_context()
    context.run(enter_turn, "turn-1")
    try:
        await asyncio.wait_for(asyncio.create_task(read(), context=context), 5)
    finally:
        await channel.close()
