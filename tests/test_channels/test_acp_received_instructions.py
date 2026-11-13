"""A prompt received by the agent stays acknowledged when its reader stops."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import acp
import pytest
from acp.schema import PromptResponse

from tests.conftest import make_event
from tests.test_channels.test_acp import _binding, _context, _prompt, _sent
from tests.test_channels.test_acp_discussion import _instructed


async def test_received_instructions_survive_abandonment_before_terminal_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channel, connection = _instructed(tmp_path)
    returned = asyncio.Event()

    async def prompt(session_id: str, prompt: Any, **kwargs: Any) -> PromptResponse:
        connection.prompt_calls.append(
            {"session_id": session_id, "prompt": prompt, "metadata": kwargs}
        )
        if len(connection.prompt_calls) == 1:
            await connection.client.session_update(
                session_id, acp.update_agent_message_text("partial")
            )
            await returned.wait()
        return PromptResponse(stop_reason="end_turn")

    monkeypatch.setattr(connection, "prompt", prompt)
    first = make_event(room_id="room-1", body="hello", index=0)
    output = await channel.on_event(first, _binding(), _context(first))
    assert output.response_stream is not None
    try:
        assert await asyncio.wait_for(anext(output.response_stream), 5) == "partial"
        session = channel.session_id("room-1")
        returned.set()
        async with asyncio.timeout(5):
            while not output.response_metadata["acp"].get("prompt_returned"):
                await asyncio.sleep(0)
        await output.response_stream.aclose()
        assert channel.session_id("room-1") == session
        second = make_event(room_id="room-1", body="next", index=1)
        await asyncio.wait_for(_prompt(channel, second, _context(first, second)), 5)
        assert "You are Dev." in _sent(connection, 0)
        assert _sent(connection, 1) == "next"
    finally:
        await channel.close()
