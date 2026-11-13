"""A temporary ACP session is released even after its RPC has returned."""

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
from tests.test_channels.test_acp_instruction import _instruction


async def test_a_received_standalone_reply_is_released_when_its_reader_is_abandoned(
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
    output = await channel.on_event(_instruction(standalone=True), _binding(), _context())
    assert output.response_stream is not None
    try:
        assert await asyncio.wait_for(anext(output.response_stream), 5) == "partial"
        assert channel.session_id("room-1") is None
        returned.set()
        async with asyncio.timeout(5):
            while not output.response_metadata["acp"].get("prompt_returned"):
                await asyncio.sleep(0)
            # Finish receipt processing without resuming the response reader.
            runner = channel._turns["session-1"].runner
            assert runner is not None
            await asyncio.shield(runner)
        await output.response_stream.aclose()
        assert connection.closed_sessions == ["session-1"]
        assert connection.cancelled_sessions == []
        assert output.response_metadata["acp"]["standalone"] is True
        assert channel.session_id("room-1") is None
        following = make_event(room_id="room-1", body="next", index=1)
        await asyncio.wait_for(_prompt(channel, following, _context(following)), 5)
        assert connection.prompt_calls[-1]["session_id"] == "session-2"
        assert "You are Dev." in _sent(connection, 1)
    finally:
        await channel.close()
