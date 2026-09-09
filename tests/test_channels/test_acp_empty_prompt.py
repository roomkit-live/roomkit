"""A solicited event can get its prompt from host-managed content."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import make_event
from tests.test_channels.test_acp import _binding, _channel, _context, _prompt, _sent


@pytest.mark.parametrize("body", ["", "   "])
async def test_host_content_supplies_a_prompt_without_mutating_the_event(
    tmp_path: Any, body: str
) -> None:
    trigger = make_event(room_id="room-1", body=body, index=1)
    before = trigger.model_dump()
    seen = []

    async def contribute(context: Any, event: Any) -> list[str]:
        seen.append(event)
        return ["The user supplied an image without a caption."]

    channel, connection, _ = _channel(tmp_path, emit_updates=False, context_contributor=contribute)
    try:
        await _prompt(channel, trigger, _context(trigger))
        assert _sent(connection).strip() == "The user supplied an image without a caption."
        assert seen == [trigger]
        assert trigger.model_dump() == before
        assert len(connection.prompt_calls) == 1
    finally:
        await channel.close()


@pytest.mark.parametrize("body", ["", "   "])
@pytest.mark.parametrize("blocks", [None, [], [" "]])
async def test_no_turn_without_text_or_host_content(tmp_path: Any, body: str, blocks: Any) -> None:
    async def contribute(context: Any, event: Any) -> list[str]:
        return blocks

    channel, connection, _ = _channel(
        tmp_path, context_contributor=contribute if blocks is not None else None
    )
    trigger = make_event(room_id="room-1", body=body, index=1)
    missed = make_event(room_id="room-1", body="Earlier text", index=0)
    try:
        output = await channel.on_event(trigger, _binding(), _context(missed, trigger))
        assert output.responded is False
        assert output.response_stream is None
        assert connection.prompt_calls == []
    finally:
        await channel.close()
