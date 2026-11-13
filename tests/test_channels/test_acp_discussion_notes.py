"""Discussion notes are captured when the channel accepts its turn."""

from __future__ import annotations

from pathlib import Path

from roomkit.channels._discussion_turn import DISCUSSION_TURN, issue_mark, retire_mark
from roomkit.channels._turn_notes import TURN_NOTES_HEADER
from tests.conftest import make_event
from tests.test_channels.test_acp import _binding, _channel, _context, _sent


async def test_an_accepted_turn_keeps_its_notes_after_the_mark_is_retired(tmp_path: Path) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    mark = issue_mark(["The room: @sre."])
    event = make_event(body="hello").model_copy(update={"metadata": {DISCUSSION_TURN: mark}})
    try:
        output = await channel.on_event(event, _binding(), _context(event))
        retire_mark(mark)
        assert output.response_stream is not None
        async for _ in output.response_stream:
            pass
        assert _sent(connection).endswith(f"{TURN_NOTES_HEADER}\n\nThe room: @sre.")

        following = await channel.on_event(event, _binding(), _context(event))
        assert following.response_stream is not None
        async for _ in following.response_stream:
            pass
        assert TURN_NOTES_HEADER not in _sent(connection, 1)
    finally:
        retire_mark(mark)
        await channel.close()
