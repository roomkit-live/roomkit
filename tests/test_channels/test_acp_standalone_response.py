"""The standalone response observer sees the completed turn retired."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from roomkit import HookExecution, HookTrigger, RoomKit
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import TextContent
from tests.test_channels.test_acp import _channel
from tests.test_framework import SimpleChannel


async def test_standalone_response_observer_sees_no_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channel, connection, _ = _channel(tmp_path, emit_updates=False)
    close_reached, release_close, observed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    counts: list[int] = []
    close_session = connection.close_session

    async def closing(session_id: str) -> None:
        close_reached.set()
        await release_close.wait()
        await close_session(session_id)

    monkeypatch.setattr(connection, "close_session", closing)
    async with RoomKit() as kit:
        kit.register_channel(SimpleChannel("source"))
        kit.register_channel(channel)
        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "source")
        await kit.attach_channel("room-1", "acp-agent", category=ChannelCategory.INTELLIGENCE)

        @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
        async def response(event: Any, context: Any) -> None:
            counts.append(channel.active_turns)
            observed.set()

        consuming = asyncio.create_task(
            kit.send_event(
                "room-1",
                "source",
                TextContent(body="Summarize."),
                addressed_to=["acp-agent"],
                event_type=EventType.INSTRUCTION,
                standalone=True,
            )
        )
        try:
            await asyncio.wait_for(observed.wait(), 5)
            await asyncio.wait_for(close_reached.wait(), 5)
            assert counts == [0]
            assert channel.active_turns == 0
            assert not consuming.done()
        finally:
            release_close.set()
            await asyncio.wait_for(consuming, 5)
