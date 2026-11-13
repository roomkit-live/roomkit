"""An ACP permission belongs to the connection while its turn can end."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import acp
import pytest

from roomkit import InboundMessage, RoomKit, TextContent
from roomkit.tools import PolicyExternalToolHandler, ToolDecision
from tests.test_channels.test_acp import _channel
from tests.test_channels.test_acp_decided_calls import _OPTIONS
from tests.test_refused_reports import _Heard, _room


class WaitingApproval(PolicyExternalToolHandler):
    def __init__(self) -> None:
        super().__init__()
        self.asked = asyncio.Event()
        self.withdrawn = asyncio.Event()
        self.completed = False
        self.cancelled = False

    async def process_tool_call(self, name: str, arguments: Any, **kwargs: Any) -> ToolDecision:
        self.asked.set()
        try:
            await self.withdrawn.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.completed = True
        return ToolDecision(approved=False, reason="turn ended")

    async def on_tool_cancelled(self, name: str, arguments: Any, **kwargs: Any) -> None:
        self.withdrawn.set()
        await super().on_tool_cancelled(name, arguments, **kwargs)


@pytest.mark.parametrize("announced", [True, False], ids=["announced", "not-announced"])
async def test_turn_cancellation_leaves_approval_to_the_handlers_withdrawal(
    tmp_path: Path, announced: bool
) -> None:
    handler = WaitingApproval()
    kit = RoomKit()
    channel, connection, _ = _channel(tmp_path, handler=handler, emit_updates=False)
    permission: list[asyncio.Task[Any]] = []

    async def prompt(session_id: str, *args: Any, **kwargs: Any) -> Any:
        if announced:
            await connection.client.session_update(
                session_id, acp.start_tool_call("read", "Read", status="in_progress", raw_input={})
            )
        request = acp.update_tool_call("read", title="Read", raw_input={})
        # The connection owns the request, independently of the prompt task.
        permission.append(
            asyncio.create_task(
                connection.client.request_permission(session_id, request, _OPTIONS)
            )
        )
        await asyncio.Event().wait()

    connection.prompt = prompt
    kit.register_channel(channel)
    await _room(kit, "acp-agent")
    heard = _Heard(kit)
    turn = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
        )
    )
    try:
        await asyncio.wait_for(handler.asked.wait(), 3)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        if not announced:
            # Until announced or decided, Python has no call end to report.
            assert not handler.withdrawn.is_set()
            assert not permission[0].done()
            handler.withdrawn.set()
        answer = await asyncio.wait_for(permission[0], 3)
        assert answer.outcome.option_id == "r"
        assert handler.completed and not handler.cancelled
    finally:
        handler.withdrawn.set()
        await asyncio.gather(*permission, return_exceptions=True)
        await kit.close()
    assert heard.sync == []
    assert [(e.tool_call_id, e.cancelled, e.refused) for e in heard.observed] == (
        [("read", True, False)] if announced else []
    )
