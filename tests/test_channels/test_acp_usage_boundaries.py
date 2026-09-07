"""A terminal usage report is immutable while the stream consumer catches up."""

from __future__ import annotations

import asyncio
import socket
from copy import deepcopy
from typing import Any

import acp
import pytest
from acp.schema import Cost, NewSessionResponse, PromptResponse, Usage

from roomkit import ACPChannel, AIResponseEvent, HookExecution, HookTrigger, RoomKit
from roomkit.models.delivery import InboundMessage
from roomkit.models.event import TextContent
from tests.conftest import make_event
from tests.test_channels.test_acp import TestEndOfTurnReport as _EndOfTurnReport
from tests.test_channels.test_acp import _InProcessEchoAgent, _SocketPairTransport
from tests.test_channels.test_acp_usage import _binding, _channel, _context, _envelope, _update
from tests.test_framework import SimpleChannel


@pytest.mark.parametrize("transport_report", [True, False])
async def test_late_notification_cannot_overwrite_terminal_snapshot(
    tmp_path: Any, transport_report: bool
) -> None:
    channel, connection, _ = _channel(tmp_path)
    reports = _EndOfTurnReport._capture(channel)
    envelope = _envelope()

    async def recovered(session_id: str, prompt: Any, **kwargs: Any) -> PromptResponse:
        await connection.client.session_update(session_id, acp.update_agent_message_text("answer"))
        return PromptResponse(stop_reason="end_turn", field_meta={"roomkit.live/usage": envelope})

    connection.prompt = recovered
    output = await channel.on_event(make_event(body="recover"), _binding(), _context())
    stream = output.response_stream
    assert await anext(stream) == "answer"
    await channel._turns["session-1"].runner
    # The terminal result is queued, but the consumer has not read it yet.
    late_meta = {"roomkit.live/usage": _envelope("session-1")} if transport_report else None
    await connection.client.session_update(
        "session-1", _update(cost=Cost(amount=99, currency="EUR"), field_meta=late_meta)
    )
    assert [chunk async for chunk in stream] == []
    assert reports[0].usage_metadata["session_id"] == "original-session"
    assert reports[0].usage_metadata["usage_report"] == envelope["usage_report"]
    assert reports[0].usage["cost"] == 0
    assert reports[0].usage["currency"] == "USD"
    await channel.close()


class _UsageSocketPairTransport(_SocketPairTransport):
    @property
    def provides_usage_metadata(self) -> bool:
        return True


class _UsageAgent(_InProcessEchoAgent):
    def __init__(self, envelope: dict[str, Any]) -> None:
        self.envelope = envelope

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        return NewSessionResponse(session_id=self.envelope["session_id"])

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        await self.conn.session_update(
            session_id, _update(field_meta={"roomkit.live/usage": self.envelope})
        )
        await self.conn.session_update(session_id, acp.update_agent_message_text("answer"))
        return PromptResponse(
            stop_reason="end_turn",
            usage=Usage(input_tokens=2, output_tokens=3, total_tokens=5),
            field_meta={"roomkit.live/usage": self.envelope},
        )


@pytest.mark.parametrize("trusted", [True, False])
async def test_two_real_sdk_sessions_in_one_room_reach_framework_hooks(
    tmp_path: Any, trusted: bool
) -> None:
    """Real JSON-RPC/SDK models, custom transports, and registered RoomKit hook."""
    kit = RoomKit()
    reports: list[AIResponseEvent] = []
    servers: list[Any] = []
    writers: list[Any] = []
    received = asyncio.Event()

    @kit.hook(HookTrigger.ON_AI_RESPONSE, execution=HookExecution.ASYNC)
    async def observe(event: AIResponseEvent, ctx: Any) -> None:
        reports.append(event)
        if len(reports) == 2:
            received.set()

    try:
        kit.register_channel(SimpleChannel("sms"))
        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "sms")
        for name in ("one", "two"):
            left, right = socket.socketpair()
            ar, aw = await asyncio.open_connection(sock=left)
            cr, cw = await asyncio.open_connection(sock=right)
            writers.append(aw)
            envelope = deepcopy(_envelope(f"{name}-session"))
            envelope["agent_id"] = name
            servers.append(asyncio.create_task(acp.run_agent(_UsageAgent(envelope), aw, ar)))
            kit.register_channel(
                ACPChannel(
                    name,
                    cwd=tmp_path,
                    transport=(_UsageSocketPairTransport if trusted else _SocketPairTransport)(
                        cr, cw
                    ),
                )
            )
            await kit.attach_channel("room-1", name)
        result = await kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id="user", content=TextContent(body="go"))
        )
        assert result.error is None
        await asyncio.wait_for(received.wait(), 2)
        assert {event.usage_metadata["session_id"] for event in reports} == {
            "one-session",
            "two-session",
        }
        for event in reports:
            assert event.room_id == "room-1"
            assert event.usage_metadata["transport"] == "socketpair"
            assert event.usage["input_tokens"] == 2
            if trusted:
                assert event.usage_metadata["agent_id"] == event.channel_id
                assert event.usage_metadata["usage_report"]["report_id"] == "report-1"
                assert event.usage["cost"] == 0
            else:
                assert "agent_id" not in event.usage_metadata
                assert "node_id" not in event.usage_metadata
                assert event.usage_metadata["usage_report"]["identity_source"] == "roomkit"
                assert "cost" not in event.usage
    finally:
        await kit.close()
        for task in servers:
            task.cancel()
        await asyncio.gather(*servers, return_exceptions=True)
        for writer in writers:
            writer.close()
