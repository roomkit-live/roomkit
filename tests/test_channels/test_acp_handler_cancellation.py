"""An ACP permission request pending while its turn ends, or the agent withdraws it."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from pathlib import Path
from typing import Any

import acp
import pytest

from roomkit import ACPChannel, InboundMessage, RoomKit, TextContent
from roomkit.tools import PolicyExternalToolHandler, ToolDecision
from tests.test_channels.test_acp import _channel, _SocketPairTransport
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
async def test_turn_cancellation_answers_a_pending_permission_cancelled(
    tmp_path: Path, announced: bool
) -> None:
    """ACP: a client that cancels a prompt MUST answer each of its pending
    permission requests ``cancelled``. The handler's decision is abandoned,
    and the call, announced or not, is reported once, cancelled (RFC §6.4,
    §9.3)."""
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
        answer = await asyncio.wait_for(permission[0], 3)
        assert answer.outcome.outcome == "cancelled"
        assert handler.cancelled and not handler.completed
        assert handler.withdrawn.is_set()
    finally:
        handler.withdrawn.set()
        await asyncio.gather(*permission, return_exceptions=True)
        await kit.close()
    assert heard.sync == []
    assert [(e.tool_call_id, e.cancelled, e.refused) for e in heard.observed] == [
        ("read", True, False)
    ]


class HeldApproval(PolicyExternalToolHandler):
    """Approves each call once released, recording each one it was asked."""

    def __init__(self) -> None:
        super().__init__()
        self.asked: asyncio.Queue[str] = asyncio.Queue()
        self.release = asyncio.Event()
        self.withdrawn: list[str] = []
        self.cancelled = False

    async def process_tool_call(self, name: str, arguments: Any, **kwargs: Any) -> ToolDecision:
        await self.asked.put(kwargs["tool_call_id"])
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return ToolDecision(approved=True)

    async def on_tool_cancelled(self, name: str, arguments: Any, **kwargs: Any) -> None:
        self.withdrawn.append(kwargs["tool_call_id"])
        await super().on_tool_cancelled(name, arguments, **kwargs)


class _WireAgent:
    """The agent's end of a socket pair, one JSON-RPC message per line, so
    that the SDK's own router takes what it sends."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer

    async def read(self) -> dict[str, Any]:
        return json.loads(await asyncio.wait_for(self.reader.readline(), 3))

    def send(self, message: dict[str, Any]) -> None:
        self.writer.write(json.dumps({"jsonrpc": "2.0", **message}).encode() + b"\n")

    async def prompted(self) -> dict[str, Any]:
        """Answer ``initialize`` and ``session/new``; return the prompt."""
        while (request := await self.read())["method"] != "session/prompt":
            result = {"sessionId": "s1"} if request["method"] == "session/new" else {}
            if request["method"] == "initialize":
                result = {"protocolVersion": 1, "agentCapabilities": {}}
            self.send({"id": request["id"], "result": result})
        return request

    def ask(self, request_id: str, tool_call_id: str) -> None:
        self.send(
            {
                "id": request_id,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "s1",
                    "toolCall": {"toolCallId": tool_call_id, "title": "Read", "rawInput": {}},
                    "options": [option.model_dump(by_alias=True) for option in _OPTIONS],
                },
            }
        )


async def test_an_agents_cancel_request_leaves_the_decision_to_the_handler(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The agent withdraws a pending permission request with
    ``$/cancel_request``: ACP lets a client ignore it, and RoomKit does,
    without an error. The handler keeps deciding the request, and its
    decision is the answer the agent gets."""
    caplog.set_level(logging.DEBUG)
    agent_sock, client_sock = socket.socketpair()
    agent_reader, agent_writer = await asyncio.open_connection(sock=agent_sock)
    client_reader, client_writer = await asyncio.open_connection(sock=client_sock)
    agent = _WireAgent(agent_reader, agent_writer)
    handler = HeldApproval()
    channel = ACPChannel(
        "acp-agent",
        transport=_SocketPairTransport(client_reader, client_writer),
        cwd=tmp_path,
        external_tool_handler=handler,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await _room(kit, "acp-agent")
    turn = asyncio.create_task(
        kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id="u", content=TextContent(body="go"))
        )
    )
    try:
        prompt = await agent.prompted()
        agent.ask("first", "read")
        assert await asyncio.wait_for(handler.asked.get(), 3) == "read"
        agent.send({"method": "$/cancel_request", "params": {"requestId": "first"}})
        # Taken in wire order: once the second request reaches the handler,
        # the notification before it was taken.
        agent.ask("second", "write")
        assert await asyncio.wait_for(handler.asked.get(), 3) == "write"
        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == []
        assert handler.withdrawn == [] and not handler.cancelled
        handler.release.set()
        answers = {(reply := await agent.read())["id"]: reply["result"] for _ in range(2)}
        assert answers == {
            "first": {"outcome": {"outcome": "selected", "optionId": "a"}},
            "second": {"outcome": {"outcome": "selected", "optionId": "a"}},
        }
        agent.send({"id": prompt["id"], "result": {"stopReason": "end_turn"}})
        await asyncio.wait_for(turn, 3)
    finally:
        handler.release.set()
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await kit.close()
        agent_writer.close()
