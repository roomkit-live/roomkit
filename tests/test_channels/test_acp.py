"""Tests for the Agent Client Protocol intelligence channel."""

from __future__ import annotations

import asyncio
import sys
from io import StringIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import acp
import pytest
from acp.schema import (
    Implementation,
    InitializeResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
)

from roomkit import ACPChannel, CLIChannel, RoomKit
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.room import Room
from roomkit.models.streaming import (
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
)
from roomkit.realtime.base import EphemeralEvent
from roomkit.realtime.memory import InMemoryRealtime
from roomkit.tools.external import ExternalToolHandler, ToolDecision
from tests.conftest import make_event
from tests.test_framework import SimpleChannel

_ECHO_AGENT = """
import asyncio
from uuid import uuid4
from acp import (
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    update_agent_message_text,
)

class EchoAgent(Agent):
    def on_connect(self, conn):
        self.conn = conn

    async def initialize(self, protocol_version, **kwargs):
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(self, cwd, **kwargs):
        return NewSessionResponse(session_id=uuid4().hex)

    async def prompt(self, session_id, prompt, **kwargs):
        await self.conn.session_update(
            session_id=session_id,
            update=update_agent_message_text(prompt[0].text),
        )
        return PromptResponse(stop_reason="end_turn")

asyncio.run(run_agent(EchoAgent()))
"""


class _RecordingToolHandler(ExternalToolHandler):
    def __init__(self, approved: bool = True) -> None:
        self.approved = approved
        self.started = False
        self.stopped = False
        self.requests: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def process_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        **kwargs: Any,
    ) -> ToolDecision:
        self.requests.append({"name": tool_name, "input": tool_input, **kwargs})
        return ToolDecision(approved=self.approved)

    async def on_tool_result(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        result: str,
        **kwargs: Any,
    ) -> None:
        self.results.append({"name": tool_name, "input": tool_input, "result": result, **kwargs})


class _FakeACPConnection:
    def __init__(self, client: Any, *, emit_updates: bool = True) -> None:
        self.client = client
        self.emit_updates = emit_updates
        self.initialize_calls: list[dict[str, Any]] = []
        self.new_session_calls: list[dict[str, Any]] = []
        self.prompt_calls: list[dict[str, Any]] = []
        self.closed_sessions: list[str] = []
        self.cancelled_sessions: list[str] = []
        self.permission_responses: list[Any] = []
        self.authenticated_with: str | None = None
        self._session_counter = 0

    async def initialize(
        self,
        protocol_version: int,
        *,
        client_capabilities: Any,
        client_info: Any,
    ) -> InitializeResponse:
        self.initialize_calls.append(
            {
                "protocol_version": protocol_version,
                "capabilities": client_capabilities,
                "client_info": client_info,
            }
        )
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_info=Implementation(
                name="fake-agent",
                title="Fake Agent",
                version="1.0.0",
            ),
        )

    async def authenticate(self, method_id: str) -> None:
        self.authenticated_with = method_id

    async def new_session(self, **kwargs: Any) -> NewSessionResponse:
        self._session_counter += 1
        self.new_session_calls.append(kwargs)
        return NewSessionResponse(session_id=f"session-{self._session_counter}")

    async def prompt(
        self,
        session_id: str,
        prompt: list[Any],
        **kwargs: Any,
    ) -> PromptResponse:
        self.prompt_calls.append({"session_id": session_id, "prompt": prompt, "metadata": kwargs})
        if self.emit_updates:
            await self.client.session_update(session_id, acp.update_agent_thought_text("checking"))
            await self.client.session_update(session_id, acp.update_agent_message_text("Working "))
            await self.client.session_update(
                session_id,
                acp.start_tool_call(
                    "tool-1",
                    "Read file",
                    kind="read",
                    status="in_progress",
                    raw_input={"path": "README.md"},
                ),
            )
            await self.client.session_update(
                session_id,
                acp.update_plan([acp.plan_entry("Inspect the project", status="in_progress")]),
            )
            await self.client.session_update(
                session_id,
                acp.update_tool_call(
                    "tool-1",
                    status="in_progress",
                    raw_output={"progress": "reading"},
                ),
            )
            permission = await self.client.request_permission(
                session_id,
                acp.update_tool_call(
                    "tool-1",
                    title="Read file",
                    raw_input={"path": "README.md"},
                ),
                [
                    PermissionOption(
                        option_id="allow-once",
                        name="Allow once",
                        kind="allow_once",
                    ),
                    PermissionOption(
                        option_id="reject-once",
                        name="Reject once",
                        kind="reject_once",
                    ),
                ],
            )
            self.permission_responses.append(permission)
            await self.client.session_update(
                session_id,
                acp.update_tool_call(
                    "tool-1",
                    status="completed",
                    raw_output={"content": "RoomKit"},
                ),
            )
            await self.client.session_update(session_id, acp.update_agent_message_text("done"))
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id: str) -> None:
        self.cancelled_sessions.append(session_id)

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)


class _FakeProcessContext:
    def __init__(self, connection: _FakeACPConnection) -> None:
        self.connection = connection
        self.process = SimpleNamespace(returncode=None, stderr=None)
        self.exited = False

    async def __aenter__(self) -> tuple[_FakeACPConnection, Any]:
        return self.connection, self.process

    async def __aexit__(self, *exc: object) -> None:
        self.exited = True
        self.process.returncode = 0


def _channel(
    tmp_path: Any,
    *,
    handler: ExternalToolHandler | None = None,
    emit_updates: bool = True,
) -> tuple[ACPChannel, _FakeACPConnection, _FakeProcessContext]:
    channel = ACPChannel(
        "acp-agent",
        ["fake-agent", "--acp"],
        cwd=tmp_path,
        external_tool_handler=handler,
    )
    connection = _FakeACPConnection(channel._client, emit_updates=emit_updates)
    process_context = _FakeProcessContext(connection)
    channel._create_process_context = lambda sdk: process_context  # type: ignore[method-assign]
    return channel, connection, process_context


def _binding(room_id: str = "room-1") -> ChannelBinding:
    return ChannelBinding(
        channel_id="acp-agent",
        room_id=room_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


class TestACPChannel:
    def test_requires_argument_vector_and_absolute_cwd(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="sequence"):
            ACPChannel("acp", "agent --acp", cwd=tmp_path)
        with pytest.raises(ValueError, match="absolute"):
            ACPChannel("acp", ["agent"], cwd="relative")

    async def test_streams_updates_and_reuses_room_session(self, tmp_path: Any) -> None:
        handler = _RecordingToolHandler(approved=True)
        channel, connection, _ = _channel(tmp_path, handler=handler)
        realtime = InMemoryRealtime()
        channel._realtime = realtime
        ephemeral: list[EphemeralEvent] = []

        async def capture(event: EphemeralEvent) -> None:
            ephemeral.append(event)

        await realtime.subscribe_to_room("room-1", capture)
        context = RoomContext(room=Room(id="room-1"))
        output = await channel.on_event(make_event(body="Inspect it"), _binding(), context)
        chunks = [chunk async for chunk in output.response_stream]
        await asyncio.sleep(0)

        assert connection.initialize_calls[0]["protocol_version"] == 1
        capabilities = connection.initialize_calls[0]["capabilities"]
        assert capabilities.fs.read_text_file is False
        assert capabilities.fs.write_text_file is False
        assert capabilities.terminal is False
        assert connection.prompt_calls[0]["prompt"][0].text == "Inspect it"
        assert any(isinstance(chunk, ThinkingDeltaMarker) for chunk in chunks)
        assert any(isinstance(chunk, ToolCallStartMarker) for chunk in chunks)
        assert any(isinstance(chunk, ToolCallEndMarker) for chunk in chunks)
        assert [chunk for chunk in chunks if isinstance(chunk, str)] == [
            "Working ",
            "done",
        ]
        assert connection.permission_responses[0].outcome.option_id == "allow-once"
        assert handler.results[0]["name"] == "Read file"
        assert any(event.data.get("type") == "acp_plan_update" for event in ephemeral)
        assert any(event.data.get("type") == "acp_tool_progress" for event in ephemeral)

        second = await channel.on_event(make_event(body="Continue"), _binding(), context)
        _ = [chunk async for chunk in second.response_stream]
        assert len(connection.new_session_calls) == 1
        assert channel.session_id("room-1") == "session-1"
        await channel.close()
        await realtime.close()

    async def test_official_sdk_stdio_round_trip(self, tmp_path: Any) -> None:
        channel = ACPChannel(
            "wire-agent",
            [sys.executable, "-c", _ECHO_AGENT],
            cwd=tmp_path,
        )
        binding = ChannelBinding(
            channel_id="wire-agent",
            room_id="room-1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        output = await channel.on_event(
            make_event(body="wire works"),
            binding,
            RoomContext(room=Room(id="room-1")),
        )

        assert [chunk async for chunk in output.response_stream] == ["wire works"]
        assert channel.info["sdk_version"].startswith("0.11.")
        await channel.close()

    async def test_waits_for_deferred_update_before_ending_stream(self, tmp_path: Any) -> None:
        channel, connection, _ = _channel(tmp_path, emit_updates=False)

        async def prompt_with_deferred_update(
            session_id: str,
            prompt: list[Any],
            **kwargs: Any,
        ) -> PromptResponse:
            asyncio.create_task(
                connection.client.session_update(
                    session_id,
                    acp.update_agent_message_text("last chunk"),
                )
            )
            return PromptResponse(stop_reason="end_turn")

        connection.prompt = prompt_with_deferred_update  # type: ignore[method-assign]
        output = await channel.on_event(
            make_event(body="Inspect"),
            _binding(),
            RoomContext(room=Room(id="room-1")),
        )

        assert [chunk async for chunk in output.response_stream] == ["last chunk"]
        await channel.close()

    async def test_default_permission_policy_rejects(self, tmp_path: Any) -> None:
        channel, connection, _ = _channel(tmp_path)
        output = await channel.on_event(
            make_event(body="Inspect"),
            _binding(),
            RoomContext(room=Room(id="room-1")),
        )
        _ = [chunk async for chunk in output.response_stream]

        assert connection.permission_responses[0].outcome.option_id == "reject-once"
        await channel.close()

    async def test_sessions_are_isolated_by_room_and_can_be_cancelled(self, tmp_path: Any) -> None:
        channel, connection, process_context = _channel(tmp_path, emit_updates=False)
        for room_id in ("room-1", "room-2"):
            output = await channel.on_event(
                make_event(room_id=room_id, body=room_id),
                _binding(room_id),
                RoomContext(room=Room(id=room_id)),
            )
            _ = [chunk async for chunk in output.response_stream]

        assert channel.session_id("room-1") == "session-1"
        assert channel.session_id("room-2") == "session-2"
        assert await channel.cancel("room-1") is True
        assert connection.cancelled_sessions == ["session-1"]
        assert await channel.close_session("room-1") is True
        assert connection.closed_sessions == ["session-1"]

        await channel.close()
        assert set(connection.closed_sessions) == {"session-1", "session-2"}
        assert process_context.exited is True

    async def test_skips_own_and_tool_activity_events(self, tmp_path: Any) -> None:
        channel, _, _ = _channel(tmp_path, emit_updates=False)
        context = RoomContext(room=Room(id="room-1"))
        own = make_event(channel_id="acp-agent", body="own")
        assert (await channel.on_event(own, _binding(), context)).responded is False

        tool_event = RoomEvent(
            room_id="room-1",
            type=EventType.TOOL_CALL_START,
            source=EventSource(channel_id="other", channel_type=ChannelType.AI),
            content=ToolCallContent(tool_name="Read", tool_id="tool-1"),
        )
        assert (await channel.on_event(tool_event, _binding(), context)).responded is False
        await channel.close()

    async def test_framework_persists_acp_stream_segments(self, tmp_path: Any) -> None:
        kit = RoomKit()
        source = SimpleChannel("sms")
        channel, _, _ = _channel(tmp_path)
        kit.register_channel(source)
        kit.register_channel(channel)
        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "sms")
        await kit.attach_channel(
            "room-1",
            "acp-agent",
            category=ChannelCategory.INTELLIGENCE,
        )

        result = await kit.process_inbound(
            InboundMessage(
                channel_id="sms",
                sender_id="user",
                content=TextContent(body="Inspect"),
            )
        )
        timeline = await kit.get_timeline("room-1", limit=20)

        assert result.error is None
        assert [event.type for event in timeline].count(EventType.TOOL_CALL_START) == 1
        assert [event.type for event in timeline].count(EventType.TOOL_CALL_END) == 1
        text = [event.content.body for event in timeline if isinstance(event.content, TextContent)]
        assert "Working " in text
        assert "done" in text
        await kit.close()

    async def test_framework_delivers_acp_text_before_prompt_finishes(
        self,
        tmp_path: Any,
    ) -> None:
        kit = RoomKit()
        cli = CLIChannel("cli", use_color=False)
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        first_update_sent = asyncio.Event()
        finish_prompt = asyncio.Event()

        async def delayed_prompt(
            session_id: str,
            prompt: list[Any],
            **kwargs: Any,
        ) -> PromptResponse:
            await connection.client.session_update(
                session_id,
                acp.update_agent_message_text("First chunk"),
            )
            await connection.client.session_update(
                session_id,
                acp.start_tool_call(
                    "tool-stream",
                    "Inspect files",
                    kind="read",
                    status="in_progress",
                    raw_input={"glob": "src/**/*.py"},
                ),
            )
            first_update_sent.set()
            await finish_prompt.wait()
            await connection.client.session_update(
                session_id,
                acp.update_tool_call(
                    "tool-stream",
                    status="completed",
                    raw_output={"files": 3},
                ),
            )
            await connection.client.session_update(
                session_id,
                acp.update_agent_message_text(" then second"),
            )
            return PromptResponse(stop_reason="end_turn")

        connection.prompt = delayed_prompt  # type: ignore[method-assign]
        kit.register_channel(cli)
        kit.register_channel(channel)
        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "cli")
        await kit.attach_channel(
            "room-1",
            "acp-agent",
            category=ChannelCategory.INTELLIGENCE,
        )

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            processing = asyncio.create_task(
                kit.process_inbound(
                    InboundMessage(
                        channel_id="cli",
                        sender_id="user",
                        content=TextContent(body="Inspect"),
                    )
                )
            )
            await asyncio.wait_for(first_update_sent.wait(), timeout=1)
            for _ in range(10):
                if "First chunk" in stdout.getvalue():
                    break
                await asyncio.sleep(0)

            assert "First chunk" in stdout.getvalue()
            assert "🔧 Inspect files" in stdout.getvalue()
            assert processing.done() is False

            finish_prompt.set()
            result = await asyncio.wait_for(processing, timeout=1)

        assert result.error is None
        assert "✓ Inspect files" in stdout.getvalue()
        await kit.close()

    async def test_register_channel_wires_external_tool_hooks(self, tmp_path: Any) -> None:
        handler = _RecordingToolHandler()
        channel, _, _ = _channel(tmp_path, handler=handler, emit_updates=False)
        kit = RoomKit()
        kit.register_channel(channel)

        assert handler._channel_id == "acp-agent"
        assert handler._before_tool_hook is not None
        assert handler._on_tool_hook is not None
        await kit.close()
