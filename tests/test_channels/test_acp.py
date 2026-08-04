"""Tests for the Agent Client Protocol intelligence channel."""

from __future__ import annotations

import asyncio
import socket
import sys
from io import StringIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import acp
import pytest
from acp.schema import (
    ConfigOptionUpdate,
    Implementation,
    InitializeResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SetSessionConfigOptionResponse,
)

from roomkit import ACPChannel, ACPTransport, CLIChannel, RoomKit, StdioACPTransport
from roomkit.channels._acp_client import (
    _config_labels,
    _config_values,
)
from roomkit.channels.acp_transport import _resolve_spawn_env
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType, EventType, Visibility
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.streaming import (
    ThinkingDeltaMarker,
    ToolCallEndMarker,
    ToolCallStartMarker,
)
from roomkit.providers.ai.base import ProviderError
from roomkit.realtime.base import EphemeralEvent, EphemeralEventType
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


class _InProcessEchoAgent(acp.Agent):
    """The same echo agent as ``_ECHO_AGENT``, running in this process.

    The subprocess copy above proves the stdio path; this one lets a test
    drive the real protocol without spawning anything.
    """

    def on_connect(self, conn: Any) -> None:
        self.conn = conn

    async def initialize(self, protocol_version: int, **kwargs: Any) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(self, cwd: str, **kwargs: Any) -> NewSessionResponse:
        return NewSessionResponse(session_id=uuid4().hex)

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        await self.conn.session_update(
            session_id=session_id,
            update=acp.update_agent_message_text(prompt[0].text),
        )
        return PromptResponse(stop_reason="end_turn")


class _SocketPairTransport(ACPTransport):
    """Carry ACP over an already-connected socket pair.

    Stands in for any transport handed a live pipe it did not create — a
    WebSocket relay to an agent on another machine has this same shape.
    """

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer

    @property
    def name(self) -> str:
        return "socketpair"

    async def open(self, client: Any, *, queue: Any) -> Any:
        return acp.connect_to_agent(client, self._writer, self._reader, queue=queue)

    async def close(self) -> None:
        self._writer.close()


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


def _model_option(current: str) -> SessionConfigOptionSelect:
    """The model picker an ACP agent advertises on its session."""
    return SessionConfigOptionSelect(
        id="model",
        name="Model",
        type="select",
        current_value=current,
        options=[
            SessionConfigSelectOption(value="opus", name="Opus"),
            SessionConfigSelectOption(value="sonnet", name="Sonnet"),
        ],
    )


class _FakeACPConnection:
    def __init__(self, client: Any, *, emit_updates: bool = True) -> None:
        self.client = client
        self.emit_updates = emit_updates
        self.config_options: list[Any] = [_model_option("opus")]
        self.set_config_calls: list[dict[str, Any]] = []
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
        return NewSessionResponse(
            session_id=f"session-{self._session_counter}",
            config_options=list(self.config_options),
        )

    async def set_config_option(
        self,
        *,
        config_id: str,
        session_id: str,
        value: str | bool,
    ) -> SetSessionConfigOptionResponse:
        self.set_config_calls.append(
            {"config_id": config_id, "session_id": session_id, "value": value}
        )
        if config_id == "model" and isinstance(value, str):
            self.config_options = [_model_option(value)]
        return SetSessionConfigOptionResponse(config_options=list(self.config_options))

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
                    content=[
                        {
                            "type": "diff",
                            "path": "/tmp/README.md",
                            "old_text": "old",
                            "new_text": "new",
                        }
                    ],
                ),
            )
            await self.client.session_update(session_id, acp.update_agent_message_text("done"))
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id: str) -> None:
        self.cancelled_sessions.append(session_id)

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)


class _FakeTransport(ACPTransport):
    """Hands the channel a canned connection — no process, no wire.

    The suite drives the channel through the public transport seam rather
    than by patching internals, so every test here also exercises it.
    """

    def __init__(self, connection: _FakeACPConnection) -> None:
        self.connection = connection
        self.opened = 0
        self.exited = False
        self.alive = True

    @property
    def name(self) -> str:
        return "fake"

    async def open(self, client: Any, *, queue: Any) -> Any:
        self.opened += 1
        self.alive = True
        return self.connection

    async def close(self) -> None:
        self.exited = True
        self.alive = False

    def is_alive(self) -> bool:
        return self.alive


def _channel(
    tmp_path: Any,
    *,
    handler: ExternalToolHandler | None = None,
    emit_updates: bool = True,
    room_history: int | None = None,
) -> tuple[ACPChannel, _FakeACPConnection, _FakeTransport]:
    connection = _FakeACPConnection(None, emit_updates=emit_updates)
    transport = _FakeTransport(connection)
    # Omitted rather than repeated, so the channel's own default stays under test.
    window = {} if room_history is None else {"room_history": room_history}
    channel = ACPChannel(
        "acp-agent",
        transport=transport,
        cwd=tmp_path,
        external_tool_handler=handler,
        **window,  # type: ignore[arg-type]
    )
    connection.client = channel._client
    return channel, connection, transport


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

    def test_inherit_env_validation(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="inherit_env"):
            ACPChannel("acp", ["agent"], cwd=tmp_path, inherit_env="SSH_AUTH_SOCK")
        with pytest.raises(ValueError, match="inherit_env"):
            ACPChannel("acp", ["agent"], cwd=tmp_path, inherit_env=["SSH_AUTH_SOCK", ""])
        channel = ACPChannel("acp", ["agent"], cwd=tmp_path, inherit_env=["SSH_AUTH_SOCK"])
        assert channel._transport._inherit_env == ("SSH_AUTH_SOCK",)  # type: ignore[attr-defined]
        default = ACPChannel("acp", ["agent"], cwd=tmp_path)
        assert default._transport._inherit_env == ()  # type: ignore[attr-defined]

    def test_resolve_spawn_env(self) -> None:
        environ = {"SSH_AUTH_SOCK": "/run/agent.sock", "LANG": "fr_CA.UTF-8"}

        # Named vars are read from the parent environment; unset names skip.
        assert _resolve_spawn_env(("SSH_AUTH_SOCK", "MISSING"), None, environ) == {
            "SSH_AUTH_SOCK": "/run/agent.sock"
        }
        # Explicit env entries win over inherited ones.
        assert _resolve_spawn_env(
            ("SSH_AUTH_SOCK",), {"SSH_AUTH_SOCK": "/custom.sock", "X": "1"}, environ
        ) == {"SSH_AUTH_SOCK": "/custom.sock", "X": "1"}
        # Nothing to add -> None, so the SDK keeps its trimmed default.
        assert _resolve_spawn_env((), None, environ) is None
        assert _resolve_spawn_env(("MISSING",), None, environ) is None

    async def test_spawn_receives_resolved_env(self, tmp_path: Any) -> None:
        channel = ACPChannel(
            "acp",
            ["agent"],
            cwd=tmp_path,
            env={"MAX_THINKING_TOKENS": "1024"},
            inherit_env=["SSH_AUTH_SOCK"],
        )
        captured: dict[str, Any] = {}

        class _Context:
            async def __aenter__(self) -> tuple[Any, Any]:
                return SimpleNamespace(), SimpleNamespace(returncode=None, stderr=None)

            async def __aexit__(self, *exc: object) -> None:
                return None

        def spawn(client: Any, *args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return _Context()

        sdk = SimpleNamespace(
            acp=SimpleNamespace(spawn_agent_process=spawn),
            task=SimpleNamespace(InMemoryMessageQueue=lambda: None),
        )
        with (
            patch.dict("os.environ", {"SSH_AUTH_SOCK": "/run/agent.sock"}),
            patch("roomkit.channels.acp_transport._load_sdk", return_value=sdk),
        ):
            await channel._transport.open(channel._client, queue=None)

        assert captured["env"] == {
            "SSH_AUTH_SOCK": "/run/agent.sock",
            "MAX_THINKING_TOKENS": "1024",
        }

    def test_command_and_transport_are_mutually_exclusive(self, tmp_path: Any) -> None:
        # Two distinct mistakes, two distinct messages: "both" and "neither"
        # need different corrections.
        transport = _FakeTransport(_FakeACPConnection(None))
        with pytest.raises(ValueError, match="one, not both"):
            ACPChannel("acp", ["agent"], transport=transport, cwd=tmp_path)
        with pytest.raises(ValueError, match="pass command to spawn"):
            ACPChannel("acp", cwd=tmp_path)

    def test_spawn_only_options_are_rejected_with_a_transport(self, tmp_path: Any) -> None:
        # env/inherit_env shape the subprocess spawn. Accepting them next to a
        # transport that does its own connecting would silently drop them.
        transport = _FakeTransport(_FakeACPConnection(None))
        with pytest.raises(ValueError, match="env and inherit_env"):
            ACPChannel("acp", transport=transport, cwd=tmp_path, env={"A": "1"})
        with pytest.raises(ValueError, match="env and inherit_env"):
            ACPChannel("acp", transport=transport, cwd=tmp_path, inherit_env=["PATH"])

    def test_info_reports_the_transport_in_use(self, tmp_path: Any) -> None:
        assert ACPChannel("acp", ["agent"], cwd=tmp_path).info["transport"] == "stdio"
        channel, _connection, _transport = _channel(tmp_path)
        assert channel.info["transport"] == "fake"

    def test_stdio_transport_validates_its_own_arguments(self, tmp_path: Any) -> None:
        # Reachable directly, not only through the channel that builds one.
        with pytest.raises(ValueError, match="sequence"):
            StdioACPTransport("agent --acp", cwd=tmp_path)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="absolute"):
            StdioACPTransport(["agent"], cwd="relative")
        assert StdioACPTransport(["agent"], cwd=tmp_path).name == "stdio"

    async def test_a_dead_transport_reconnects_and_drops_sessions(self, tmp_path: Any) -> None:
        """The pipe dying takes its sessions with it — a reconnect never resumes.

        This is the process-death path (``returncode`` set) generalised: the
        channel asks the transport, so any transport that can tell gets the
        same recovery.
        """
        channel, _connection, transport = _channel(tmp_path)
        await channel._ensure_connection()
        session = await channel._session_for("room-1", await channel._ensure_connection())
        assert transport.opened == 1

        transport.alive = False
        await channel._ensure_connection()

        assert transport.opened == 2
        assert transport.exited is True
        assert channel.session_id("room-1") is None
        assert session not in channel._session_rooms
        await channel.close()

    async def test_a_failed_handshake_closes_the_transport(self, tmp_path: Any) -> None:
        channel, connection, transport = _channel(tmp_path)

        async def _boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("handshake refused")

        connection.initialize = _boom  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="handshake refused"):
            await channel._ensure_connection()

        assert transport.exited is True
        assert channel._connection is None

    def test_config_values_reads_current_values(self) -> None:
        assert _config_values([_model_option("sonnet")]) == {"model": "sonnet"}
        # Dumped payloads use camelCase; hand-built snake_case is tolerated.
        assert _config_values(
            [
                {"id": "model", "currentValue": "opus"},
                {"id": "fast", "current_value": True},
            ]
        ) == {"model": "opus", "fast": True}
        # Anything without a usable id/value pair is skipped, not guessed at.
        junk = [{"id": "", "currentValue": "x"}, {"currentValue": "y"}, "junk"]
        assert _config_values(junk) == {}
        assert _config_values(None) == {}

    def test_config_labels_name_the_current_value(self) -> None:
        # The real agent's default entry is the opaque value "default".
        assert _config_labels([_model_option("sonnet")]) == {"model": "Sonnet"}
        # Grouped selects are searched too.
        grouped = [
            {
                "id": "model",
                "currentValue": "haiku",
                "options": [
                    {
                        "group": "anthropic",
                        "name": "Anthropic",
                        "options": [{"value": "haiku", "name": "Haiku"}],
                    }
                ],
            }
        ]
        assert _config_labels(grouped) == {"model": "Haiku"}
        # An unknown or unlabelled current value yields no label to show.
        assert _config_labels([{"id": "model", "currentValue": "ghost", "options": []}]) == {}
        assert _config_labels([{"id": "fast", "currentValue": True}]) == {}

    async def test_session_config_follows_the_agent(self, tmp_path: Any) -> None:
        channel, connection, _ = _channel(tmp_path)
        realtime = InMemoryRealtime()
        channel._realtime = realtime
        ephemeral: list[EphemeralEvent] = []

        async def capture(event: EphemeralEvent) -> None:
            ephemeral.append(event)

        await realtime.subscribe_to_room("room-1", capture)
        assert channel.session_config("room-1") == {}  # no session yet

        context = RoomContext(room=Room(id="room-1"))
        output = await channel.on_event(make_event(body="Inspect it"), _binding(), context)
        _ = [chunk async for chunk in output.response_stream]
        await asyncio.sleep(0)
        assert channel.session_config("room-1") == {"model": "opus"}
        # The opening state is announced too — a status bar has nothing to
        # show otherwise until a change happens to occur.
        opening = [e for e in ephemeral if e.data.get("type") == "acp_config_options"]
        assert opening[0].data["values"] == {"model": "opus"}
        # Labels ride along: "opus" is an id, "Opus" is what a bar shows.
        assert opening[0].data["labels"] == {"model": "Opus"}

        # The agent reports a switch the user made inside it (`/model` is
        # handled locally by the agent and never reaches RoomKit).
        await channel._receive_update(
            "session-1",
            ConfigOptionUpdate(
                session_update="config_option_update",
                config_options=[_model_option("sonnet")],
            ),
        )
        await asyncio.sleep(0)

        assert channel.session_config("room-1") == {"model": "sonnet"}
        published = [e for e in ephemeral if e.data.get("type") == "acp_config_options"]
        assert published[-1].data["values"] == {"model": "sonnet"}
        assert published[-1].data["config_options"][0]["currentValue"] == "sonnet"
        await channel.close()
        await realtime.close()

    async def test_set_config_option_opens_a_session_when_needed(self, tmp_path: Any) -> None:
        channel, connection, _ = _channel(tmp_path)
        realtime = InMemoryRealtime()
        channel._realtime = realtime
        ephemeral: list[EphemeralEvent] = []

        async def capture(event: EphemeralEvent) -> None:
            ephemeral.append(event)

        await realtime.subscribe_to_room("room-1", capture)

        values = await channel.set_config_option("room-1", "model", "sonnet")
        await asyncio.sleep(0)

        assert values == {"model": "sonnet"}
        # Our own switch is announced like the agent's own would be.
        announced = [e for e in ephemeral if e.data.get("type") == "acp_config_options"]
        assert announced[-1].data["values"] == {"model": "sonnet"}
        # Descriptors stay available for a picker to render.
        options = channel.config_options("room-1")
        assert options[0]["id"] == "model"
        assert [entry["value"] for entry in options[0]["options"]] == ["opus", "sonnet"]
        assert connection.set_config_calls == [
            {"config_id": "model", "session_id": "session-1", "value": "sonnet"}
        ]
        assert channel.session_config("room-1") == {"model": "sonnet"}
        # The session opened for the switch is the one the first prompt uses.
        context = RoomContext(room=Room(id="room-1"))
        output = await channel.on_event(make_event(body="Go"), _binding(), context)
        _ = [chunk async for chunk in output.response_stream]
        assert len(connection.new_session_calls) == 1
        await channel.close()

    async def test_close_forgets_session_config(self, tmp_path: Any) -> None:
        channel, _connection, _ = _channel(tmp_path)
        await channel.set_config_option("room-1", "model", "sonnet")
        await channel.close()
        assert channel.session_config("room-1") == {}

    async def test_close_survives_an_agent_that_stopped_answering(self, tmp_path: Any) -> None:
        # A hung close_session must not hang the caller: shutdown is bounded
        # and the subprocess is torn down regardless.
        channel, connection, process_context = _channel(tmp_path)
        context = RoomContext(room=Room(id="room-1"))
        output = await channel.on_event(make_event(body="Go"), _binding(), context)
        _ = [chunk async for chunk in output.response_stream]

        async def never_answers(session_id: str) -> None:
            await asyncio.sleep(3600)

        connection.close_session = never_answers  # type: ignore[method-assign]
        with patch("roomkit.channels.acp._SHUTDOWN_TIMEOUT", 0.05):
            await asyncio.wait_for(channel.close(), timeout=5)

        assert process_context.exited is True
        assert channel.session_config("room-1") == {}

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

    async def test_official_sdk_round_trip_over_a_custom_transport(self, tmp_path: Any) -> None:
        """The whole protocol over a pipe the channel did not open itself.

        Real SDK on both ends, real JSON-RPC over a socket pair — no
        subprocess and no stdio anywhere. This is what a transport carrying
        ACP across a network hop has to look like.
        """
        agent_sock, client_sock = socket.socketpair()
        agent_reader, agent_writer = await asyncio.open_connection(sock=agent_sock)
        client_reader, client_writer = await asyncio.open_connection(sock=client_sock)
        serving = asyncio.create_task(
            acp.run_agent(_InProcessEchoAgent(), agent_writer, agent_reader)
        )

        channel = ACPChannel(
            "socket-agent",
            transport=_SocketPairTransport(client_reader, client_writer),
            cwd=tmp_path,
        )
        binding = ChannelBinding(
            channel_id="socket-agent",
            room_id="room-1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )
        output = await channel.on_event(
            make_event(body="over a socket"),
            binding,
            RoomContext(room=Room(id="room-1")),
        )

        assert [chunk async for chunk in output.response_stream] == ["over a socket"]
        assert channel.info["transport"] == "socketpair"
        assert channel.session_id("room-1") is not None
        await channel.close()
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        agent_writer.close()

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

    async def test_closing_a_session_frees_everything_keyed_by_it(self, tmp_path: Any) -> None:
        """A channel that outlives its sessions must not accumulate them.

        Each new session takes a fresh id, so config options kept past a
        close pile up one dead entry per cycle — and the room's turn lock
        outlives the room. Both go when the session does.
        """
        channel, _, _ = _channel(tmp_path, emit_updates=False)
        for _ in range(3):
            output = await channel.on_event(
                make_event(room_id="room-1", body="hi"),
                _binding("room-1"),
                RoomContext(room=Room(id="room-1")),
            )
            _ = [chunk async for chunk in output.response_stream]
            assert await channel.close_session("room-1") is True

        assert channel._sessions == {}
        assert channel._session_rooms == {}
        assert channel._session_options == {}
        assert channel._room_locks == {}
        await channel.close()

    async def test_a_waiter_on_a_retired_room_lock_does_not_race_a_new_caller(
        self, tmp_path: Any
    ) -> None:
        """Retiring the lock must not hand two callers the critical section.

        A coroutine queued on the lock while ``close_session`` retires it
        wakes owning an object no longer in the map; it has to retry on the
        current lock rather than run alongside whoever took that one.
        """
        channel, _, _ = _channel(tmp_path, emit_updates=False)
        output = await channel.on_event(
            make_event(room_id="room-1", body="hi"),
            _binding("room-1"),
            RoomContext(room=Room(id="room-1")),
        )
        _ = [chunk async for chunk in output.response_stream]

        inside: list[str] = []

        async def critical(name: str) -> None:
            async with channel._room_turn_lock("room-1"):
                inside.append(name)
                await asyncio.sleep(0.02)  # overlap is observable here
                assert inside[-1] == name, f"{name} ran alongside {inside[-1]}"

        retired = asyncio.create_task(critical("queued-on-the-old-lock"))
        await asyncio.sleep(0)  # let it queue behind nothing yet
        await channel.close_session("room-1")
        fresh = asyncio.create_task(critical("took-the-new-lock"))
        await asyncio.gather(retired, fresh)
        assert len(inside) == 2
        await channel.close()

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
        tool_end = next(
            event
            for event in timeline
            if event.type == EventType.TOOL_CALL_END and isinstance(event.content, ToolCallContent)
        )
        # The raw result stays the raw output; the ACP display payload
        # (diff blocks) travels in structured_content for UI surfaces.
        assert tool_end.content.result == {"content": "RoomKit"}
        acp_content = (tool_end.content.structured_content or {}).get("acp_content")
        assert acp_content is not None
        assert acp_content[0]["type"] == "diff"
        assert acp_content[0]["path"] == "/tmp/README.md"
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

    async def test_a_turn_that_dies_mid_tool_closes_the_call(self, tmp_path: Any) -> None:
        """An agent that disappears mid-tool must not leave the card spinning.

        The stream is what the timeline is built from, so the closing marker
        has to arrive there — before the error that ends the turn.
        """
        channel, connection, _ = _channel(tmp_path, emit_updates=False)

        async def prompt_then_die(
            session_id: str,
            prompt: list[Any],
            **kwargs: Any,
        ) -> PromptResponse:
            await connection.client.session_update(
                session_id,
                acp.start_tool_call(
                    "tool-1",
                    "Terminal",
                    kind="execute",
                    status="in_progress",
                    raw_input={"command": "echo hi"},
                ),
            )
            raise RuntimeError("agent process restarted")

        connection.prompt = prompt_then_die  # type: ignore[method-assign]
        output = await channel.on_event(
            make_event(body="Run it"),
            _binding(),
            RoomContext(room=Room(id="room-1")),
        )

        chunks: list[Any] = []
        with pytest.raises(ProviderError):
            async for chunk in output.response_stream:
                chunks.append(chunk)

        assert [type(chunk) for chunk in chunks] == [ToolCallStartMarker, ToolCallEndMarker]
        end = chunks[-1]
        assert end.tool_id == "tool-1"
        assert end.tool_name == "Terminal"
        assert end.status == "failed"
        # Why it failed matters to whoever reads the thread: a tool that never
        # returned because the turn died is not a tool that failed on its own.
        assert "ended before" in (end.error or "")
        await channel.close()

    async def test_a_cancelled_turn_closes_the_tool_it_left_open(self, tmp_path: Any) -> None:
        """Stop comes back through the ordinary end of a turn, not an error.

        ``cancel()`` forwards to the ACP session and the prompt returns with
        its stop reason, so a turn stopped by the user looks like any other
        finished one — and takes its unfinished tools with it.
        """
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        realtime = InMemoryRealtime()
        channel._realtime = realtime
        ephemeral: list[EphemeralEvent] = []

        async def capture(event: EphemeralEvent) -> None:
            ephemeral.append(event)

        await realtime.subscribe_to_room("room-1", capture)

        async def prompt_then_stop(
            session_id: str,
            prompt: list[Any],
            **kwargs: Any,
        ) -> PromptResponse:
            await connection.client.session_update(
                session_id,
                acp.start_tool_call(
                    "tool-1",
                    "Write",
                    kind="edit",
                    status="in_progress",
                    raw_input={"path": "/tmp/notes.md"},
                ),
            )
            return PromptResponse(stop_reason="cancelled")

        connection.prompt = prompt_then_stop  # type: ignore[method-assign]
        output = await channel.on_event(
            make_event(body="Write it"),
            _binding(),
            RoomContext(room=Room(id="room-1")),
        )
        chunks = [chunk async for chunk in output.response_stream]
        await asyncio.sleep(0)  # let the realtime fan-out run

        ends = [chunk for chunk in chunks if isinstance(chunk, ToolCallEndMarker)]
        assert len(ends) == 1
        assert ends[0].tool_id == "tool-1"
        assert ends[0].status == "failed"
        ephemeral_ends = [
            event
            for event in ephemeral
            if event.type is EphemeralEventType.TOOL_CALL_END
            and event.data["tool_calls"][0]["id"] == "tool-1"
        ]
        assert len(ephemeral_ends) == 1
        assert ephemeral_ends[0].data["tool_calls"][0]["status"] == "failed"
        await channel.close()
        await realtime.close()

    async def test_framework_stores_the_end_of_a_tool_a_dead_turn_abandoned(
        self,
        tmp_path: Any,
    ) -> None:
        """The spinner that survives a reload is a stored row, so check the store.

        A ``TOOL_CALL_START`` with no ``TOOL_CALL_END`` renders as pending
        forever; the timeline is what a reloading page reads back.
        """
        kit = RoomKit()
        source = SimpleChannel("sms")
        channel, connection, _ = _channel(tmp_path, emit_updates=False)

        async def prompt_then_die(
            session_id: str,
            prompt: list[Any],
            **kwargs: Any,
        ) -> PromptResponse:
            await connection.client.session_update(
                session_id,
                acp.update_agent_message_text("Running it"),
            )
            await connection.client.session_update(
                session_id,
                acp.start_tool_call(
                    "tool-1",
                    "Terminal",
                    kind="execute",
                    status="in_progress",
                    raw_input={"command": "echo hi"},
                ),
            )
            raise RuntimeError("agent process restarted")

        connection.prompt = prompt_then_die  # type: ignore[method-assign]
        kit.register_channel(source)
        kit.register_channel(channel)
        await kit.create_room(room_id="room-1")
        await kit.attach_channel("room-1", "sms")
        await kit.attach_channel(
            "room-1",
            "acp-agent",
            category=ChannelCategory.INTELLIGENCE,
        )

        await kit.process_inbound(
            InboundMessage(
                channel_id="sms",
                sender_id="user",
                content=TextContent(body="Run it"),
            )
        )
        timeline = await kit.get_timeline("room-1", limit=20)

        starts = [event for event in timeline if event.type == EventType.TOOL_CALL_START]
        ends = [event for event in timeline if event.type == EventType.TOOL_CALL_END]
        assert len(starts) == 1
        assert len(ends) == 1
        assert isinstance(ends[0].content, ToolCallContent)
        assert ends[0].content.tool_id == "tool-1"
        assert ends[0].content.status == "failed"
        assert ends[0].index > starts[0].index
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


class TestRoomCatchUp:
    """RFC §19.3.2 — an agent skipped while unaddressed catches up from the timeline.

    An ACP session holds its history inside the agent's process, so what it was
    not told is gone for good. It reads the room's tail at the moment it *is*
    asked to act, bounded, and never past what visibility would have delivered.
    """

    @staticmethod
    def _context(*recent: RoomEvent, extra_bindings: list[ChannelBinding] | None = None) -> Any:
        return RoomContext(
            room=Room(id="room-1"),
            bindings=[
                _binding(),
                ChannelBinding(
                    channel_id="ch1",
                    room_id="room-1",
                    channel_type=ChannelType.SMS,
                ),
                *(extra_bindings or []),
            ],
            recent_events=list(recent),
        )

    @staticmethod
    async def _prompt(channel: ACPChannel, event: RoomEvent, context: Any) -> None:
        """Run one turn to completion, the way the router consumes it."""
        output = await channel.on_event(event, _binding(), context)
        assert output.response_stream is not None
        [chunk async for chunk in output.response_stream]
        await asyncio.sleep(0)

    @staticmethod
    def _sent(connection: _FakeACPConnection, turn: int = 0) -> str:
        return str(connection.prompt_calls[turn]["prompt"][0].text)

    async def test_the_window_follows_room_history(self, tmp_path: Any) -> None:
        # What the framework loads is the largest window any bound channel
        # declares (over a floor it keeps for hooks), so the two must agree.
        default, _, _ = _channel(tmp_path, emit_updates=False)
        assert default.recent_events_window == 20
        wide, _, _ = _channel(tmp_path, emit_updates=False, room_history=120)
        assert wide.recent_events_window == 120
        off, _, _ = _channel(tmp_path, emit_updates=False, room_history=0)
        assert off.recent_events_window == 0

    async def test_a_negative_window_is_a_mistake_not_an_off_switch(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="room_history"):
            _channel(tmp_path, room_history=-1)

    async def test_cold_start_carries_the_room(self, tmp_path: Any) -> None:
        # The session opens on the first prompt, so it was born after the
        # conversation: everything visible is new to it.
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        missed = [
            make_event(room_id="room-1", body="on part sur quoi ?", index=0),
            make_event(room_id="room-1", body="j'ai ecrit hello.py", index=1),
        ]
        trigger = make_event(room_id="room-1", body="what did the others do?", index=2)
        await self._prompt(channel, trigger, self._context(*missed, trigger))

        sent = self._sent(connection)
        assert "[Room context — 2 messages you did not receive." in sent
        assert "[1] ch1: on part sur quoi ?" in sent
        assert "[2] ch1: j'ai ecrit hello.py" in sent
        assert sent.endswith("what did the others do?")
        await channel.close()

    async def test_second_turn_carries_only_the_gap(self, tmp_path: Any) -> None:
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        first = make_event(room_id="room-1", body="first request", index=0)
        await self._prompt(channel, first, self._context(first))

        missed = make_event(room_id="room-1", body="said while you were away", index=1)
        second = make_event(room_id="room-1", body="second request", index=2)
        await self._prompt(channel, second, self._context(first, missed, second))

        sent = self._sent(connection, turn=1)
        assert "said while you were away" in sent
        assert "first request" not in sent
        assert sent.endswith("second request")
        await channel.close()

    async def test_nothing_missed_sends_the_request_alone(self, tmp_path: Any) -> None:
        # An ordinary back-and-forth pays nothing for the catch-up.
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        trigger = make_event(room_id="room-1", body="just this", index=0)
        await self._prompt(channel, trigger, self._context(trigger))

        assert self._sent(connection) == "just this"
        await channel.close()

    async def test_visibility_withheld_stays_withheld(self, tmp_path: Any) -> None:
        # RFC §7.5 rule 8 — catching up is not a second door into the room.
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        hidden = make_event(
            room_id="room-1",
            body="SECRET",
            index=0,
            visibility=Visibility.TRANSPORT,
        )
        shown = make_event(room_id="room-1", body="ordinary", index=1)
        trigger = make_event(room_id="room-1", body="go", index=2)
        await self._prompt(channel, trigger, self._context(hidden, shown, trigger))

        sent = self._sent(connection)
        assert "SECRET" not in sent
        assert "ordinary" in sent
        assert "1 message you did not receive" in sent
        await channel.close()

    async def test_its_own_words_are_not_quoted_back(self, tmp_path: Any) -> None:
        # The session already holds what it said; a block headed "you did not
        # receive" is the wrong place to return it.
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        mine = make_event(
            room_id="room-1",
            channel_id="acp-agent",
            channel_type=ChannelType.AI,
            body="what I answered",
            index=0,
        )
        theirs = make_event(room_id="room-1", body="what they said", index=1)
        trigger = make_event(room_id="room-1", body="go", index=2)
        await self._prompt(channel, trigger, self._context(mine, theirs, trigger))

        sent = self._sent(connection)
        assert "what I answered" not in sent
        assert "what they said" in sent
        await channel.close()

    async def test_the_bound_says_what_it_hides(self, tmp_path: Any) -> None:
        # §19.3.2: an agent that knows it was truncated can ask for the rest.
        channel, connection, _ = _channel(tmp_path, emit_updates=False, room_history=2)
        missed = [make_event(room_id="room-1", body=f"m{i}", index=i) for i in range(5)]
        trigger = make_event(room_id="room-1", body="go", index=5)
        await self._prompt(channel, trigger, self._context(*missed, trigger))

        sent = self._sent(connection)
        assert "the 2 most recent of 5 messages you did not receive" in sent
        assert "[1] ch1: m3" in sent
        assert "[2] ch1: m4" in sent
        assert "m2" not in sent
        await channel.close()

    async def test_a_silenced_turn_keeps_the_catch_up_for_the_next(self, tmp_path: Any) -> None:
        # A muted binding has its stream closed unconsumed, so the agent was
        # told nothing — the mark must not move.
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        missed = make_event(room_id="room-1", body="said while you were away", index=0)
        silenced = make_event(room_id="room-1", body="silenced request", index=1)
        output = await channel.on_event(silenced, _binding(), self._context(missed, silenced))
        await output.response_stream.aclose()

        heard = make_event(room_id="room-1", body="heard request", index=2)
        await self._prompt(channel, heard, self._context(missed, silenced, heard))

        sent = self._sent(connection)
        assert "said while you were away" in sent
        assert "silenced request" in sent
        await channel.close()

    async def test_disabled_never_catches_up(self, tmp_path: Any) -> None:
        channel, connection, _ = _channel(tmp_path, emit_updates=False, room_history=0)
        missed = make_event(room_id="room-1", body="you missed this", index=0)
        trigger = make_event(room_id="room-1", body="go", index=1)
        await self._prompt(channel, trigger, self._context(missed, trigger))

        assert self._sent(connection) == "go"
        await channel.close()

    async def test_people_are_named_not_the_channel_they_arrived_on(self, tmp_path: Any) -> None:
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        marie = Participant(id="p-1", room_id="room-1", channel_id="ch1", display_name="Marie")
        missed = make_event(room_id="room-1", body="bonjour", index=0)
        missed = missed.model_copy(
            update={
                "source": EventSource(
                    channel_id="ch1", channel_type=ChannelType.SMS, participant_id="p-1"
                )
            }
        )
        trigger = make_event(room_id="room-1", body="go", index=1)
        context = self._context(missed, trigger)
        context.participants = [marie]
        await self._prompt(channel, trigger, context)

        assert "[1] Marie · ch1: bonjour" in self._sent(connection)
        await channel.close()

    async def test_a_closed_session_starts_over(self, tmp_path: Any) -> None:
        # The mark tracks what *this session* was told. A new session opens
        # empty and has missed everything.
        channel, connection, _ = _channel(tmp_path, emit_updates=False)
        first = make_event(room_id="room-1", body="first request", index=0)
        await self._prompt(channel, first, self._context(first))
        assert await channel.close_session("room-1") is True

        second = make_event(room_id="room-1", body="second request", index=1)
        await self._prompt(channel, second, self._context(first, second))

        assert "first request" in self._sent(connection, turn=1)
        await channel.close()
