"""Tests for CLIChannel."""

from __future__ import annotations

from io import StringIO
from unittest.mock import AsyncMock, call, patch

import pytest

from roomkit.channels._cli_markdown import MarkdownStreamRenderer
from roomkit.channels.cli import CLIChannel, _default_agent_label
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelMediaType, ChannelType, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.room import Room
from roomkit.models.streaming import ThinkingDeltaMarker


def _make_binding(channel_id: str = "cli") -> ChannelBinding:
    return ChannelBinding(
        channel_id=channel_id,
        room_id="room-1",
        channel_type=ChannelType.CLI,
        capabilities=ChannelCapabilities(media_types=[ChannelMediaType.TEXT]),
    )


def _make_context() -> RoomContext:
    return RoomContext(room=Room(id="room-1"), bindings=[_make_binding()])


def _make_event(
    channel_id: str = "agent-writer",
    body: str = "Hello from the agent",
) -> RoomEvent:
    return RoomEvent(
        room_id="room-1",
        source=EventSource(channel_id=channel_id, channel_type=ChannelType.AI),
        content=TextContent(body=body),
    )


# -- Unit tests ---------------------------------------------------------------


class TestCLIChannelBasics:
    def test_channel_type(self) -> None:
        cli = CLIChannel("cli")
        assert cli.channel_type == ChannelType.CLI

    def test_capabilities_text_only(self) -> None:
        cli = CLIChannel("cli")
        caps = cli.capabilities()
        assert ChannelMediaType.TEXT in caps.media_types

    def test_supports_streaming(self) -> None:
        cli = CLIChannel("cli")
        assert cli.supports_streaming_delivery is True

    def test_default_channel_id(self) -> None:
        cli = CLIChannel()
        assert cli.channel_id == "cli"

    def test_markdown_requires_optional_renderer(self) -> None:
        with (
            patch(
                "roomkit.channels._cli_markdown.require_markdown_support",
                side_effect=ImportError("missing"),
            ),
            pytest.raises(ImportError, match="missing"),
        ):
            CLIChannel("cli", markdown=True)


class TestHandleInbound:
    async def test_creates_room_event(self) -> None:
        cli = CLIChannel("cli")
        context = _make_context()
        msg = InboundMessage(
            channel_id="cli",
            sender_id="user",
            content=TextContent(body="hello"),
        )
        event = await cli.handle_inbound(msg, context)
        assert event.room_id == "room-1"
        assert event.source.channel_id == "cli"
        assert event.source.channel_type == ChannelType.CLI
        assert event.source.participant_id == "user"
        assert isinstance(event.content, TextContent)
        assert event.content.body == "hello"


class TestDeliver:
    async def test_prints_agent_response(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        event = _make_event(body="The article is ready.")
        binding = _make_binding()
        context = _make_context()

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            await cli.deliver(event, binding, context)
            output = mock_out.getvalue()

        assert "Writer:" in output
        assert "The article is ready." in output

    async def test_skips_own_messages(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        event = _make_event(channel_id="cli", body="user message")
        binding = _make_binding()
        context = _make_context()

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            await cli.deliver(event, binding, context)
            assert mock_out.getvalue() == ""

    async def test_skips_empty_text(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        event = _make_event(body="")
        binding = _make_binding()
        context = _make_context()

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            await cli.deliver(event, binding, context)
            assert mock_out.getvalue() == ""

    async def test_custom_agent_label(self) -> None:
        cli = CLIChannel(
            "cli",
            use_color=False,
            agent_label=lambda cid: "Bot",
        )
        event = _make_event(body="hi")
        binding = _make_binding()
        context = _make_context()

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            await cli.deliver(event, binding, context)
            assert "Bot:" in mock_out.getvalue()

    async def test_renders_complete_markdown(self) -> None:
        cli = CLIChannel(
            "cli",
            use_color=False,
            markdown=True,
            agent_label=lambda _channel_id: "Bot",
        )
        event = _make_event(body="# Result\n\n- **first**\n- second")

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            await cli.deliver(event, _make_binding(), _make_context())
            output = mock_out.getvalue()

        assert "Bot:" in output
        assert "Result" in output
        assert "first" in output
        assert "second" in output


class TestDeliverStream:
    async def test_streams_chunks_to_stdout(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        event = _make_event(body="")
        binding = _make_binding()
        context = _make_context()

        async def chunks() -> None:
            for c in ["Hello", " ", "world"]:
                yield c  # type: ignore[misc]

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            await cli.deliver_stream(chunks(), event, binding, context)  # type: ignore[arg-type]
            output = mock_out.getvalue()

        assert "Hello world" in output

    async def test_stream_skips_own_messages(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        event = _make_event(channel_id="cli", body="")
        binding = _make_binding()
        context = _make_context()

        async def chunks() -> None:
            yield "hi"  # type: ignore[misc]

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            await cli.deliver_stream(chunks(), event, binding, context)  # type: ignore[arg-type]
            assert mock_out.getvalue() == ""

    async def test_renders_tool_activity_inline(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        event = _make_event(body="")
        binding = _make_binding()
        context = _make_context()

        tool_start = RoomEvent(
            room_id="room-1",
            type=EventType.TOOL_CALL_START,
            source=event.source,
            content=ToolCallContent(
                tool_name="Read file",
                tool_id="tool-1",
                arguments={"path": "README.md"},
                status="pending",
            ),
        )
        tool_end = tool_start.model_copy(
            update={
                "type": EventType.TOOL_CALL_END,
                "content": tool_start.content.model_copy(
                    update={"status": "completed", "duration_ms": 42}
                ),
            }
        )

        async def chunks() -> None:
            yield tool_start  # type: ignore[misc]
            yield "The file is valid."  # type: ignore[misc]
            yield tool_end  # type: ignore[misc]

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            await cli.deliver_stream(chunks(), event, binding, context)  # type: ignore[arg-type]
            output = mock_out.getvalue()

        assert "🔧 Read file" in output
        assert '"path": "README.md"' in output
        assert "The file is valid." in output
        assert "✓ Read file (42 ms)" in output

    async def test_markdown_renderer_receives_every_stream_delta(self) -> None:
        cli = CLIChannel(
            "cli",
            use_color=False,
            show_thinking=True,
            markdown=True,
            agent_label=lambda _channel_id: "Bot",
        )
        event = _make_event(body="")
        tool_start = RoomEvent(
            room_id="room-1",
            type=EventType.TOOL_CALL_START,
            source=event.source,
            content=ToolCallContent(
                tool_name="Search",
                tool_id="tool-1",
                arguments={"query": "streaming"},
                status="pending",
            ),
        )

        async def chunks() -> None:
            yield ThinkingDeltaMarker(thinking="Checking")  # type: ignore[misc]
            yield "# Head"  # type: ignore[misc]
            yield "ing\n\n"  # type: ignore[misc]
            yield tool_start  # type: ignore[misc]
            yield "Done."  # type: ignore[misc]

        with patch("roomkit.channels._cli_markdown.MarkdownStreamRenderer") as renderer_type:
            renderer = renderer_type.return_value
            await cli.deliver_stream(
                chunks(),
                event,
                _make_binding(),
                _make_context(),
            )  # type: ignore[arg-type]

        assert renderer.add_text.call_args_list == [
            call("# Head"),
            call("ing\n\n"),
            call("Done."),
        ]
        renderer.add_thinking.assert_called_once_with("Checking")
        renderer.add_tool_event.assert_called_once_with(tool_start)
        renderer.close.assert_called_once_with()

    def test_markdown_live_renderer_refreshes_for_each_delta(self) -> None:
        output = StringIO()
        renderer = MarkdownStreamRenderer("Bot", file=output, use_color=False)

        renderer.add_text("# Head")
        renderer.add_text("ing")
        renderer.add_thinking("Considering tools")
        renderer.close()

        assert renderer.update_count == 3
        rendered = output.getvalue()
        assert "Heading" in rendered
        assert "Considering tools" in rendered


class TestRun:
    async def test_processes_input_and_exits_on_quit(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        inputs = iter(["hello world", "quit"])

        with patch("builtins.input", side_effect=inputs):
            await cli.run(kit, room_id="room-1")

        kit.process_inbound.assert_called_once()
        call_args = kit.process_inbound.call_args[0][0]
        assert call_args.channel_id == "cli"
        assert call_args.sender_id == "user"
        assert call_args.content.body == "hello world"

    async def test_skips_empty_lines(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        inputs = iter(["", "  ", "quit"])

        with patch("builtins.input", side_effect=inputs):
            await cli.run(kit, room_id="room-1")

        kit.process_inbound.assert_not_called()

    async def test_handles_eof(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()

        with patch("builtins.input", side_effect=EOFError):
            await cli.run(kit, room_id="room-1")

    async def test_handles_keyboard_interrupt(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            await cli.run(kit, room_id="room-1")

    async def test_custom_sender_id(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        inputs = iter(["test", "quit"])

        with patch("builtins.input", side_effect=inputs):
            await cli.run(kit, room_id="room-1", sender_id="alice")

        call_args = kit.process_inbound.call_args[0][0]
        assert call_args.sender_id == "alice"

    async def test_welcome_message(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()

        with (
            patch("builtins.input", side_effect=EOFError),
            patch("builtins.print") as mock_print,
        ):
            await cli.run(kit, room_id="room-1", welcome="Welcome!")

        mock_print.assert_any_call("Welcome!")


class TestDefaultAgentLabel:
    def test_strips_agent_prefix(self) -> None:
        assert _default_agent_label("agent-researcher") == "Researcher"

    def test_handles_underscores(self) -> None:
        assert _default_agent_label("agent-content_writer") == "Content Writer"

    def test_handles_hyphens(self) -> None:
        assert _default_agent_label("agent-code-reviewer") == "Code Reviewer"

    def test_no_prefix(self) -> None:
        assert _default_agent_label("writer") == "Writer"

    def test_plain_id(self) -> None:
        assert _default_agent_label("ai") == "Ai"
