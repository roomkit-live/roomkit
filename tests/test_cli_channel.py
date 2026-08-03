"""Tests for CLIChannel."""

from __future__ import annotations

from io import StringIO
from unittest.mock import AsyncMock, call, patch

import pytest

from roomkit.channels._cli_markdown import MarkdownStreamRenderer
from roomkit.channels.cli import CLIChannel, _default_agent_label, _speaker_label
from roomkit.core.framework import RoomKit
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import (
    ChannelMediaType,
    ChannelType,
    EventType,
    HookTrigger,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.models.identity import IdentityHookResult, IdentityResult
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.models.streaming import ThinkingDeltaMarker
from tests.test_framework import SimpleChannel
from tests.test_identity_pipeline import CountingIdentityResolver


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

    def test_console_requires_optional_renderer(self) -> None:
        with (
            patch(
                "roomkit.channels._cli_markdown.require_console_support",
                side_effect=ImportError("missing"),
            ),
            pytest.raises(ImportError, match="missing"),
        ):
            CLIChannel("cli", console=True)

    def test_console_swaps_default_prompt(self) -> None:
        assert CLIChannel("cli", console=True)._prompt == "❯ "

    def test_console_keeps_custom_prompt(self) -> None:
        assert CLIChannel("cli", console=True, prompt=">> ")._prompt == ">> "

    def test_classic_mode_keeps_default_prompt(self) -> None:
        assert CLIChannel("cli")._prompt == "You: "


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


class TestConsoleMode:
    async def test_console_renderer_receives_every_stream_delta(self) -> None:
        cli = CLIChannel(
            "cli",
            use_color=False,
            show_thinking=True,
            console=True,
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

        with patch("roomkit.console._chat.ConsoleStreamRenderer") as renderer_type:
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

    async def test_console_hides_thinking_when_disabled(self) -> None:
        cli = CLIChannel("cli", use_color=False, console=True)

        async def chunks() -> None:
            yield ThinkingDeltaMarker(thinking="Secret")  # type: ignore[misc]
            yield "Answer"  # type: ignore[misc]

        with patch("roomkit.console._chat.ConsoleStreamRenderer") as renderer_type:
            renderer = renderer_type.return_value
            await cli.deliver_stream(
                chunks(),
                _make_event(body=""),
                _make_binding(),
                _make_context(),
            )  # type: ignore[arg-type]

        renderer.add_thinking.assert_not_called()
        renderer.add_text.assert_called_once_with("Answer")

    async def test_pinned_shell_selects_pinned_renderer(self) -> None:
        cli = CLIChannel("cli", use_color=False, console=True)
        cli._pinned_shell_active = True
        cli._shell_width = 100

        async def chunks() -> None:
            yield "Answer"  # type: ignore[misc]

        with patch("roomkit.console._chat.PinnedStreamRenderer") as renderer_type:
            renderer = renderer_type.return_value
            await cli.deliver_stream(
                chunks(),
                _make_event(body=""),
                _make_binding(),
                _make_context(),
            )  # type: ignore[arg-type]

        renderer_type.assert_called_once()
        assert renderer_type.call_args.kwargs["width"] == 100
        renderer.add_text.assert_called_once_with("Answer")
        renderer.close.assert_called_once_with()

    async def test_console_deliver_renders_message(self) -> None:
        cli = CLIChannel(
            "cli",
            use_color=False,
            console=True,
            agent_label=lambda _channel_id: "Concierge",
        )
        event = _make_event(body="All set.")

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            await cli.deliver(event, _make_binding(), _make_context())
            output = mock_out.getvalue()

        assert "@concierge" in output  # handle above the answer
        assert "● All set." in output  # marker in front of the answer


class TestSpeakerLabel:
    """Who the transcript names, when a room holds people as well as agents."""

    def _context(self, *participants: Participant) -> RoomContext:
        return RoomContext(room=Room(id="room-1"), participants=list(participants))

    def _from(self, channel_id: str, participant_id: str | None = None) -> RoomEvent:
        return RoomEvent(
            room_id="room-1",
            type=EventType.MESSAGE,
            source=EventSource(
                channel_id=channel_id,
                channel_type=ChannelType.SMS,
                participant_id=participant_id,
            ),
            content=TextContent(body="hi"),
        )

    def test_a_person_is_named_with_the_channel_they_speak_through(self) -> None:
        marie = Participant(id="p-1", room_id="room-1", channel_id="sms", display_name="Marie")
        label = _speaker_label(
            self._from("sms", "p-1"), self._context(marie), _default_agent_label
        )
        assert label == "Marie · sms"

    def test_two_people_on_one_channel_are_told_apart(self) -> None:
        # The reason this exists: the channel id names neither of them.
        marie = Participant(id="p-1", room_id="room-1", channel_id="sms", display_name="Marie")
        jean = Participant(id="p-2", room_id="room-1", channel_id="sms", display_name="Jean")
        ctx = self._context(marie, jean)
        assert _speaker_label(self._from("sms", "p-1"), ctx, _default_agent_label) == "Marie · sms"
        assert _speaker_label(self._from("sms", "p-2"), ctx, _default_agent_label) == "Jean · sms"

    def test_resolution_also_works_through_the_identity_id(self) -> None:
        # The identity pipeline stamps an Identity.id, not a Participant.id.
        marie = Participant(
            id="p-1",
            room_id="room-1",
            channel_id="sms",
            display_name="Marie",
            identity_id="identity-9",
        )
        label = _speaker_label(
            self._from("sms", "identity-9"), self._context(marie), _default_agent_label
        )
        assert label == "Marie · sms"

    def test_an_unnamed_participant_shows_what_is_known(self) -> None:
        unknown = Participant(id="+15551234567", room_id="room-1", channel_id="sms")
        label = _speaker_label(
            self._from("sms", "+15551234567"), self._context(unknown), _default_agent_label
        )
        assert label == "+15551234567 · sms"

    def test_an_agent_keeps_its_channel_label(self) -> None:
        # No participant, nothing changes — this is every room that works today.
        label = _speaker_label(self._from("claude-code"), self._context(), _default_agent_label)
        assert label == "Claude Code"

    def test_an_unknown_participant_id_falls_back(self) -> None:
        label = _speaker_label(self._from("sms", "ghost"), self._context(), _default_agent_label)
        assert label == "Sms"


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

    async def test_command_is_awaited_by_the_loop(self) -> None:
        # The classic loop reads stdin itself, so a command that prompts must
        # run *between* reads — awaited here, never spawned beside the loop.
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()
        order: list[str] = []

        async def agents(argument: str) -> None:
            order.append(f"command:{argument}")

        def reader(_prompt: str = "") -> str:
            order.append("read")
            return next(inputs)

        inputs = iter(["/agents pick", "hello", "quit"])

        with patch("builtins.input", side_effect=reader):
            await cli.run(kit, room_id="room-1", commands={"/agents": agents})

        assert order == ["read", "command:pick", "read", "read"]
        assert kit.process_inbound.call_args[0][0].content.body == "hello"

    async def test_command_failure_does_not_end_the_loop(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        async def boom(_argument: str) -> None:
            raise RuntimeError("nope")

        with patch("builtins.input", side_effect=iter(["/boom", "hello", "quit"])):
            await cli.run(kit, room_id="room-1", commands={"/boom": boom})

        assert kit.process_inbound.call_args[0][0].content.body == "hello"

    async def test_addressed_to_names_the_recipient(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()
        seen: list[str] = []

        def address(line: str) -> list[str]:
            seen.append(line)
            return ["codex"]

        with patch("builtins.input", side_effect=iter(["review it", "quit"])):
            await cli.run(kit, room_id="room-1", addressed_to=address)

        assert seen == ["review it"]  # the hook sees the submitted line
        assert kit.process_inbound.call_args[0][0].addressed_to == ["codex"]

    async def test_no_hook_leaves_the_message_unaddressed(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        with patch("builtins.input", side_effect=iter(["hello", "quit"])):
            await cli.run(kit, room_id="room-1")

        assert kit.process_inbound.call_args[0][0].addressed_to is None

    async def test_address_hook_runs_after_content_factory(self) -> None:
        # An "@agent ..." line moves the focus in content_factory; the
        # address must reflect that, so ordering is part of the contract.
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()
        focus = {"agent": "claude-code"}

        def factory(line: str):
            if line.startswith("@"):
                target, _, rest = line[1:].partition(" ")
                focus["agent"] = target
                return TextContent(body=rest)
            return TextContent(body=line)

        with patch("builtins.input", side_effect=iter(["@codex go", "quit"])):
            await cli.run(
                kit,
                room_id="room-1",
                content_factory=factory,
                addressed_to=lambda _line: [focus["agent"]],
            )

        message = kit.process_inbound.call_args[0][0]
        assert message.content.body == "go"
        assert message.addressed_to == ["codex"]

    async def test_visibility_scopes_the_question_and_its_answer(self) -> None:
        # A scope that hid the question and published the answer would be
        # worse than none, so the hook sets both.
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        with patch("builtins.input", side_effect=iter(["private", "quit"])):
            await cli.run(kit, room_id="room-1", visibility=lambda _l: ["cli", "codex"])

        message = kit.process_inbound.call_args[0][0]
        assert message.visibility == "cli,codex"
        assert message.response_visibility == "cli,codex"

    async def test_visibility_accepts_a_keyword(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        with patch("builtins.input", side_effect=iter(["hey", "quit"])):
            await cli.run(kit, room_id="room-1", visibility=lambda _l: "transport")

        assert kit.process_inbound.call_args[0][0].visibility == "transport"

    async def test_no_visibility_hook_leaves_everything_visible(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        with patch("builtins.input", side_effect=iter(["hey", "quit"])):
            await cli.run(kit, room_id="room-1")

        message = kit.process_inbound.call_args[0][0]
        assert message.visibility == "all"
        assert message.response_visibility is None

    async def test_visibility_can_decline_per_line(self) -> None:
        cli = CLIChannel("cli", use_color=False)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()
        seen: list[str] = []

        def scope(line: str):
            seen.append(line)
            return ["cli", "codex"] if line.startswith("/private") else None

        with patch("builtins.input", side_effect=iter(["public one", "quit"])):
            await cli.run(kit, room_id="room-1", visibility=scope)

        assert seen == ["public one"]
        assert kit.process_inbound.call_args[0][0].visibility == "all"

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

    async def test_console_run_prints_banner_and_notes(self) -> None:
        cli = CLIChannel("cli", use_color=False, console=True)
        kit = AsyncMock()
        kit.list_bindings = AsyncMock(return_value=[])
        kit.channels = {}

        with (
            patch("builtins.input", side_effect=EOFError),
            patch("sys.stdout", new_callable=StringIO) as mock_out,
            patch("roomkit.console._shell.run_console_shell") as mock_shell,
        ):
            await cli.run(kit, room_id="room-1", welcome="Type 'quit' to exit.")
            output = mock_out.getvalue()

        assert "RoomKit v" in output
        assert "room-1" in output
        assert "Type 'quit' to exit." in output
        # Non-TTY stdout (StringIO) → classic fallback, never the shell.
        mock_shell.assert_not_called()


class TestTheTerminalNamesItsOwnSender:
    """The human at the terminal is named by the room, not addressed (RFC §11.6).

    ``run()`` defaults ``sender_id`` to ``"user"`` and calls it a Participant ID.
    No resolver matches that, so resolving it returns UNKNOWN per line typed and
    the standard refusal hook makes everything typed at the keyboard disappear.
    """

    def test_the_channel_declares_it(self) -> None:
        assert CLIChannel("cli").sender_is_participant is True

    async def test_a_hook_refusing_unknown_senders_does_not_swallow_what_is_typed(
        self,
    ) -> None:
        resolver = CountingIdentityResolver()
        kit = RoomKit(identity_resolver=resolver)
        kit.register_channel(CLIChannel("cli"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "cli")

        @kit.identity_hook(HookTrigger.ON_IDENTITY_UNKNOWN)
        async def refuse(
            event: RoomEvent, context: RoomContext, id_result: IdentityResult
        ) -> IdentityHookResult:
            return IdentityHookResult.reject("unknown sender")

        for line in ("hello", "still here", "and again"):
            result = await kit.process_inbound(
                InboundMessage(
                    channel_id="cli",
                    sender_id="user",
                    content=TextContent(body=line),
                )
            )
            assert not result.blocked
            assert result.event is not None
            assert result.event.source.participant_id == "user"

        assert resolver.calls == []
        stored = await kit.store.list_events("r1")
        assert [e.content.body for e in stored if isinstance(e.content, TextContent)] == [
            "hello",
            "still here",
            "and again",
        ]

    async def test_a_channel_that_carries_addresses_still_resolves(self) -> None:
        """The declaration is the CLI's, not the kit's."""
        resolver = CountingIdentityResolver()
        kit = RoomKit(identity_resolver=resolver)
        kit.register_channel(CLIChannel("cli"))
        kit.register_channel(SimpleChannel("sms"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "cli")
        await kit.attach_channel("r1", "sms")

        await kit.process_inbound(
            InboundMessage(channel_id="cli", sender_id="user", content=TextContent(body="hi"))
        )
        await kit.process_inbound(
            InboundMessage(channel_id="sms", sender_id="+15551234", content=TextContent(body="hi"))
        )

        assert resolver.calls == ["+15551234"]


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
