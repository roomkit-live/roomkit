"""Tests for the pinned-bar console shell (roomkit.console._shell).

Drives a real PromptSession through prompt_toolkit's pipe input and dummy
output. prompt_toolkit is in the dev extras (console extra), so CI runs this.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from roomkit.channels.cli import CLIChannel
from roomkit.console import terminal_input
from roomkit.console._chat import BannerModel, ConsoleBannerData
from roomkit.console._shell import active_shell_app, run_console_shell


def _banner(models: list[BannerModel] | None = None) -> ConsoleBannerData:
    return ConsoleBannerData(version="0.0.0", room_id="room-1", models=models or [])


async def _run_shell(channel, kit, pipe_input, **kwargs):
    return await asyncio.wait_for(
        run_console_shell(
            channel,
            kit,
            "room-1",
            sender_id=kwargs.pop("sender_id", "user"),
            banner=kwargs.pop("banner", _banner()),
            input=pipe_input,
            output=DummyOutput(),
            **kwargs,
        ),
        timeout=5,
    )


class TestShellSubmission:
    async def test_line_submits_inbound_message(self) -> None:
        cli = CLIChannel("cli", console=True)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        with create_pipe_input() as pipe:
            pipe.send_text("hello world\n")
            pipe.send_text("quit\n")
            await _run_shell(cli, kit, pipe)

        kit.process_inbound.assert_called_once()
        message = kit.process_inbound.call_args[0][0]
        assert message.channel_id == "cli"
        assert message.sender_id == "user"
        assert message.content.body == "hello world"

    async def test_custom_sender_id(self) -> None:
        cli = CLIChannel("cli", console=True)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        with create_pipe_input() as pipe:
            pipe.send_text("test\n")
            pipe.send_text("quit\n")
            await _run_shell(cli, kit, pipe, sender_id="alice")

        assert kit.process_inbound.call_args[0][0].sender_id == "alice"

    async def test_submissions_process_sequentially(self) -> None:
        cli = CLIChannel("cli", console=True)
        started: list[str] = []
        gates: list[asyncio.Event] = [asyncio.Event(), asyncio.Event()]

        async def process(message) -> None:
            index = len(started)
            started.append(message.content.body)
            await gates[index].wait()

        kit = AsyncMock()
        kit.process_inbound = AsyncMock(side_effect=process)

        with create_pipe_input() as pipe:
            pipe.send_text("first\n")
            pipe.send_text("second\n")

            async def release() -> None:
                # Wait until the first turn is in flight, prove the second
                # has not started, then release both.
                while not started:
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.05)
                assert started == ["first"]
                gates[0].set()
                while len(started) < 2:
                    await asyncio.sleep(0.01)
                gates[1].set()
                pipe.send_text("quit\n")

            releaser = asyncio.create_task(release())
            await _run_shell(cli, kit, pipe)
            await releaser

        assert started == ["first", "second"]

    async def test_content_factory_none_skips_line(self) -> None:
        cli = CLIChannel("cli", console=True)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        with create_pipe_input() as pipe:
            pipe.send_text("/local-command\n")
            pipe.send_text("quit\n")
            await _run_shell(cli, kit, pipe, content_factory=lambda _line: None)

        kit.process_inbound.assert_not_called()


class TestShellShutdown:
    async def test_quit_cancels_in_flight_and_exits(self) -> None:
        cli = CLIChannel("cli", console=True)
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def hang(_message) -> None:
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        kit = AsyncMock()
        kit.process_inbound = AsyncMock(side_effect=hang)

        with create_pipe_input() as pipe:
            pipe.send_text("long task\n")

            async def quit_once_started() -> None:
                await asyncio.wait_for(started.wait(), 2)
                pipe.send_text("quit\n")

            quitter = asyncio.create_task(quit_once_started())
            await _run_shell(cli, kit, pipe)
            await quitter

        assert cancelled.is_set()
        assert cli._pinned_shell_active is False
        assert active_shell_app() is None

    async def test_queued_submissions_dropped_on_quit(self) -> None:
        cli = CLIChannel("cli", console=True)
        started: list[str] = []
        first_started = asyncio.Event()

        async def hang(message) -> None:
            started.append(message.content.body)
            first_started.set()
            await asyncio.sleep(3600)

        kit = AsyncMock()
        kit.process_inbound = AsyncMock(side_effect=hang)

        with create_pipe_input() as pipe:
            pipe.send_text("first\n")
            pipe.send_text("second\n")

            async def quit_once_started() -> None:
                await asyncio.wait_for(first_started.wait(), 2)
                pipe.send_text("quit\n")

            quitter = asyncio.create_task(quit_once_started())
            await _run_shell(cli, kit, pipe)
            await quitter

        assert started == ["first"]

    async def test_eof_exits_cleanly(self) -> None:
        cli = CLIChannel("cli", console=True)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        with create_pipe_input() as pipe:
            pipe.close()
            await _run_shell(cli, kit, pipe)

        kit.process_inbound.assert_not_called()
        assert cli._pinned_shell_active is False

    async def test_shell_sets_and_clears_state(self) -> None:
        cli = CLIChannel("cli", console=True)
        seen: dict[str, object] = {}

        async def snapshot(_message) -> None:
            seen["active"] = cli._pinned_shell_active
            seen["app"] = active_shell_app()

        kit = AsyncMock()
        kit.process_inbound = AsyncMock(side_effect=snapshot)

        with create_pipe_input() as pipe:
            pipe.send_text("check\n")
            pipe.send_text("quit\n")
            await _run_shell(
                cli,
                kit,
                pipe,
                banner=_banner([BannerModel(channel_id="ai", model="mock", provider="Mock")]),
            )

        assert seen["active"] is True
        assert seen["app"] is not None
        assert cli._pinned_shell_active is False
        assert active_shell_app() is None


class TestTerminalInput:
    async def test_fallback_without_active_shell(self) -> None:
        assert active_shell_app() is None
        with patch("builtins.input", return_value="y") as mock_input:
            answer = await terminal_input("Allow? ")
        assert answer == "y"
        mock_input.assert_called_once_with("Allow? ")


class TestUserLineEcho:
    async def test_submitted_line_echoed_with_prompt_prefix(self) -> None:
        cli = CLIChannel("cli", console=True, use_color=False)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        with (
            create_pipe_input() as pipe,
            patch("roomkit.console._shell.print_user_line") as mock_echo,
        ):
            pipe.send_text("hello there\n")
            pipe.send_text("quit\n")
            await _run_shell(cli, kit, pipe)

        echoed = [call.args[0] for call in mock_echo.call_args_list]
        assert echoed == ["hello there", "quit"]
        assert mock_echo.call_args_list[0].kwargs["prompt"] == cli._prompt


class TestPinToBottom:
    def test_writes_cursor_position_escape(self) -> None:
        from io import StringIO
        from types import SimpleNamespace

        from prompt_toolkit.data_structures import Size

        from roomkit.console._shell import _pin_to_bottom

        fake_session = SimpleNamespace(
            output=SimpleNamespace(get_size=lambda: Size(rows=50, columns=120))
        )
        with patch("sys.stdout", new_callable=StringIO) as out:
            _pin_to_bottom(fake_session)  # type: ignore[arg-type]
        assert out.getvalue() == "\x1b[50;1H"

    async def test_not_written_when_test_output_injected(self) -> None:
        # With an injected output (tests, embedding), the escape must not
        # leak to the process stdout.
        cli = CLIChannel("cli", console=True)
        kit = AsyncMock()
        kit.process_inbound = AsyncMock()

        with (
            create_pipe_input() as pipe,
            patch("roomkit.console._shell._pin_to_bottom") as mock_pin,
        ):
            pipe.send_text("quit\n")
            await _run_shell(cli, kit, pipe)

        mock_pin.assert_not_called()
