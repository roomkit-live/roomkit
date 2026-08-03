"""Tests for the console's inline picker (roomkit.console.terminal_select).

The interesting case is the nested one: the picker runs *while* the pinned
shell owns the terminal, borrowing its input and output.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from roomkit.channels.cli import CLIChannel
from roomkit.console import _terminal, terminal_select
from roomkit.console._chat import ConsoleBannerData
from roomkit.console._shell import run_console_shell

DOWN = "\x1b[B"
UP = "\x1b[A"
ENTER = "\r"
ESCAPE = "\x1b"

OPTIONS = [("claude-code", "Claude Code"), ("codex", "Codex"), ("gemini", "Gemini")]


async def _pick_under_shell(keys: str, **kwargs) -> str | None:
    """Run the picker from inside a live shell turn, driven by *keys*.

    The keys are sent only once the picker exists: until then the shell's own
    prompt owns the input, and would read them as a typed line.
    """
    cli = CLIChannel("cli", console=True)
    kit = AsyncMock()
    captured: dict[str, str | None] = {}
    picker_up = asyncio.Event()
    real_build = _terminal._build_picker

    def spy_build(*args, **kwargs_):
        app = real_build(*args, **kwargs_)
        picker_up.set()
        return app

    async def process(_message) -> None:
        with patch.object(_terminal, "_build_picker", spy_build):
            captured["result"] = await terminal_select(OPTIONS, **kwargs)

    kit.process_inbound = AsyncMock(side_effect=process)

    with create_pipe_input() as pipe:
        pipe.send_text("go\n")

        async def send_keys() -> None:
            await asyncio.wait_for(picker_up.wait(), timeout=5)
            await asyncio.sleep(0.05)  # let run_async attach its reader
            pipe.send_text(keys)

        keyer = asyncio.create_task(send_keys())

        async def quit_when_done() -> None:
            for _ in range(200):
                await asyncio.sleep(0.01)
                if "result" in captured:
                    break
            pipe.send_text("quit\n")

        quitter = asyncio.create_task(quit_when_done())
        await asyncio.wait_for(
            run_console_shell(
                cli,
                kit,
                "room-1",
                sender_id="user",
                banner=ConsoleBannerData(version="0.0.0", room_id="room-1"),
                input=pipe,
                output=DummyOutput(),
            ),
            timeout=10,
        )
        await quitter
        await keyer
    return captured.get("result")


class TestUnderTheShell:
    async def test_arrow_keys_and_enter_choose(self) -> None:
        assert await _pick_under_shell(DOWN + ENTER) == "codex"

    async def test_enter_takes_the_default(self) -> None:
        assert await _pick_under_shell(ENTER, default="gemini") == "gemini"

    async def test_wraps_around_the_ends(self) -> None:
        # Up from the first entry lands on the last.
        assert await _pick_under_shell(UP + ENTER) == "gemini"

    async def test_escape_cancels(self) -> None:
        assert await _pick_under_shell(ESCAPE) is None

    async def test_unknown_default_starts_at_the_first_option(self) -> None:
        assert await _pick_under_shell(ENTER, default="nobody") == "claude-code"


class TestWithoutAShell:
    async def test_numbered_fallback_reads_a_choice(self) -> None:
        with patch("builtins.input", return_value="2") as mock_input:
            assert await terminal_select(OPTIONS, title="Pick one") == "codex"
        mock_input.assert_called_once()

    async def test_empty_answer_takes_the_default(self) -> None:
        with patch("builtins.input", return_value=""):
            assert await terminal_select(OPTIONS, default="gemini") == "gemini"

    async def test_out_of_range_answer_cancels(self) -> None:
        with patch("builtins.input", return_value="9"):
            assert await terminal_select(OPTIONS) is None

    async def test_eof_cancels(self) -> None:
        with patch("builtins.input", side_effect=EOFError):
            assert await terminal_select(OPTIONS) is None

    async def test_bare_strings_are_their_own_values(self) -> None:
        with patch("builtins.input", return_value="1"):
            assert await terminal_select(["alpha", "beta"]) == "alpha"

    async def test_no_options_returns_none_without_asking(self) -> None:
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            assert await terminal_select([]) is None
