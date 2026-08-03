"""Tests for the CLI console presenter (roomkit.console._chat).

Requires the ``rich`` library, like ``tests/test_console.py`` — rich is in
the dev extras.
"""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from roomkit import __version__
from roomkit.channels.ai import AIChannel
from roomkit.channels.cli import CLIChannel
from roomkit.console._chat import (
    ConsoleStreamRenderer,
    PinnedStreamRenderer,
    collect_banner_data,
    format_tool_start_line,
    format_turn_footer,
    print_banner,
    tool_result_preview,
)
from roomkit.core.framework import RoomKit
from roomkit.models.enums import ChannelCategory, EventType
from roomkit.models.event import EventSource, RoomEvent, TextContent, ToolCallContent
from roomkit.providers.ai import MockAIProvider
from tests.test_framework import SimpleChannel


async def _kit_with_room(*channels_and_categories) -> RoomKit:
    kit = RoomKit()
    await kit.create_room(room_id="room-1")
    for channel, category in channels_and_categories:
        kit.register_channel(channel)
        await kit.attach_channel("room-1", channel.channel_id, category=category)
    return kit


class TestCollectBannerData:
    async def test_ai_channel_yields_model_line(self) -> None:
        provider = MockAIProvider()
        kit = await _kit_with_room(
            (CLIChannel("cli"), ChannelCategory.TRANSPORT),
            (AIChannel("ai", provider=provider), ChannelCategory.INTELLIGENCE),
        )

        data = await collect_banner_data(kit, "room-1")

        assert data.version == __version__
        assert data.room_id == "room-1"
        assert [m.model for m in data.models] == ["mock"]
        assert data.models[0].provider == "MockAIProvider"
        assert {ch.channel_id for ch in data.channels} == {"cli", "ai"}

    async def test_no_intelligence_binding_yields_no_models(self) -> None:
        kit = await _kit_with_room((CLIChannel("cli"), ChannelCategory.TRANSPORT))

        data = await collect_banner_data(kit, "room-1")

        assert data.models == []
        assert [ch.channel_id for ch in data.channels] == ["cli"]

    async def test_two_ai_channels_yield_two_model_lines(self) -> None:
        kit = await _kit_with_room(
            (AIChannel("ai-1", provider=MockAIProvider()), ChannelCategory.INTELLIGENCE),
            (AIChannel("ai-2", provider=MockAIProvider()), ChannelCategory.INTELLIGENCE),
        )

        data = await collect_banner_data(kit, "room-1")

        assert [m.channel_id for m in data.models] == ["ai-1", "ai-2"]

    async def test_intelligence_channel_without_model_name(self) -> None:
        # An agent-style channel (e.g. ACP) has a provider name but no model.
        agent = SimpleChannel("agent")
        agent._provider = SimpleNamespace(name="acp")
        kit = await _kit_with_room((agent, ChannelCategory.INTELLIGENCE))

        data = await collect_banner_data(kit, "room-1")

        assert len(data.models) == 1
        assert data.models[0].model is None
        assert data.models[0].provider == "acp"

    async def test_intelligence_channel_without_provider_at_all(self) -> None:
        kit = await _kit_with_room((SimpleChannel("bare"), ChannelCategory.INTELLIGENCE))

        data = await collect_banner_data(kit, "room-1")

        assert data.models == []
        assert data.channels[0].category == "intelligence"


class TestAIChannelProvider:
    def test_provider_property_returns_constructor_provider(self) -> None:
        provider = MockAIProvider()
        assert AIChannel("ai", provider=provider).provider is provider


class TestPrintBanner:
    async def test_banner_contains_version_model_and_room(self) -> None:
        kit = await _kit_with_room(
            (CLIChannel("cli"), ChannelCategory.TRANSPORT),
            (AIChannel("ai", provider=MockAIProvider()), ChannelCategory.INTELLIGENCE),
        )
        data = await collect_banner_data(kit, "room-1")

        output = StringIO()
        print_banner(data, file=output, use_color=False, notes="Type 'quit' to exit.")
        rendered = output.getvalue()

        assert f"RoomKit v{__version__}" in rendered
        assert "mock" in rendered
        assert "room-1" in rendered
        assert "ai (intelligence)" in rendered
        assert "Type 'quit' to exit." in rendered
        assert "\x1b[" not in rendered  # no ANSI escapes without color


class TestConsoleStreamRenderer:
    def test_renders_markdown_thinking_and_tool_lines(self) -> None:
        source = EventSource(channel_id="ai", channel_type="ai")
        tool_start = RoomEvent(
            room_id="room-1",
            type=EventType.TOOL_CALL_START,
            source=source,
            content=ToolCallContent(
                tool_name="search",
                tool_id="tool-1",
                arguments={"query": "roomkit"},
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

        output = StringIO()
        renderer = ConsoleStreamRenderer("Bot", file=output, use_color=False)
        renderer.add_thinking("Weighing options")
        renderer.add_text("# Head")
        renderer.add_text("ing")
        renderer.add_tool_event(tool_start)
        renderer.add_tool_event(tool_end)
        renderer.close()

        rendered = output.getvalue()
        assert renderer.update_count == 5
        assert "@bot" in rendered  # the quiet handle, not a shouted label
        assert "💭 Weighing options" in rendered
        assert "Heading" in rendered
        assert "⏺ search" in rendered
        assert "⎿ ✓ search · 42 ms" in rendered

    def test_failed_tool_line(self) -> None:
        source = EventSource(channel_id="ai", channel_type="ai")
        tool_end = RoomEvent(
            room_id="room-1",
            type=EventType.TOOL_CALL_END,
            source=source,
            content=ToolCallContent(
                tool_name="search",
                tool_id="tool-1",
                arguments={},
                status="failed",
            ),
        )

        output = StringIO()
        renderer = ConsoleStreamRenderer("Bot", file=output, use_color=False)
        renderer.add_tool_event(tool_end)
        renderer.close()

        assert "⎿ ✗ search failed" in output.getvalue()


def _tool_events() -> tuple[RoomEvent, RoomEvent]:
    source = EventSource(channel_id="ai", channel_type="ai")
    start = RoomEvent(
        room_id="room-1",
        type=EventType.TOOL_CALL_START,
        source=source,
        content=ToolCallContent(
            tool_name="search",
            tool_id="tool-1",
            arguments={"query": "roomkit"},
            status="pending",
        ),
    )
    end = start.model_copy(
        update={
            "type": EventType.TOOL_CALL_END,
            "content": start.content.model_copy(update={"status": "completed", "duration_ms": 42}),
        }
    )
    return start, end


class TestToolRendering:
    def test_start_line_without_arguments_has_no_parens(self) -> None:
        content = ToolCallContent(tool_name="Write", tool_id="t1", arguments={})
        assert format_tool_start_line(content) == "⏺ Write"

    def test_start_line_with_arguments_keeps_parens(self) -> None:
        content = ToolCallContent(tool_name="search", tool_id="t1", arguments={"q": "x"})
        assert format_tool_start_line(content) == '⏺ search({"q": "x"})'

    def test_preview_from_string_result(self) -> None:
        content = ToolCallContent(
            tool_name="Terminal",
            tool_id="t1",
            status="completed",
            result="Hello, world!\n",
        )
        assert tool_result_preview(content) == [("dim", "Hello, world!")]

    def test_preview_from_acp_diff_block(self) -> None:
        # The real ACP dump uses camelCase aliases (oldText/newText).
        content = ToolCallContent(
            tool_name="Edit",
            tool_id="t1",
            status="completed",
            result=[
                {
                    "type": "diff",
                    "path": "/tmp/hello.c",
                    "oldText": "int main() { return 1; }\n",
                    "newText": "int main() { return 0; }\n",
                }
            ],
        )
        preview = tool_result_preview(content)
        kinds = [kind for kind, _line in preview]
        assert preview[0] == ("dim", "/tmp/hello.c")
        assert "del" in kinds
        assert "add" in kinds
        assert any(line.startswith("+") and "return 0" in line for _k, line in preview)

    def test_preview_diff_tolerates_snake_case_keys(self) -> None:
        content = ToolCallContent(
            tool_name="Edit",
            tool_id="t1",
            status="completed",
            result=[{"type": "diff", "path": "/f", "old_text": "a\n", "new_text": "b\n"}],
        )
        kinds = [kind for kind, _line in tool_result_preview(content)]
        assert "add" in kinds
        assert "del" in kinds

    def test_preview_from_acp_text_content_block(self) -> None:
        content = ToolCallContent(
            tool_name="Read",
            tool_id="t1",
            status="completed",
            result=[{"type": "content", "content": {"type": "text", "text": "line one"}}],
        )
        assert tool_result_preview(content) == [("dim", "line one")]

    def test_preview_caps_lines_with_marker(self) -> None:
        content = ToolCallContent(
            tool_name="Terminal",
            tool_id="t1",
            status="completed",
            result="\n".join(f"line {i}" for i in range(12)),
        )
        preview = tool_result_preview(content)
        assert len(preview) == 6  # 5 lines + the marker
        assert preview[-1] == ("dim", "… +7 lines")

    def test_preview_uses_error_on_failure(self) -> None:
        content = ToolCallContent(
            tool_name="Terminal",
            tool_id="t1",
            status="failed",
            error="command not found: cc",
            result={"ignored": True},
        )
        assert tool_result_preview(content) == [("dim", "command not found: cc")]

    def test_preview_none_result_is_empty(self) -> None:
        content = ToolCallContent(tool_name="x", tool_id="t1", status="completed")
        assert tool_result_preview(content) == []

    def test_preview_prefers_acp_display_content_over_raw_result(self) -> None:
        # claude-agent-acp sends BOTH a raw text confirmation and the diff
        # blocks; the diff is the display payload and must win.
        content = ToolCallContent(
            tool_name="Write",
            tool_id="t1",
            status="completed",
            result="File created successfully at: /tmp/hello.rs",
            structured_content={
                "acp_content": [
                    {
                        "type": "diff",
                        "path": "/tmp/hello.rs",
                        "newText": 'fn main() { println!("Hello"); }\n',
                    }
                ]
            },
        )
        preview = tool_result_preview(content)
        assert preview[0] == ("dim", "/tmp/hello.rs")
        assert any(kind == "add" for kind, _line in preview)
        assert all("File created successfully" not in line for _k, line in preview)

    def test_sections_separated_by_blank_lines(self) -> None:
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)
        start, end_event = _tool_events()

        renderer.add_text("Intro paragraph.\n\n")
        renderer.add_tool_event(start)
        renderer.add_tool_event(end_event)
        renderer.add_text("After the tools.\n\n")
        renderer.close()

        lines = output.getvalue().splitlines()
        tool_index = next(i for i, line in enumerate(lines) if line.startswith("⏺ search"))
        assert lines[tool_index - 1] == ""  # blank before the tool block
        resumed_index = next(i for i, line in enumerate(lines) if "After the tools." in line)
        assert lines[resumed_index - 1] == ""  # blank before resumed text
        assert lines[0].startswith("@bot")  # no leading blank at turn start
        # Prose leads with the marker, before and after the tool round.
        assert lines[1].startswith("● Intro paragraph.")
        assert lines[resumed_index].startswith("● After the tools.")

    def test_pinned_renderer_shows_result_preview(self) -> None:
        source = EventSource(channel_id="ai", channel_type="ai")
        end = RoomEvent(
            room_id="room-1",
            type=EventType.TOOL_CALL_END,
            source=source,
            content=ToolCallContent(
                tool_name="Terminal",
                tool_id="t1",
                status="completed",
                duration_ms=5179,
                result="Hello, world!",
            ),
        )
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)
        renderer.add_tool_event(end)
        renderer.close()
        rendered = output.getvalue()
        assert "⎿ ✓ Terminal · 5.2s" in rendered
        assert "Hello, world!" in rendered


class TestTurnLayout:
    def test_handle_then_marked_answer(self) -> None:
        output = StringIO()
        renderer = PinnedStreamRenderer("Claude Code", file=output, use_color=False)
        renderer.add_text("Salut !\n\n")
        renderer.close()

        lines = output.getvalue().splitlines()
        assert lines[0] == "@claude code"  # a handle, not a shouted name
        assert lines[1].startswith("● Salut !")

    def test_handle_is_not_detached_from_what_it_introduces(self) -> None:
        # The turn opens on a tool, not on prose: still no blank line
        # between the handle and the block it names.
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)
        start, _end = _tool_events()
        renderer.add_tool_event(start)
        renderer.close()

        lines = output.getvalue().splitlines()
        assert lines[0] == "@bot"
        assert lines[1].startswith("⏺ search")

    def test_continuation_lines_align_under_the_text(self) -> None:
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False, width=30)
        renderer.add_text("one two three four five six seven eight nine\n\n")
        renderer.close()

        body = [line for line in output.getvalue().splitlines() if line.strip()]
        wrapped = next(line for line in body[2:] if not line.startswith(("@", "●", "  ⎿")))
        assert wrapped.startswith("  ")  # under the prose, not under the marker

    def test_footer_reports_what_the_turn_cost(self) -> None:
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)
        start, end_event = _tool_events()
        renderer.add_text("Done.\n\n")
        renderer.add_tool_event(start)
        renderer.add_tool_event(end_event)
        renderer.close()

        footer = [line for line in output.getvalue().splitlines() if line.strip()][-1]
        assert footer.startswith("  ⎿ took ")
        assert footer.endswith("· 1 tool")

    def test_silent_turn_has_no_footer(self) -> None:
        # Nothing was said; a duration line alone would be noise.
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)
        renderer.close()
        assert "took" not in output.getvalue()

    def test_footer_formatting(self) -> None:
        assert format_turn_footer(842, 0) == "  ⎿ took 842 ms"
        assert format_turn_footer(3000, 1) == "  ⎿ took 3.0s · 1 tool"
        assert format_turn_footer(150_000, 4) == "  ⎿ took 2m 30s · 4 tools"


class TestPinnedStreamRenderer:
    def test_flushes_completed_blocks_on_blank_line(self) -> None:
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)

        renderer.add_text("Para one.\n\nPar")
        rendered = output.getvalue()
        assert "@bot" in rendered
        assert "● Para one." in rendered
        assert "Par" == "Par" and "Par\n" not in rendered  # tail retained

        renderer.close()
        assert "Par" in output.getvalue().split("Para one.", 1)[1]

    def test_incomplete_tail_waits_for_close(self) -> None:
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)

        renderer.add_text("no boundary yet")
        assert output.getvalue() == ""

        renderer.close()
        assert "no boundary yet" in output.getvalue()

    def test_fence_never_split(self) -> None:
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)

        # Fenced block with an internal blank line, fed in small chunks.
        for chunk in ["```\ncode line one\n", "\n", "code line two\n", "```", "\n"]:
            renderer.add_text(chunk)
            # Nothing may flush while the fence is open: the only blank-line
            # boundaries so far are inside the fence.
            assert "code line one" not in output.getvalue()

        renderer.add_text("\nafter\n\n")
        rendered = output.getvalue()
        assert "code line one" in rendered
        assert "code line two" in rendered
        assert "after" in rendered
        renderer.close()

    def test_thinking_lines_with_single_prefix(self) -> None:
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)

        renderer.add_thinking("\nFirst thought\nSecond ")
        rendered = output.getvalue()
        assert "💭 First thought" in rendered
        assert "Second" not in rendered  # incomplete line retained

        renderer.add_thinking("thought\n")
        rendered = output.getvalue()
        assert "Second thought" in rendered
        assert rendered.count("💭") == 1

    def test_new_thinking_block_after_text_gets_prefix(self) -> None:
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)

        renderer.add_thinking("Round one\n")
        renderer.add_text("Answer part.\n\n")
        renderer.add_thinking("Round two\n")
        renderer.close()

        assert output.getvalue().count("💭") == 2

    def test_tool_lines_immediate_and_force_flush_text(self) -> None:
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)
        start, end = _tool_events()

        renderer.add_text("Looking that up")  # no boundary — normally buffered
        renderer.add_tool_event(start)
        rendered = output.getvalue()
        assert "Looking that up" in rendered  # force-flushed before the tool line
        assert "⏺ search" in rendered

        renderer.add_tool_event(end)
        assert "⎿ ✓ search · 42 ms" in output.getvalue()
        renderer.close()

    def test_message_event_not_double_rendered(self) -> None:
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)

        renderer.add_text("Streamed text.\n\n")
        message_event = RoomEvent(
            room_id="room-1",
            source=EventSource(channel_id="ai", channel_type="ai"),
            content=TextContent(body="Streamed text."),
        )
        before = output.getvalue()
        renderer.add_tool_event(message_event)
        assert output.getvalue() == before
        renderer.close()

    def test_close_idempotent(self) -> None:
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)
        renderer.add_text("tail")
        renderer.close()
        length = len(output.getvalue())
        renderer.close()
        assert len(output.getvalue()) == length
