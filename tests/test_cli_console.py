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
    print_banner,
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
        assert "● Bot" in rendered
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


class TestPinnedStreamRenderer:
    def test_flushes_completed_blocks_on_blank_line(self) -> None:
        output = StringIO()
        renderer = PinnedStreamRenderer("Bot", file=output, use_color=False)

        renderer.add_text("Para one.\n\nPar")
        rendered = output.getvalue()
        assert "● Bot" in rendered
        assert "Para one." in rendered
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
