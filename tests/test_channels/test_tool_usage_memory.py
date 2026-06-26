"""Per-conversation tool-usage memory: the store and its wiring into context.

Two concerns:
* the store (:class:`ToolUsageMemory`) — recording, the digest, the re-reveal
  name set, dedup, bounding, infra-tool exclusion, room scoping;
* the wiring in ``_build_context`` — the digest lands in the system prompt, and
  a previously-called tool is re-revealed under Tool Search (selectively).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from roomkit.channels._tool_usage import ToolUsageMemory
from roomkit.channels.ai import AIChannel, _current_loop_ctx, _ToolLoopContext
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.context import RoomContext
from roomkit.models.enums import (
    ChannelCategory,
    ChannelDirection,
    ChannelMediaType,
    ChannelType,
)
from roomkit.models.event import EventSource, RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider

# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class TestToolUsageMemory:
    def test_records_and_renders_digest_with_args_and_result(self) -> None:
        mem = ToolUsageMemory()
        mem.record("r1", "SpotifyPlayback", {"action": "get"}, '{"artist": "Zach Bryan"}')
        digest = mem.render_digest("r1")
        assert digest is not None
        assert "SpotifyPlayback" in digest
        assert "action='get'" in digest
        assert "Zach Bryan" in digest

    def test_digest_disclaims_it_is_not_the_full_toolset(self) -> None:
        """The digest must NOT read as the agent's complete capability, or it
        makes the model deny tools it hasn't used yet (observed: it refused a
        Spotify search it actually had). It points back to find_tools."""
        mem = ToolUsageMemory()
        mem.record("r1", "SpotifyPlayback", {"action": "get"}, "track")
        digest = mem.render_digest("r1") or ""
        assert "find_tools" in digest
        assert "not your full toolset" in digest.lower()

    def test_tool_names_returns_called_tools(self) -> None:
        mem = ToolUsageMemory()
        mem.record("r1", "SpotifyPlayback", {"action": "skip"}, "Skipped.")
        mem.record("r1", "web_search", {"query": "x"}, "1 result")
        assert mem.tool_names("r1") == {"SpotifyPlayback", "web_search"}

    def test_empty_room_yields_no_digest_no_names(self) -> None:
        mem = ToolUsageMemory()
        assert mem.render_digest("r1") is None
        assert mem.tool_names("r1") == set()
        assert mem.render_digest(None) is None
        assert mem.tool_names(None) == set()

    def test_infra_tools_are_not_recorded(self) -> None:
        mem = ToolUsageMemory()
        mem.record("r1", "find_tools", {"query": "music"}, "{}")
        mem.record("r1", "list_tools", {}, "{}")
        mem.record("r1", "read_stored_result", {"result_id": "x"}, "{}")
        assert mem.tool_names("r1") == set()
        assert mem.render_digest("r1") is None

    def test_consecutive_identical_calls_collapse(self) -> None:
        mem = ToolUsageMemory()
        mem.record("r1", "SpotifyPlayback", {"action": "get"}, "track A")
        mem.record("r1", "SpotifyPlayback", {"action": "get"}, "track B")  # newer wins
        digest = mem.render_digest("r1")
        assert digest is not None
        assert digest.count("SpotifyPlayback") == 1
        assert "track B" in digest and "track A" not in digest

    def test_digest_bounded_by_recent_calls(self) -> None:
        mem = ToolUsageMemory(digest_max_calls=3)
        for i in range(5):
            mem.record("r1", f"tool_{i}", {"i": i}, str(i))
        digest = mem.render_digest("r1") or ""
        assert "tool_2" in digest and "tool_3" in digest and "tool_4" in digest
        assert "tool_0" not in digest and "tool_1" not in digest  # oldest calls dropped

    def test_reveal_bounded_by_distinct_tools(self) -> None:
        mem = ToolUsageMemory(reveal_max_tools=3)
        for i in range(5):
            mem.record("r1", f"tool_{i}", {"i": i}, str(i))
        assert mem.tool_names("r1") == {"tool_2", "tool_3", "tool_4"}  # oldest tools dropped

    def test_reveal_outlives_the_digest_window(self) -> None:
        """The reveal set is distinct-tool based, so a tool used early stays
        callable even after newer calls pushed it out of the call-based digest —
        capability must not expire just because the transcript scrolled."""
        mem = ToolUsageMemory(digest_max_calls=2, reveal_max_tools=10)
        mem.record("r1", "SpotifyPlayback", {"action": "skip"}, "ok")
        for i in range(3):
            mem.record("r1", "web_search", {"q": i}, "results")
        assert "SpotifyPlayback" not in (mem.render_digest("r1") or "")  # off the digest
        assert "SpotifyPlayback" in mem.tool_names("r1")  # still revealable

    def test_rooms_are_isolated(self) -> None:
        mem = ToolUsageMemory()
        mem.record("r1", "tool_a", {}, "a")
        mem.record("r2", "tool_b", {}, "b")
        assert mem.tool_names("r1") == {"tool_a"}
        assert mem.tool_names("r2") == {"tool_b"}

    def test_long_result_is_truncated(self) -> None:
        mem = ToolUsageMemory()
        mem.record("r1", "dump", {}, "x" * 500)
        digest = mem.render_digest("r1")
        assert digest is not None
        assert "…" in digest
        assert "x" * 500 not in digest


# ---------------------------------------------------------------------------
# Wiring into _build_context
# ---------------------------------------------------------------------------


def _binding(tools: list[dict] | None = None) -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        direction=ChannelDirection.BIDIRECTIONAL,
        capabilities=ChannelCapabilities(media_types=[ChannelMediaType.TEXT]),
        metadata={"tools": tools or []},
    )


def _context() -> RoomContext:
    return RoomContext(room=Room(id="r1"), bindings=[_binding()])


def _event() -> RoomEvent:
    return RoomEvent(
        room_id="r1",
        source=EventSource(channel_id="user", channel_type=ChannelType.SMS, provider="mock"),
        content=TextContent(body="next song"),
    )


def _channel(tool_search: bool | None = None) -> AIChannel:
    return AIChannel(
        "ai1",
        provider=MockAIProvider(ai_responses=[AIResponse(content="ok", tool_calls=[])]),
        tool_handler=AsyncMock(),
        tool_search=tool_search,
    )


_CATALOGUE = [
    {"name": "SpotifyPlayback", "description": "Control playback"},
    {"name": "unrelated_tool", "description": "Something else"},
]


class TestToolUsageInContext:
    async def test_digest_injected_into_system_prompt(self) -> None:
        ch = _channel()
        ch._tool_usage.record("r1", "SpotifyPlayback", {"action": "get"}, '{"artist": "Zach"}')
        _current_loop_ctx.set(_ToolLoopContext(room_id="r1"))
        try:
            ctx = await ch._build_context(_event(), _binding(_CATALOGUE), _context())
        finally:
            _current_loop_ctx.set(None)
        assert "Tools you've already used here" in (ctx.system_prompt or "")
        assert "SpotifyPlayback" in (ctx.system_prompt or "")

    async def test_called_tool_is_revealed_under_tool_search(self) -> None:
        """With Tool Search ON, a tool the agent already used stays callable,
        while an unused catalogue tool stays hidden."""
        ch = _channel(tool_search=True)
        ch._tool_usage.record("r1", "SpotifyPlayback", {"action": "skip"}, "Skipped.")
        _current_loop_ctx.set(_ToolLoopContext(room_id="r1"))
        try:
            ctx = await ch._build_context(_event(), _binding(_CATALOGUE), _context())
        finally:
            _current_loop_ctx.set(None)
        names = {t.name for t in ctx.tools}
        assert "SpotifyPlayback" in names  # re-revealed because it was used
        assert "unrelated_tool" not in names  # still hidden behind Tool Search

    async def test_unused_tool_stays_hidden_without_prior_call(self) -> None:
        """Control: with Tool Search ON and nothing recorded, the catalogue is
        hidden — proves the reveal in the test above comes from usage memory."""
        ch = _channel(tool_search=True)
        _current_loop_ctx.set(_ToolLoopContext(room_id="r1"))
        try:
            ctx = await ch._build_context(_event(), _binding(_CATALOGUE), _context())
        finally:
            _current_loop_ctx.set(None)
        names = {t.name for t in ctx.tools}
        assert "SpotifyPlayback" not in names
        assert "unrelated_tool" not in names

    async def test_reveal_survives_the_for_loop_refilter(self) -> None:
        """Regression: the per-round re-filter runs under the for_loop CHILD ctx,
        which resets ``revealed_tools`` but inherits ``sticky_tools``. The used-tool
        re-exposition must survive that child re-filter — otherwise the model has
        to re-run find_tools every turn (the bug this guards). ``_build_context``
        alone passing is NOT enough: production overwrites round 0's tools by
        re-filtering ``all_context_tools`` under the child ctx."""
        ch = _channel(tool_search=True)
        ch._tool_usage.record("r1", "SpotifyPlayback", {"action": "skip"}, "Skipped.")
        parent = _ToolLoopContext(room_id="r1")
        _current_loop_ctx.set(parent)
        try:
            await ch._build_context(_event(), _binding(_CATALOGUE), _context())
            assert "SpotifyPlayback" in parent.sticky_tools  # seeded on the parent
            child = _ToolLoopContext.for_loop(parent, "r1")
            assert "SpotifyPlayback" in child.sticky_tools  # inherited by the loop
            _current_loop_ctx.set(child)
            kept = {t.name for t in ch._apply_tool_filters(parent.all_context_tools)}
        finally:
            _current_loop_ctx.set(None)
        assert "SpotifyPlayback" in kept  # callable under the child re-filter
        assert "unrelated_tool" not in kept
