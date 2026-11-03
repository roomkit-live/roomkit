"""Per-conversation tool-usage memory: the store and its wiring into context.

Two concerns:
* the store (:class:`ToolUsageMemory`) — recording, the re-reveal name set,
  bounding, infra-tool exclusion, room scoping, hydration;
* the wiring in ``_build_context`` — a previously-called tool is re-revealed
  under Tool Search (selectively). The calls themselves ride the history:
  ``tests/test_channels/test_tool_history.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from roomkit.channels._tool_usage import _MAX_ROOMS, ToolUsageMemory
from roomkit.channels.ai import AIChannel
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
from roomkit.providers.ai.base import (
    AIContext,
    AIResponse,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext

# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def _channel_memory(**kwargs: int) -> ToolUsageMemory:
    """The store as a channel builds it: the channel's own discovery and
    housekeeping tools (their ``in_tool_memory`` trait) are not recorded."""
    channel = AIChannel("ai", provider=MockAIProvider(responses=["ok"]))
    return ToolUsageMemory(recorded=channel._in_tool_memory, **kwargs)


class TestToolUsageMemory:
    def test_tool_names_returns_called_tools(self) -> None:
        mem = ToolUsageMemory()
        mem.record("r1", "SpotifyPlayback", {"action": "skip"}, "Skipped.")
        mem.record("r1", "web_search", {"query": "x"}, "1 result")
        assert mem.tool_names("r1") == {"SpotifyPlayback", "web_search"}

    def test_empty_room_yields_no_names(self) -> None:
        mem = ToolUsageMemory()
        assert mem.tool_names("r1") == set()
        assert mem.tool_names(None) == set()

    def test_seeding_rooms_from_nothing_stays_under_the_rooms_bound(self) -> None:
        """A room with no stored rows counts toward the bound, as roomkit-rs reads
        it (HITL d8f31c25): a channel answering without a tool in many rooms would
        otherwise grow until a tool call somewhere trims it."""
        mem = _channel_memory()
        for at in range(_MAX_ROOMS + 50):
            mem.seed(f"r{at}", [])
        assert len(mem._by_room) == _MAX_ROOMS

    def test_infra_tools_are_not_recorded(self) -> None:
        mem = _channel_memory()
        mem.record("r1", "find_tools", {"query": "music"}, "{}")
        mem.record("r1", "list_tools", {}, "{}")
        mem.record("r1", "read_stored_result", {"result_id": "x"}, "{}")
        assert mem.tool_names("r1") == set()

    def test_reveal_bounded_by_distinct_tools(self) -> None:
        mem = ToolUsageMemory(reveal_max_tools=3)
        for i in range(5):
            mem.record("r1", f"tool_{i}", {"i": i}, str(i))
        assert mem.tool_names("r1") == {"tool_2", "tool_3", "tool_4"}  # oldest tools dropped

    def test_reveal_outlives_the_replayed_calls(self) -> None:
        """The reveal set is distinct-tool based, so a tool used early stays
        callable after newer calls pushed it out of the calls the history
        replays — capability must not expire just because the transcript
        scrolled."""
        mem = ToolUsageMemory(reveal_max_tools=10)
        mem.record("r1", "SpotifyPlayback", {"action": "skip"}, "ok")
        for i in range(20):
            mem.record("r1", "web_search", {"q": i}, "results")
        assert "SpotifyPlayback" in mem.tool_names("r1")

    def test_rooms_are_isolated(self) -> None:
        mem = ToolUsageMemory()
        mem.record("r1", "tool_a", {}, "a")
        mem.record("r2", "tool_b", {}, "b")
        assert mem.tool_names("r1") == {"tool_a"}
        assert mem.tool_names("r2") == {"tool_b"}

    def test_record_revealed_feeds_reveal(self) -> None:
        """A find_tools reveal keeps the tool callable next turn ("rest of the
        session", as its description promises)."""
        mem = ToolUsageMemory()
        mem.record_revealed("r1", {"square_create-cart", "square_update-cart"})
        assert mem.tool_names("r1") == {"square_create-cart", "square_update-cart"}

    def test_record_revealed_filters_infra_and_shares_the_reveal_cap(self) -> None:
        mem = _channel_memory(reveal_max_tools=3)
        mem.record("r1", "tool_used", {}, "ok")
        mem.record_revealed("r1", {"find_tools", "read_stored_result"})  # infra: ignored
        assert mem.tool_names("r1") == {"tool_used"}
        mem.record_revealed("r1", [f"tool_{i}" for i in range(3)])
        # Shared recency window: the reveal burst ages the older used tool out.
        assert mem.tool_names("r1") == {"tool_0", "tool_1", "tool_2"}

    def test_hydration_lifecycle(self) -> None:
        """A fresh room needs hydration; seeding fills the reveal set and is
        one-shot — even an EMPTY history marks the room hydrated so it is
        not re-queried every turn."""
        mem = _channel_memory()
        assert mem.needs_hydration("r1") is True
        mem.seed(
            "r1",
            [
                {"name": "square_get-menu", "arguments": {"unitToken": "L0N2"}, "result": "{...}"},
                {"name": "read_stored_result", "arguments": {}, "result": "infra: filtered"},
            ],
        )
        assert mem.needs_hydration("r1") is False
        assert mem.tool_names("r1") == {"square_get-menu"}

        mem.seed("r2", [])
        assert mem.needs_hydration("r2") is False  # empty history, still one-shot

    def test_live_entries_preempt_hydration(self) -> None:
        """A room already carrying live calls (or reveals) must not be seeded
        with stale history on top."""
        mem = ToolUsageMemory()
        mem.record("r1", "tool_a", {}, "ok")
        assert mem.needs_hydration("r1") is False
        mem2 = ToolUsageMemory()
        mem2.record_revealed("r2", {"tool_b"})
        assert mem2.needs_hydration("r2") is False


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


async def _first_round(ch: AIChannel) -> AIContext:
    """The context of the turn's first round: Tool Search's collapse of the
    toolset ``_build_context`` resolved, as both loops declare it (RMK-293)."""
    ctx = await ch._build_context(_event(), _binding(_CATALOGUE), _context())
    tools = ch._apply_tool_filters(ch._get_loop_ctx().all_context_tools)
    return ctx.model_copy(update={"tools": tools})


class TestToolUsageInContext:
    async def test_called_tool_is_revealed_under_tool_search(self) -> None:
        """With Tool Search ON, a tool the agent already used stays callable,
        while an unused catalogue tool stays hidden."""
        ch = _channel(tool_search=True)
        ch._tool_usage.record("r1", "SpotifyPlayback", {"action": "skip"}, "Skipped.")
        _current_loop_ctx.set(_ToolLoopContext(room_id="r1"))
        try:
            ctx = await _first_round(ch)
        finally:
            _current_loop_ctx.set(None)
        names = {t.name for t in ctx.tools}
        assert "SpotifyPlayback" in names  # re-revealed because it was used
        assert "unrelated_tool" not in names  # still hidden behind Tool Search

    async def test_found_tool_stays_callable_next_turn_without_being_called(self) -> None:
        """A find_tools reveal must survive into the NEXT turn even when the
        tool was never called — a tool found in turn N is often only called in
        turn N+1, after the user confirms (the create-cart regression)."""
        ch = _channel(tool_search=True)
        ch._tool_usage.record_revealed("r1", {"SpotifyPlayback"})
        _current_loop_ctx.set(_ToolLoopContext(room_id="r1"))
        try:
            ctx = await _first_round(ch)
        finally:
            _current_loop_ctx.set(None)
        names = {t.name for t in ctx.tools}
        assert "SpotifyPlayback" in names  # re-revealed although never called
        assert "unrelated_tool" not in names  # the rest of the catalogue stays hidden

    async def test_find_tools_records_matches_as_revealed(self) -> None:
        """The wiring: a real _handle_find_tools call lands its matches in
        ToolUsageMemory, so the reveal outlives the loop that searched."""
        ch = _channel(tool_search=True)
        loop_ctx = _ToolLoopContext(room_id="r1")
        loop_ctx.all_context_tools = [
            type("T", (), {"name": c["name"], "description": c["description"], "tags": []})()
            for c in _CATALOGUE
        ]
        _current_loop_ctx.set(loop_ctx)
        try:
            await ch._handle_find_tools({"query": "control playback"})
        finally:
            _current_loop_ctx.set(None)
        assert "SpotifyPlayback" in ch._tool_usage.tool_names("r1")

    async def test_build_context_hydrates_from_loader_once(self) -> None:
        """With a framework-injected loader, the first _build_context for a
        room seeds the memory from persisted history (the reveal set);
        subsequent builds must NOT re-query the store."""
        ch = _channel(tool_search=True)
        loader = AsyncMock(
            return_value=[
                {"name": "SpotifyPlayback", "arguments": {"action": "skip"}, "result": "ok"}
            ]
        )
        ch._tool_usage_loader = loader
        for _ in range(2):
            _current_loop_ctx.set(_ToolLoopContext(room_id="r1"))
            try:
                ctx = await _first_round(ch)
            finally:
                _current_loop_ctx.set(None)
        loader.assert_awaited_once_with("r1")
        assert "SpotifyPlayback" in {t.name for t in ctx.tools}  # re-revealed

    async def test_build_context_survives_loader_failure(self) -> None:
        """A broken store must not take the turn down — hydration degrades to
        the empty memory (and is not retried every turn)."""
        ch = _channel(tool_search=True)
        ch._tool_usage_loader = AsyncMock(side_effect=RuntimeError("store down"))
        _current_loop_ctx.set(_ToolLoopContext(room_id="r1"))
        try:
            ctx = await _first_round(ch)
        finally:
            _current_loop_ctx.set(None)
        assert ctx is not None
        assert ch._tool_usage.needs_hydration("r1") is False

    async def test_unused_tool_stays_hidden_without_prior_call(self) -> None:
        """Control: with Tool Search ON and nothing recorded, the catalogue is
        hidden — proves the reveal in the test above comes from usage memory."""
        ch = _channel(tool_search=True)
        _current_loop_ctx.set(_ToolLoopContext(room_id="r1"))
        try:
            ctx = await _first_round(ch)
        finally:
            _current_loop_ctx.set(None)
        names = {t.name for t in ctx.tools}
        assert "SpotifyPlayback" not in names
        assert "unrelated_tool" not in names

    async def test_framework_loader_reads_persisted_tool_calls(self) -> None:
        """End to end through the framework seam: register_channel injects the
        loader, and it rebuilds call dicts from persisted tool-call events —
        the piece that lets a conversation outlive the channel object."""
        from roomkit import RoomKit
        from roomkit.models.enums import EventType
        from roomkit.models.event import ToolCallContent

        kit = RoomKit()
        ch = _channel(tool_search=True)
        kit.register_channel(ch)
        assert ch._tool_usage_loader is not None

        await kit.create_room(room_id="r1")
        source = EventSource(channel_id="ai1", channel_type=ChannelType.SMS, provider="mock")
        # The start carries the model's request, the end what ran: here a
        # BEFORE_TOOL_USE hook put a real account back in place of a token.
        for event_type, arguments, status in (
            (EventType.TOOL_CALL_START, {"action": "skip", "account": "<ACCOUNT_1>"}, "pending"),
            (EventType.TOOL_CALL_END, {"action": "skip", "account": "alice"}, "completed"),
        ):
            await kit._store.add_event_auto_index(
                "r1",
                RoomEvent(
                    room_id="r1",
                    source=source,
                    type=event_type,
                    content=ToolCallContent(
                        tool_name="SpotifyPlayback",
                        tool_id="tc1",
                        arguments=arguments,
                        result="Skipped." if status == "completed" else None,
                        status=status,
                    ),
                ),
            )

        calls = await ch._tool_usage_loader("r1")
        # The model's own request, never the BEFORE_TOOL_USE rewrite.
        assert calls == [
            {
                "name": "SpotifyPlayback",
                "arguments": {"action": "skip", "account": "<ACCOUNT_1>"},
                "result": "Skipped.",
                "outcome": None,
            }
        ]
        # And the channel-side seam consumes it: the memory rebuilds.
        ch._tool_usage.seed("r1", calls)
        assert "SpotifyPlayback" in ch._tool_usage.tool_names("r1")

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
            assert parent.all_context_tools is not None
            kept = {t.name for t in ch._apply_tool_filters(parent.all_context_tools)}
        finally:
            _current_loop_ctx.set(None)
        assert "SpotifyPlayback" in kept  # callable under the child re-filter
        assert "unrelated_tool" not in kept
