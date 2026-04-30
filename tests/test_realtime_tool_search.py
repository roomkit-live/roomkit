"""Tests for RealtimeVoiceChannel Tool Search support.

Covers the scorer + visibility logic in RealtimeToolSearchSupport in
isolation, plus end-to-end auto-activation, dispatch, reconfigure, and
session cleanup wiring through RealtimeVoiceChannel.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import RoomKit
from roomkit.channels._realtime_tool_search import (
    RealtimeToolSearchSupport,
    _score,
    _tokenize,
)
from roomkit.channels._tool_search_constants import (
    DEFAULT_TOOL_SEARCH_THRESHOLD,
    TOOL_FIND_TOOLS,
    TOOL_LIST_TOOLS,
    TOOL_SEARCH_PREAMBLE,
)
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> MockRealtimeProvider:
    return MockRealtimeProvider()


@pytest.fixture
def transport() -> MockRealtimeTransport:
    return MockRealtimeTransport()


def _tool(name: str, description: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
    }


def _make_catalogue(n: int) -> list[dict[str, Any]]:
    """Generate a catalogue large enough to trip auto-activation."""
    descriptions = {
        "lookup_contact_by_phone": "Look up a contact by phone number.",
        "send_sms": "Send an SMS message to a phone number.",
        "send_email": "Send an email message.",
        "create_invoice": "Create an invoice for a customer.",
        "fetch_weather": "Fetch the current weather for a location.",
        "schedule_meeting": "Schedule a meeting on the calendar.",
    }
    base = [_tool(n, d) for n, d in descriptions.items()]
    # Pad with filler tools so the catalogue exceeds n.
    filler = [_tool(f"filler_tool_{i}", f"Filler tool number {i}") for i in range(n)]
    return base + filler


# ---------------------------------------------------------------------------
# Pure-logic tests
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_splits_on_non_word(self) -> None:
        assert _tokenize("send_sms to_phone") == ["send", "sms", "to", "phone"]

    def test_lowercases(self) -> None:
        assert _tokenize("LookupContact") == ["lookupcontact"]

    def test_empty_string(self) -> None:
        assert _tokenize("") == []


class TestScore:
    def test_name_match_outweighs_description_match(self) -> None:
        # "phone" appears once in name (3pts) vs once in description (1pt).
        name_match = _score(["phone"], _tool("phone_lookup", "find a contact"))
        desc_match = _score(["phone"], _tool("contact_lookup", "find by phone"))
        assert name_match > desc_match
        assert name_match == 3
        assert desc_match == 1

    def test_zero_for_no_overlap(self) -> None:
        assert _score(["unrelated"], _tool("send_sms", "send a message")) == 0

    def test_empty_query(self) -> None:
        assert _score([], _tool("anything", "anything")) == 0


# ---------------------------------------------------------------------------
# RealtimeToolSearchSupport unit tests (no channel)
# ---------------------------------------------------------------------------


class TestVisibleTools:
    def test_search_tools_always_first(self) -> None:
        catalogue = [_tool("a"), _tool("b")]
        support = RealtimeToolSearchSupport(catalogue, pinned=["a"])
        support.init_session("s1")
        visible = support.visible_tools("s1", catalogue)
        names = [t["name"] for t in visible]
        # Search tools come first, pinned next, exposed empty.
        assert names[:2] == [TOOL_FIND_TOOLS, TOOL_LIST_TOOLS]
        assert "a" in names
        assert "b" not in names  # not pinned, not exposed

    def test_pinned_always_visible_unaffected_by_search(self) -> None:
        catalogue = [_tool("pinned_one"), _tool("hidden_one"), _tool("hidden_two")]
        support = RealtimeToolSearchSupport(catalogue, pinned=["pinned_one"])
        support.init_session("s1")
        # No find_tools yet — only pinned + search infra are visible.
        visible = support.visible_tools("s1", catalogue)
        names = [t["name"] for t in visible]
        assert "pinned_one" in names
        assert "hidden_one" not in names

    def test_session_isolation(self) -> None:
        catalogue = [_tool("a", "alpha"), _tool("b", "beta")]
        support = RealtimeToolSearchSupport(catalogue)
        support.init_session("s1")
        support.init_session("s2")
        # Expose 'a' on s1 only.
        support._exposed["s1"] = {"a"}
        v1 = {t["name"] for t in support.visible_tools("s1", catalogue)}
        v2 = {t["name"] for t in support.visible_tools("s2", catalogue)}
        assert "a" in v1
        assert "a" not in v2


class TestFindToolsHandler:
    async def test_find_tools_returns_matches(self) -> None:
        catalogue = [
            _tool("lookup_phone", "look up a contact by phone number"),
            _tool("send_sms", "send an SMS message"),
            _tool("send_email", "send an email message"),
        ]
        support = RealtimeToolSearchSupport(catalogue)
        support.init_session("s1")

        result_str, updated = await support.handle_tool_call(
            TOOL_FIND_TOOLS, {"query": "send a message"}, "s1"
        )
        result = json.loads(result_str)
        match_names = [m["name"] for m in result["matches"]]
        assert "send_sms" in match_names
        assert "send_email" in match_names
        # Caller MUST receive an updated tool list to push via reconfigure.
        assert updated is not None
        names_in_updated = [t["name"] for t in updated]
        assert "send_sms" in names_in_updated

    async def test_find_tools_swaps_exposure_window(self) -> None:
        """A second find_tools must replace, not extend, the exposure window."""
        catalogue = [
            _tool("send_sms", "send an SMS"),
            _tool("send_email", "send an email"),
            _tool("create_invoice", "create an invoice"),
        ]
        support = RealtimeToolSearchSupport(catalogue)
        support.init_session("s1")

        await support.handle_tool_call(TOOL_FIND_TOOLS, {"query": "send"}, "s1")
        first_exposed = set(support._exposed["s1"])
        assert "send_sms" in first_exposed
        assert "send_email" in first_exposed

        # Different query should drop send_* and reveal create_invoice.
        await support.handle_tool_call(TOOL_FIND_TOOLS, {"query": "invoice"}, "s1")
        second_exposed = set(support._exposed["s1"])
        assert "create_invoice" in second_exposed
        assert "send_sms" not in second_exposed

    async def test_find_tools_no_query_returns_error(self) -> None:
        support = RealtimeToolSearchSupport([_tool("foo")])
        support.init_session("s1")
        result_str, updated = await support.handle_tool_call(TOOL_FIND_TOOLS, {"query": ""}, "s1")
        result = json.loads(result_str)
        assert "error" in result
        assert updated is None

    async def test_find_tools_no_match_returns_empty_no_reconfigure(self) -> None:
        support = RealtimeToolSearchSupport([_tool("send_sms", "send an SMS")])
        support.init_session("s1")
        result_str, updated = await support.handle_tool_call(
            TOOL_FIND_TOOLS, {"query": "completely-unrelated-zzzz"}, "s1"
        )
        result = json.loads(result_str)
        assert result["matches"] == []
        # No matches → no reconfigure (None signals "do not push").
        assert updated is None

    async def test_find_tools_skips_infra_and_pinned(self) -> None:
        """Infra tools and pinned tools never appear in find_tools matches."""
        catalogue = [
            _tool("pinned_send_sms", "send an SMS message"),
            _tool("send_email", "send an email message"),
        ]
        support = RealtimeToolSearchSupport(catalogue, pinned=["pinned_send_sms"])
        support.init_session("s1")
        result_str, _ = await support.handle_tool_call(TOOL_FIND_TOOLS, {"query": "send"}, "s1")
        match_names = [m["name"] for m in json.loads(result_str)["matches"]]
        assert "pinned_send_sms" not in match_names
        assert "send_email" in match_names
        # Search infra should also never appear.
        assert TOOL_FIND_TOOLS not in match_names


class TestListToolsHandler:
    async def test_list_tools_lists_all(self) -> None:
        catalogue = [_tool("a"), _tool("b"), _tool("c")]
        support = RealtimeToolSearchSupport(catalogue)
        support.init_session("s1")
        result_str, updated = await support.handle_tool_call(TOOL_LIST_TOOLS, {}, "s1")
        result = json.loads(result_str)
        assert {t["name"] for t in result["tools"]} == {"a", "b", "c"}
        assert result["count"] == 3
        # list_tools is informational only — never reconfigures.
        assert updated is None

    async def test_list_tools_filters_by_category(self) -> None:
        catalogue = [
            _tool("admin_list_users"),
            _tool("admin_delete_user"),
            _tool("public_search"),
        ]
        support = RealtimeToolSearchSupport(catalogue)
        support.init_session("s1")
        result_str, _ = await support.handle_tool_call(
            TOOL_LIST_TOOLS, {"category": "admin_"}, "s1"
        )
        names = {t["name"] for t in json.loads(result_str)["tools"]}
        assert names == {"admin_list_users", "admin_delete_user"}

    async def test_list_tools_truncates_at_60(self) -> None:
        catalogue = [_tool(f"t{i}", "x") for i in range(80)]
        support = RealtimeToolSearchSupport(catalogue)
        support.init_session("s1")
        result_str, _ = await support.handle_tool_call(TOOL_LIST_TOOLS, {}, "s1")
        result = json.loads(result_str)
        assert len(result["tools"]) == 60
        assert result["truncated"] is True


# ---------------------------------------------------------------------------
# Channel integration tests
# ---------------------------------------------------------------------------


class TestAutoActivation:
    async def test_disabled_when_catalogue_below_threshold(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """Catalogue below threshold → no Tool Search support is constructed."""
        few_tools = [_tool(f"t{i}") for i in range(5)]
        channel = RealtimeVoiceChannel(
            "rt-small",
            provider=provider,
            transport=transport,
            tools=few_tools,
            tool_search_threshold=20,  # default — auto-detect
        )
        assert channel._tool_search_support is None

    async def test_auto_enables_when_catalogue_exceeds_threshold(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        big_catalogue = _make_catalogue(40)  # comfortably above default 20
        channel = RealtimeVoiceChannel(
            "rt-big",
            provider=provider,
            transport=transport,
            tools=big_catalogue,
        )
        assert channel._tool_search_support is not None

    async def test_force_enable_below_threshold(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        channel = RealtimeVoiceChannel(
            "rt-forced",
            provider=provider,
            transport=transport,
            tools=[_tool("a"), _tool("b")],
            tool_search=True,  # force on
        )
        assert channel._tool_search_support is not None

    async def test_force_disable_above_threshold(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        channel = RealtimeVoiceChannel(
            "rt-disabled",
            provider=provider,
            transport=transport,
            tools=_make_catalogue(40),
            tool_search=False,  # force off
        )
        assert channel._tool_search_support is None

    async def test_default_threshold_constant(self) -> None:
        """Threshold default matches Google's published guidance."""
        assert DEFAULT_TOOL_SEARCH_THRESHOLD == 20


class TestSessionStartWiring:
    async def test_session_start_sends_only_search_and_pinned_tools(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """Connect-time tool list = search infra + pinned, not the full catalogue."""
        catalogue = _make_catalogue(40)
        pinned = ["lookup_contact_by_phone"]

        original_connect = provider.connect
        connect_calls: list[dict[str, Any]] = []

        async def spy_connect(session: Any, **kwargs: Any) -> None:
            connect_calls.append(kwargs)
            await original_connect(session, **kwargs)

        provider.connect = spy_connect  # type: ignore[method-assign]

        channel = RealtimeVoiceChannel(
            "rt-search",
            provider=provider,
            transport=transport,
            tools=catalogue,
            tool_search_pinned=pinned,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-search")

        await channel.start_session(room.id, "user-1", "fake-ws")

        sent_tools = connect_calls[0]["tools"]
        names = [t["name"] for t in sent_tools]
        assert TOOL_FIND_TOOLS in names
        assert TOOL_LIST_TOOLS in names
        assert "lookup_contact_by_phone" in names
        # Hidden tools must not be in the initial connect.
        assert "filler_tool_0" not in names
        # Sanity: a small list, not the full 46.
        assert len(sent_tools) < 10

    async def test_session_start_appends_search_preamble(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        catalogue = _make_catalogue(40)
        original_connect = provider.connect
        connect_calls: list[dict[str, Any]] = []

        async def spy_connect(session: Any, **kwargs: Any) -> None:
            connect_calls.append(kwargs)
            await original_connect(session, **kwargs)

        provider.connect = spy_connect  # type: ignore[method-assign]

        channel = RealtimeVoiceChannel(
            "rt-search",
            provider=provider,
            transport=transport,
            tools=catalogue,
            system_prompt="Base prompt.",
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-search")

        await channel.start_session(room.id, "user-1", "fake-ws")

        sent_prompt = connect_calls[0]["system_prompt"]
        assert "Base prompt." in sent_prompt
        # Spot-check a stable phrase from TOOL_SEARCH_PREAMBLE.
        assert "find_tools" in sent_prompt
        assert TOOL_SEARCH_PREAMBLE.split("\n")[0][:20] in sent_prompt


class TestDispatch:
    async def test_find_tools_dispatch_pushes_reconfigure(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        catalogue = _make_catalogue(40)
        channel = RealtimeVoiceChannel(
            "rt-search",
            provider=provider,
            transport=transport,
            tools=catalogue,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-search")

        session = await channel.start_session(room.id, "user-1", "fake-ws")

        provider.reconfigure = AsyncMock()  # type: ignore[method-assign]

        await provider.simulate_tool_call(
            session, "call-1", TOOL_FIND_TOOLS, {"query": "send a message"}
        )
        await asyncio.sleep(0.05)

        # Must have pushed a reconfigure with the new visible tool list.
        provider.reconfigure.assert_called_once()
        sent_tools = provider.reconfigure.call_args.kwargs.get("tools")
        assert sent_tools is not None
        sent_names = [t["name"] for t in sent_tools]
        # Search tools always retained; matched tools now visible.
        assert TOOL_FIND_TOOLS in sent_names
        assert "send_sms" in sent_names

        # Tool result was also submitted.
        assert len(provider.tool_results) == 1
        result = json.loads(provider.tool_results[0][2])
        assert "matches" in result

    async def test_list_tools_does_not_reconfigure(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """list_tools is informational — must not reconfigure."""
        catalogue = _make_catalogue(40)
        channel = RealtimeVoiceChannel(
            "rt-search",
            provider=provider,
            transport=transport,
            tools=catalogue,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-search")

        session = await channel.start_session(room.id, "user-1", "fake-ws")

        provider.reconfigure = AsyncMock()  # type: ignore[method-assign]

        await provider.simulate_tool_call(session, "call-1", TOOL_LIST_TOOLS, {})
        await asyncio.sleep(0.05)

        provider.reconfigure.assert_not_called()
        # But a result was still submitted.
        assert len(provider.tool_results) == 1

    async def test_find_tools_no_match_does_not_reconfigure(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        """When find_tools matches nothing, the visible surface stays as-is."""
        catalogue = _make_catalogue(40)
        channel = RealtimeVoiceChannel(
            "rt-search",
            provider=provider,
            transport=transport,
            tools=catalogue,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-search")

        session = await channel.start_session(room.id, "user-1", "fake-ws")

        provider.reconfigure = AsyncMock()  # type: ignore[method-assign]

        await provider.simulate_tool_call(
            session,
            "call-1",
            TOOL_FIND_TOOLS,
            {"query": "zzzzqqqq"},
        )
        await asyncio.sleep(0.05)

        provider.reconfigure.assert_not_called()


class TestCleanup:
    async def test_end_session_removes_exposure_state(
        self,
        provider: MockRealtimeProvider,
        transport: MockRealtimeTransport,
    ) -> None:
        catalogue = _make_catalogue(40)
        channel = RealtimeVoiceChannel(
            "rt-search",
            provider=provider,
            transport=transport,
            tools=catalogue,
        )
        kit = RoomKit()
        kit.register_channel(channel)
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rt-search")

        session = await channel.start_session(room.id, "user-1", "fake-ws")
        sid = session.id
        assert channel._tool_search_support is not None
        assert sid in channel._tool_search_support._exposed

        await channel.end_session(session)

        assert sid not in channel._tool_search_support._exposed
