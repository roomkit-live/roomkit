"""What a find_tools query must find first, whatever else the catalogue holds.

Properties of :func:`search_catalogue` on catalogues where a tool is easily
buried: a source served as one tool per action, every tool tagged with the
source's words, beside a tool of another source, untagged, whose name is what
the model asked for. Then the exact-name rule through both channels that run
``find_tools``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from roomkit.channels._realtime_tool_search import RealtimeToolSearchSupport
from roomkit.channels._tool_search import search_catalogue
from roomkit.channels._tool_search_constants import TOOL_FIND_TOOLS
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall, AIToolResultPart
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.tool_loop_modes import respond

_NONE: frozenset[str] = frozenset()


def _tool(name: str, description: str = "", tags: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "description": description, "tags": tags or []}


def _source(prefix: str, actions: list[str], tags: list[str]) -> list[dict[str, Any]]:
    """A source served as one tool per action, each tagged with the source's words."""
    return [
        _tool(f"{prefix}_{action}", f"{action.replace('_', ' ').capitalize()}.", [*tags, action])
        for action in actions
    ]


_MAIL_ACTIONS = [
    "search_messages",
    "send_message",
    "read_message",
    "list_messages",
    "move_message",
]
_MAIL_TAGS = ["mail", "message", "search", "send", "read", "inbox"]


def _mail_catalogue(*local: dict[str, Any]) -> list[dict[str, Any]]:
    """Two tagged mail sources, a few unrelated ones, and the given local tools."""
    return [
        *_source("acmemail", _MAIL_ACTIONS, _MAIL_TAGS),
        *_source("globexmail", _MAIL_ACTIONS, _MAIL_TAGS),
        *_source("calendar", ["list_events", "create_event"], ["calendar", "event"]),
        *_source("boards", ["create_card", "move_card"], ["board", "card", "kanban"]),
        _tool("weather", "Get the forecast."),
        *local,
    ]


def _names(ranked: list[dict[str, Any]]) -> list[str]:
    return [tool["name"] for tool in ranked]


class TestExactName:
    """A query that is a tool's name finds that tool first."""

    def test_the_named_tool_comes_first_ahead_of_richer_matches(self) -> None:
        # The tagged sources carry both words in name, tags and description;
        # the local tool only in its name. Scored alone, it comes after them.
        cat = _mail_catalogue(_tool("search_messages", "Look through the local archive."))
        ranked = search_catalogue(cat, "search_messages", 5, exclude_names=_NONE)
        assert _names(ranked)[0] == "search_messages"

    def test_the_name_is_compared_without_case_or_surrounding_space(self) -> None:
        cat = _mail_catalogue(_tool("search_messages", "Look through the local archive."))
        ranked = search_catalogue(cat, "  Search_Messages ", 5, exclude_names=_NONE)
        assert _names(ranked)[0] == "search_messages"

    def test_it_holds_the_single_place_a_one_result_search_has(self) -> None:
        cat = _mail_catalogue(_tool("search_messages", "Look through the local archive."))
        ranked = search_catalogue(cat, "search_messages", 1, exclude_names=_NONE)
        assert _names(ranked) == ["search_messages"]

    def test_an_excluded_tool_is_never_named_even_by_its_name(self) -> None:
        cat = _mail_catalogue(_tool("search_messages", "Look through the local archive."))
        ranked = search_catalogue(
            cat, "search_messages", 5, exclude_names=frozenset({"search_messages"})
        )
        assert "search_messages" not in _names(ranked)


class TestAWordCountsOnce:
    """A word counts once per tool, from the strongest field that carries it."""

    @pytest.mark.parametrize("tagged_first", [True, False])
    def test_tags_repeating_the_name_add_nothing(self, tagged_first: bool) -> None:
        # Same name words, same description; one carries them again as tags.
        # Equal scores keep the catalogue's order, so each twin leads when
        # listed first. Counted twice, the tagged one would lead in both orders.
        tagged = _tool("alpha_lookup_order", "Look up an order.", ["lookup", "order"])
        untagged = _tool("beta_lookup_order", "Look up an order.")
        twins = [tagged, untagged] if tagged_first else [untagged, tagged]
        cat = [*twins, *_source("calendar", ["list_events"], ["calendar"]), _tool("weather")]
        ranked = search_catalogue(cat, "lookup order", 5, exclude_names=_NONE)
        assert _names(ranked)[:2] == _names(twins)

    def test_a_word_in_the_name_outweighs_it_as_a_source_wide_tag(self) -> None:
        # Every tool of the source is tagged "search"; one is named for it. The
        # descriptions carry no query word, so the name against the tag
        # decides, and the named tool comes last so a tie would lose it.
        actions = ["send_message", "read_message", "list_messages", "search_messages"]
        cat = [
            *[_tool(f"acmemail_{a}", "An Acme mail action.", [*_MAIL_TAGS, a]) for a in actions],
            *_source("calendar", ["list_events"], ["calendar"]),
            _tool("weather", "Get the forecast."),
        ]
        ranked = search_catalogue(cat, "search messages", 5, exclude_names=_NONE)
        assert _names(ranked)[0] == "acmemail_search_messages"


class TestACatalogueAsItComes:
    """Names follow no convention the scorer can rely on, and a host's dicts may lack one."""

    def test_a_verb_shared_by_many_names_stays_a_common_word(self) -> None:
        # Named verb first, as many MCP servers name their tools. A rarity read
        # off the name's prefix would make each verb a source of its own, rare,
        # and the get_* tools would answer a calendar question.
        cat = [
            _tool("get_weather", "Get the current weather for a city."),
            _tool("get_forecast", "Get the weather forecast for the next days."),
            _tool("get_time", "Get the current time in a timezone."),
            _tool("get_stock_price", "Get the latest price of a stock."),
            _tool("get_news", "Get the latest news headlines."),
            _tool("get_contact", "Get a contact's phone number and email."),
            _tool("send_sms", "Send an SMS to a phone number."),
            _tool("send_email", "Send an email to an address."),
            _tool("create_event", "Create a calendar event."),
            _tool("list_events", "List the calendar events of a day."),
            _tool("delete_event", "Delete a calendar event."),
            _tool("search_web", "Search the web for a query."),
            _tool("search_files", "Search the files of the workspace."),
            _tool("list_files", "List the files in a directory."),
            _tool("read_file", "Read the content of a file."),
        ]
        calendar = search_catalogue(cat, "get the calendar events", 5, exclude_names=_NONE)
        assert _names(calendar)[0] == "list_events"
        files = search_catalogue(cat, "get my files", 5, exclude_names=_NONE)
        assert _names(files)[:3] == ["search_files", "list_files", "read_file"]

    def test_a_tool_without_a_name_is_scored_not_fatal(self) -> None:
        cat = [{"name": None, "description": "Relay a note."}, _tool("send_sms", "Send an SMS.")]
        ranked = search_catalogue(cat, "send sms", 5, exclude_names=_NONE)
        assert _names(ranked)[0] == "send_sms"


# ---------------------------------------------------------------------------
# Through the channels that run find_tools
# ---------------------------------------------------------------------------

_LOCAL = _tool("search_messages", "Look through the local archive.")


def _ai_tools() -> list[AITool]:
    return [
        AITool(name=t["name"], description=t["description"], parameters={}, tags=t["tags"])
        for t in _mail_catalogue(_LOCAL)
    ]


async def _served(name: str, arguments: dict[str, Any]) -> str:
    return f'{{"{name}": "ok"}}'


def _found(provider: MockAIProvider) -> list[str]:
    [result] = [
        part
        for message in provider.calls[-1].messages
        if message.role == "tool" and isinstance(message.content, list)
        for part in message.content
        if isinstance(part, AIToolResultPart) and part.name == TOOL_FIND_TOOLS
    ]
    return [match["name"] for match in json.loads(str(result.result))["matches"]]


async def test_the_text_channel_finds_a_named_tool_first(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    AIToolCall(
                        id="c0",
                        name=TOOL_FIND_TOOLS,
                        arguments={"query": "search_messages", "max_results": 3},
                    )
                ],
            ),
            AIResponse(content="done", finish_reason="stop"),
        ],
        streaming=streaming,
    )
    channel = AIChannel("ai1", provider=provider, tool_handler=_served, tool_search=True)
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [tool.model_dump() for tool in _ai_tools()]},
    )
    await respond(
        channel,
        make_event(room_id="r1", body="go", channel_id="sms1"),
        binding,
        RoomContext(room=Room(id="r1")),
    )
    assert _found(provider)[0] == "search_messages"


async def test_the_realtime_channel_finds_a_named_tool_first() -> None:
    support = RealtimeToolSearchSupport(_mail_catalogue(_LOCAL))
    support.init_session("s1")
    result, names = await support.handle_tool_call(
        TOOL_FIND_TOOLS, {"query": "search_messages", "max_results": 3}, "s1"
    )
    assert [match["name"] for match in json.loads(result)["matches"]][0] == "search_messages"
    assert names[0] == "search_messages"
