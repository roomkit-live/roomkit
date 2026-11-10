"""A room's declaration is kept from one turn to the next (RFC §6.4, RMK-345).

Where the provider holds a tool declared but unseen, a tool an earlier turn
opened (a reveal, an unlock) stays held rather than joining the tools shown,
so the next turn's declaration, the head of the cached prefix, is the same.
The turn reopens it with an exchange before its input: a call of the
discovery tool the turn declares, and a result that references it.
"""

from __future__ import annotations

import json
from pathlib import Path

from roomkit.channels import FallbackRouter
from roomkit.channels._tool_reopen import (
    REOPENING,
    insert_before,
    reopening_exchanges,
    room_declaration,
)
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding, RetryPolicy
from roomkit.models.context import RoomContext
from roomkit.models.delivery import STANDALONE
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.event import RoomEvent
from roomkit.models.room import Room
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AITool,
    AIToolCallPart,
    AIToolResultPart,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills import SkillRegistry
from tests.conftest import make_event
from tests.test_deferred_tools import (
    _CATALOGUE,
    _DONE,
    _FIXTURES,
    FailingHolder,
    HoldingProvider,
    _declaration,
    _round,
    _searching,
    _served,
    _tool,
    _turn,
)
from tests.tool_loop_modes import LoopRun, respond

_REVEAL = _round("c0", "find_tools", {"query": "track shipment", "max_results": 1})


def _reopening(context: AIContext) -> tuple[AIToolCallPart, AIToolResultPart]:
    """The exchange before the turn's input: its call and its result. (The
    mock keeps the round's context, whose messages the loop goes on to grow.)"""
    turn_input = next(i for i, m in enumerate(context.messages) if m.content == "go")
    call_message, result_message = context.messages[turn_input - 2 : turn_input]
    [call] = call_message.content
    [result] = result_message.content
    assert isinstance(call, AIToolCallPart) and isinstance(result, AIToolResultPart)
    return call, result


async def test_a_tool_revealed_in_one_turn_is_reopened_in_the_next(streaming: bool) -> None:
    provider = HoldingProvider(
        ai_responses=[_REVEAL, _DONE, _round("c1", "track_shipment"), _DONE],
        streaming=streaming,
    )
    ch = _searching(provider)

    await _turn(ch, _CATALOGUE)
    run = await _turn(ch, _CATALOGUE)

    first_turn, next_turn = provider.calls[0], provider.calls[2]
    assert _declaration(next_turn) == _declaration(first_turn)
    assert ("track_shipment", True) in _declaration(next_turn)
    call, result = _reopening(next_turn)
    assert (call.name, result.name, result.references) == (
        "find_tools",
        "find_tools",
        ["track_shipment"],
    )
    # Callable, and the exchange is context: it ran nothing and recorded nothing.
    assert [c.name for c in run.calls] == ["track_shipment"]
    assert not run.calls[0].failed


async def test_a_tool_a_skill_unlocked_is_reopened_with_the_skill(streaming: bool) -> None:
    registry = SkillRegistry()
    registry.discover(_FIXTURES)
    provider = HoldingProvider(
        ai_responses=[
            _round("c0", "activate_skill", {"name": "quote-policy"}),
            _DONE,
            _round("c1", "inventory"),
            _DONE,
        ],
        streaming=streaming,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=_served, skills=registry)
    tools = [_tool("lookup"), _tool("inventory")]

    await _turn(ch, tools)
    run = await _turn(ch, tools)

    assert _declaration(provider.calls[2]) == _declaration(provider.calls[0])
    call, result = _reopening(provider.calls[2])
    assert (call.name, call.arguments, result.references) == (
        "activate_skill",
        {"name": "quote-policy"},
        ["inventory"],
    )
    assert [c.name for c in run.calls] == ["inventory"]
    assert not run.calls[0].failed


async def test_a_fallback_that_cannot_hold_receives_the_reopened_tool(streaming: bool) -> None:
    fallback = MockAIProvider(ai_responses=[_DONE], streaming=streaming)
    ch = _searching(
        HoldingProvider(ai_responses=[_REVEAL, _DONE], streaming=streaming),
        fallback=fallback,
        retry_policy=RetryPolicy(max_retries=0),
    )
    await _turn(ch, _CATALOGUE)
    ch._router = FallbackRouter(FailingHolder(streaming=streaming), fallback)

    await _turn(ch, _CATALOGUE)

    declared = _declaration(fallback.calls[0])
    assert ("track_shipment", False) in declared
    assert all(not held for _, held in declared)


async def test_a_provider_that_cannot_hold_shows_the_tool_it_opened(streaming: bool) -> None:
    provider = MockAIProvider(ai_responses=[_REVEAL, _DONE, _DONE], streaming=streaming)
    ch = _searching(provider)

    await _turn(ch, _CATALOGUE)
    await _turn(ch, _CATALOGUE)

    next_turn = provider.calls[2]
    assert "track_shipment" in {t.name for t in next_turn.tools}
    assert not any(isinstance(p, AIToolCallPart) for m in next_turn.messages for p in m.content)


def test_the_room_declaration_keeps_what_its_first_turn_showed() -> None:
    kept = frozenset({"lookup", "find_tools"})
    shown = {"lookup", "find_tools", "track_shipment", "plan_tasks"}

    assert room_declaration(None, shown, set()) == frozenset(shown)
    # A tool the channel never holds joins; one opened since stays held.
    assert room_declaration(kept, shown, {"plan_tasks", "find_tools"}) == frozenset(
        {"lookup", "find_tools", "plan_tasks"}
    )
    # One the turn no longer shows (the policy denies it now) leaves.
    assert room_declaration(kept, {"find_tools"}, set()) == frozenset({"find_tools"})


async def test_a_pinned_tool_a_skill_gates_is_reopened_not_shown(streaming: bool) -> None:
    """A host's pinned tool is held while a skill gates it; once a turn opened
    it, the next reopens it like any other rather than showing it."""
    registry = SkillRegistry()
    registry.discover(_FIXTURES)
    provider = HoldingProvider(
        ai_responses=[
            _round("c0", "activate_skill", {"name": "quote-policy"}),
            _DONE,
            _round("c1", "inventory"),
            _DONE,
        ],
        streaming=streaming,
    )
    ch = AIChannel(
        "ai1",
        provider=provider,
        tool_handler=_served,
        tool_search=True,
        tool_search_pinned={"lookup", "inventory"},
        skills=registry,
    )
    tools = [*_CATALOGUE, _tool("inventory")]

    await _turn(ch, tools)
    run = await _turn(ch, tools)

    assert _declaration(provider.calls[2]) == _declaration(provider.calls[0])
    _call, result = _reopening(provider.calls[2])
    assert result.references == ["inventory"]
    assert [c.name for c in run.calls] == ["inventory"] and not run.calls[0].failed


def _history() -> list[RoomEvent]:
    return [
        make_event(room_id="r1", body="earlier question", channel_id="sms1"),
        make_event(
            room_id="r1", body="earlier answer", channel_id="ai1", channel_type=ChannelType.AI
        ),
    ]


async def _turn_after(ch: AIChannel, tools: list[AITool], event: RoomEvent) -> LoopRun:
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [tool.model_dump() for tool in tools]},
    )
    context = RoomContext(room=Room(id="r1"), bindings=[binding], recent_events=_history())
    return await respond(ch, event, binding, context)


def _reopenings(context: AIContext) -> list[AIMessage]:
    return [m for m in context.messages if m.metadata.get(REOPENING)]


async def test_the_exchange_sits_between_the_history_and_the_input(streaming: bool) -> None:
    provider = HoldingProvider(ai_responses=[_REVEAL, _DONE, _DONE], streaming=streaming)
    ch = _searching(provider)
    await _turn(ch, _CATALOGUE)

    await _turn_after(ch, _CATALOGUE, make_event(room_id="r1", body="go", channel_id="sms1"))

    messages = provider.calls[2].messages
    texts = [m.content if isinstance(m.content, str) else m.role for m in messages]
    turn_input = texts.index("go")
    # The history first, then the exchange, then the turn's input.
    assert texts[turn_input - 3 : turn_input + 1] == ["earlier answer", "assistant", "tool", "go"]


def _instruction() -> RoomEvent:
    return make_event(
        room_id="r1",
        body="Summarize.",
        channel_id="sms1",
        type=EventType.INSTRUCTION,
        metadata={STANDALONE: True},
    )


async def test_a_standalone_turn_names_no_skill_of_the_room(streaming: bool) -> None:
    """RFC §10.1.1: a standalone instruction reads none of the room's working
    state, its active skills included, so it reopens nothing."""
    registry = SkillRegistry()
    registry.discover(_FIXTURES)
    provider = HoldingProvider(
        ai_responses=[_round("c0", "activate_skill", {"name": "quote-policy"}), _DONE, _DONE],
        streaming=streaming,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=_served, skills=registry)
    tools = [_tool("lookup"), _tool("inventory")]
    await _turn(ch, tools)

    await _turn_after(ch, tools, _instruction())

    assert not _reopenings(provider.calls[2])


async def test_a_standalone_turn_leaves_the_room_declaration_as_it_was(streaming: bool) -> None:
    """A room's declaration that holds a tool the standalone turn does not
    show (a sticky one, which it does not read) is not shrunk by it."""
    provider = HoldingProvider(ai_responses=[_DONE], streaming=streaming)
    ch = _searching(provider)
    kept = frozenset({"lookup", "find_tools", "list_tools", "track_shipment"})
    ch._tool_usage.keep_declaration("r1", kept)

    await _turn_after(ch, _CATALOGUE, _instruction())

    assert ch._tool_usage.declaration("r1") == kept


async def test_a_fallback_that_cannot_hold_receives_no_exchange(streaming: bool) -> None:
    fallback = MockAIProvider(ai_responses=[_DONE], streaming=streaming)
    ch = _searching(
        HoldingProvider(ai_responses=[_REVEAL, _DONE], streaming=streaming),
        fallback=fallback,
        retry_policy=RetryPolicy(max_retries=0),
    )
    await _turn(ch, _CATALOGUE)
    ch._router = FallbackRouter(FailingHolder(streaming=streaming), fallback)

    await _turn(ch, _CATALOGUE)

    assert not _reopenings(fallback.calls[0])
    assert ("track_shipment", False) in _declaration(fallback.calls[0])


def _skill(root: Path, name: str, gated: str) -> None:
    folder = root / name
    folder.mkdir()
    front = f"---\nname: {name}\ndescription: The {name} skill.\nallowed_tools: {gated}\n---\n"
    (folder / "SKILL.md").write_text(f"{front}\nUse {gated}.\n")


async def test_each_active_skill_reopens_the_tools_it_gates(
    streaming: bool, tmp_path: Path
) -> None:
    _skill(tmp_path, "alpha", "a_tool")
    _skill(tmp_path, "zeta", "z_tool")
    registry = SkillRegistry()
    registry.discover(tmp_path)
    provider = HoldingProvider(
        ai_responses=[
            _round("c0", "activate_skill", {"name": "alpha"}),
            _round("c1", "activate_skill", {"name": "zeta"}),
            _DONE,
            _round("c2", "z_tool"),
            _DONE,
        ],
        streaming=streaming,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=_served, skills=registry)
    tools = [_tool("lookup"), _tool("a_tool"), _tool("z_tool")]

    await _turn(ch, tools)
    run = await _turn(ch, tools)

    exchange = _reopenings(provider.calls[3])
    calls = [p for m in exchange if m.role == "assistant" for p in m.content]
    results = [p for m in exchange if m.role == "tool" for p in m.content]
    assert [(c.name, c.arguments) for c in calls] == [
        ("activate_skill", {"name": "alpha"}),
        ("activate_skill", {"name": "zeta"}),
    ]
    assert [r.references for r in results] == [["a_tool"], ["z_tool"]]
    ack = json.loads(str(results[0].result))
    assert (ack["ok"], ack["name"], ack["already_active"]) == (True, "alpha", True)
    assert [c.name for c in run.calls] == ["z_tool"] and not run.calls[0].failed


def test_without_a_discovery_tool_or_an_input_nothing_is_reopened() -> None:
    opened = [_tool("track_shipment")]

    assert reopening_exchanges(opened, frozenset({"lookup"}), set(), None, "r") == ([], opened)
    messages = [AIMessage(role="user", content="go")]
    stray = AIMessage(role="user", content="not in the list")
    assert not insert_before(messages, stray, [AIMessage(role="assistant", content="x")])
    assert len(messages) == 1


async def test_a_standalone_turn_opens_no_gate_the_room_opened(streaming: bool) -> None:
    """RFC §10.1.1: a skill activated in the room is the room's working state,
    which a standalone instruction does not read: its gated tool stays shut."""
    registry = SkillRegistry()
    registry.discover(_FIXTURES)
    provider = MockAIProvider(
        ai_responses=[
            _round("c0", "activate_skill", {"name": "quote-policy"}),
            _DONE,
            _round("c1", "inventory"),
            _DONE,
        ],
        streaming=streaming,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=_served, skills=registry)
    tools = [_tool("lookup"), _tool("inventory")]
    await _turn(ch, tools)

    run = await _turn_after(ch, tools, _instruction())

    assert run.calls[0].failed and "gated" in str(run.calls[0].result)
