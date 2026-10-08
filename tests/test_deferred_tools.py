"""A tool that has to appear mid-turn is held unseen, not declared (RFC §6.4, RMK-330).

Where the provider can hold a tool declared but unseen (Anthropic's deferred
loading), what Tool Search hides and what a skill's gating keeps closed are
declared that way from the first round, and the result that opens a tool
references it: the declaration does not change within the turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding, RetryPolicy
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.tool_call import AIResponseEvent, ToolCallVerdict
from roomkit.providers.ai.base import (
    AIContext,
    AIMessage,
    AIResponse,
    AITextPart,
    AITool,
    AIToolCall,
    AIToolResultPart,
    ProviderError,
)
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.providers.anthropic import AnthropicAIProvider, AnthropicConfig
from roomkit.providers.anthropic.request import build_kwargs
from roomkit.skills import SkillRegistry
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext
from roomkit.tools.policy import ToolPolicy
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond

_FIXTURES = Path(__file__).parents[1] / "benchmarks" / "chat" / "fixtures"


class HoldingProvider(MockAIProvider):
    """A scripted model whose provider holds a tool declared but unseen."""

    @property
    def supports_deferred_tools(self) -> bool:
        return True


def _tool(name: str) -> AITool:
    return AITool(name=name, description=f"The {name} tool", parameters={})


def _round(call_id: str, name: str, arguments: dict[str, Any] | None = None) -> AIResponse:
    return AIResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[AIToolCall(id=call_id, name=name, arguments=arguments or {})],
    )


_DONE = AIResponse(content="done", finish_reason="stop")
_CATALOGUE = [_tool("lookup"), _tool("track_shipment"), _tool("refund"), _tool("export")]


async def _served(name: str, arguments: dict[str, Any]) -> str:
    return f'{{"{name}": "ok"}}'


async def _turn(ch: AIChannel, tools: list[AITool]) -> LoopRun:
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={"tools": [tool.model_dump() for tool in tools]},
    )
    return await respond(
        ch,
        make_event(room_id="r1", body="go", channel_id="sms1"),
        binding,
        RoomContext(room=Room(id="r1")),
    )


def _declaration(context: AIContext) -> list[tuple[str, bool]]:
    return [(t.name, t.defer_loading) for t in context.tools]


def _results(context: AIContext) -> list[AIToolResultPart]:
    return [
        part
        for message in context.messages
        if message.role == "tool" and isinstance(message.content, list)
        for part in message.content
        if isinstance(part, AIToolResultPart)
    ]


def _searching(provider: MockAIProvider, **kwargs: Any) -> AIChannel:
    return AIChannel(
        "ai1",
        provider=provider,
        tool_handler=_served,
        tool_search=True,
        tool_search_pinned={"lookup"},
        **kwargs,
    )


async def test_a_revealed_tool_is_referenced_not_declared(streaming: bool) -> None:
    provider = HoldingProvider(
        ai_responses=[
            _round("c0", "find_tools", {"query": "track shipment", "max_results": 1}),
            _round("c1", "track_shipment"),
            _DONE,
        ],
        streaming=streaming,
    )
    ch = _searching(provider)

    run = await _turn(ch, _CATALOGUE)

    first = _declaration(provider.calls[0])
    assert ("lookup", False) in first and ("find_tools", False) in first
    assert ("track_shipment", True) in first and ("refund", True) in first
    assert [_declaration(call) for call in provider.calls] == [first] * 3
    [found] = [part for part in _results(provider.calls[-1]) if part.name == "find_tools"]
    assert found.references == ["track_shipment"]
    assert [c.name for c in run.calls] == ["find_tools", "track_shipment"]
    assert not run.calls[-1].failed


async def test_a_skill_references_the_tools_it_opens(streaming: bool) -> None:
    registry = SkillRegistry()
    registry.discover(_FIXTURES)
    provider = HoldingProvider(
        ai_responses=[
            _round("c0", "activate_skill", {"name": "quote-policy"}),
            _round("c1", "inventory"),
            _DONE,
        ],
        streaming=streaming,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=_served, skills=registry)

    run = await _turn(ch, [_tool("lookup"), _tool("inventory")])

    first = _declaration(provider.calls[0])
    assert ("inventory", True) in first
    assert [_declaration(call) for call in provider.calls] == [first] * 3
    [activation] = [p for p in _results(provider.calls[-1]) if p.name == "activate_skill"]
    assert activation.references == ["inventory"]
    assert not run.calls[-1].failed


async def test_a_gated_tool_held_unseen_is_still_refused(streaming: bool) -> None:
    registry = SkillRegistry()
    registry.discover(_FIXTURES)
    provider = HoldingProvider(
        ai_responses=[_round("c0", "inventory"), _DONE], streaming=streaming
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=_served, skills=registry)

    run = await _turn(ch, [_tool("lookup"), _tool("inventory")])

    assert run.calls[0].failed and "gated" in str(run.calls[0].result)


async def test_a_tool_the_policy_denies_is_not_even_held(streaming: bool) -> None:
    provider = HoldingProvider(ai_responses=[_DONE], streaming=streaming)
    ch = _searching(provider, tool_policy=ToolPolicy(deny=["refund"]))

    await _turn(ch, _CATALOGUE)

    assert "refund" not in {name for name, _ in _declaration(provider.calls[0])}


async def test_a_held_tool_is_reported_once_referenced(streaming: bool) -> None:
    provider = HoldingProvider(
        ai_responses=[
            _round("c0", "find_tools", {"query": "track shipment", "max_results": 1}),
            _DONE,
        ],
        streaming=streaming,
    )
    ch = _searching(provider)
    seen: list[AIResponseEvent] = []

    async def observe(event: AIResponseEvent) -> None:
        seen.append(event)

    ch._after_response_hook = observe

    await _turn(ch, _CATALOGUE)

    declared = {tool.name: tool.origin for tool in seen[0].declared_tools}
    assert declared["track_shipment"] == "revealed"
    assert "refund" not in declared and "export" not in declared  # held, never referenced


async def test_a_provider_that_cannot_hold_declares_what_it_shows(streaming: bool) -> None:
    provider = MockAIProvider(
        ai_responses=[
            _round("c0", "find_tools", {"query": "track shipment", "max_results": 1}),
            _DONE,
        ],
        streaming=streaming,
    )
    ch = _searching(provider)

    await _turn(ch, _CATALOGUE)

    assert not any(t.defer_loading for call in provider.calls for t in call.tools)
    assert "track_shipment" not in {t.name for t in provider.calls[0].tools}
    assert "track_shipment" in {t.name for t in provider.calls[1].tools}
    assert not any(part.references for part in _results(provider.calls[1]))


def _anthropic_kwargs(tools: list[AITool], messages: list[AIMessage]) -> dict[str, Any]:
    config = AnthropicConfig(api_key="k", model="claude-sonnet-5")
    return build_kwargs(config, AIContext(messages=messages, tools=tools))


def test_anthropic_holds_a_deferred_tool_and_marks_the_last_one_it_shows() -> None:
    tools = [
        _tool("find_tools"),
        _tool("read_stored_result"),
        _tool("track_shipment").model_copy(update={"defer_loading": True}),
    ]

    rendered = _anthropic_kwargs(tools, [AIMessage(role="user", content="hi")])["tools"]

    assert [t.get("defer_loading", False) for t in rendered] == [False, False, True]
    assert [("cache_control" in t) for t in rendered] == [False, True, False]


def test_anthropic_never_defers_every_tool() -> None:
    tools = [_tool("a").model_copy(update={"defer_loading": True})]

    rendered = _anthropic_kwargs(tools, [AIMessage(role="user", content="hi")])["tools"]

    assert "defer_loading" not in rendered[0]


def test_anthropic_puts_references_alone_and_the_result_after_the_tool_results() -> None:
    """The API refuses a ``tool_reference`` mixed with other content in a
    ``tool_result``: the references go alone, the result follows the
    message's tool results, named after its call."""
    found = AIToolResultPart(
        tool_call_id="c0", name="find_tools", result='{"found": 1}', references=["track"]
    )
    other = AIToolResultPart(tool_call_id="c1", name="lookup", result="A-1 shipped")
    messages = [
        AIMessage(role="user", content="hi"),
        AIMessage(role="tool", content=[found, other]),
    ]

    blocks = _anthropic_kwargs([_tool("find_tools")], messages)["messages"][1]["content"]

    assert blocks[0]["content"] == [{"type": "tool_reference", "tool_name": "track"}]
    assert blocks[1] == {"type": "tool_result", "tool_use_id": "c1", "content": "A-1 shipped"}
    assert blocks[2]["type"] == "text"
    assert (
        blocks[2]["text"].startswith("[Result of find_tools]")
        and '"found": 1' in blocks[2]["text"]
    )


def test_the_anthropic_catalogue_says_which_models_hold_tools() -> None:
    def holds(model: str) -> bool:
        return AnthropicAIProvider(
            AnthropicConfig(api_key="k", model=model)
        ).supports_deferred_tools

    assert holds("claude-sonnet-5-5") and holds("claude-sonnet-5") and holds("claude-haiku-4-5")
    assert not holds("claude-opus-4-1") and not holds("claude-unknown-model")


async def test_a_held_tool_called_without_a_search_references_itself(streaming: bool) -> None:
    """A call to a held tool nothing referenced goes through recovery: it runs,
    its result references it, and the turn reports it as revealed."""
    provider = HoldingProvider(ai_responses=[_round("c0", "refund"), _DONE], streaming=streaming)
    ch = _searching(provider)
    seen: list[AIResponseEvent] = []

    async def observe(event: AIResponseEvent) -> None:
        seen.append(event)

    ch._after_response_hook = observe

    run = await _turn(ch, _CATALOGUE)

    assert not run.calls[0].failed
    [refund] = [p for p in _results(provider.calls[-1]) if p.name == "refund"]
    assert refund.references == ["refund"]
    assert {t.name: t.origin for t in seen[0].declared_tools}["refund"] == "revealed"
    assert [_declaration(call) for call in provider.calls] == [_declaration(provider.calls[0])] * 2


async def test_a_confused_skill_name_references_the_tools_it_reveals(streaming: bool) -> None:
    registry = SkillRegistry()
    registry.discover(_FIXTURES)
    provider = HoldingProvider(
        ai_responses=[_round("c0", "activate_skill", {"name": "track"}), _DONE],
        streaming=streaming,
    )
    ch = _searching(provider, skills=registry)

    await _turn(ch, _CATALOGUE)

    [hint] = [p for p in _results(provider.calls[-1]) if p.name == "activate_skill"]
    assert hint.references == ["track_shipment"]


async def test_an_activation_references_only_what_it_would_have_shown(streaming: bool) -> None:
    """Under Tool Search an activated skill's tool stays behind find_tools, as
    it does on a provider that cannot hold tools: nothing is referenced."""
    registry = SkillRegistry()
    registry.discover(_FIXTURES)
    provider = HoldingProvider(
        ai_responses=[_round("c0", "activate_skill", {"name": "quote-policy"}), _DONE],
        streaming=streaming,
    )
    ch = _searching(provider, skills=registry)

    await _turn(ch, [*_CATALOGUE, _tool("inventory")])

    [activation] = [p for p in _results(provider.calls[-1]) if p.name == "activate_skill"]
    assert activation.references == []


async def test_a_blocked_activation_references_nothing(streaming: bool) -> None:
    registry = SkillRegistry()
    registry.discover(_FIXTURES)
    provider = HoldingProvider(
        ai_responses=[_round("c0", "activate_skill", {"name": "quote-policy"}), _DONE],
        streaming=streaming,
    )
    ch = AIChannel("ai1", provider=provider, tool_handler=_served, skills=registry)

    async def block(event: Any, **_: Any) -> ToolCallVerdict:
        return ToolCallVerdict(result='{"error": "no"}', blocked=True)

    ch._tool_call_hook = block

    await _turn(ch, [_tool("lookup"), _tool("inventory")])

    [activation] = [p for p in _results(provider.calls[-1]) if p.name == "activate_skill"]
    assert activation.references == []


class FailingHolder(HoldingProvider):
    async def generate(self, context: AIContext) -> AIResponse:
        raise ProviderError("down", provider="holding", retryable=True, status_code=503)


async def test_a_fallback_that_cannot_hold_receives_what_the_turn_shows(streaming: bool) -> None:
    fallback = MockAIProvider(ai_responses=[_DONE], streaming=streaming)
    ch = _searching(
        FailingHolder(streaming=streaming),
        fallback_provider=fallback,
        retry_policy=RetryPolicy(max_retries=0),
    )

    await _turn(ch, _CATALOGUE)

    declared = _declaration(fallback.calls[0])
    assert all(not held for _, held in declared)
    assert "refund" not in {name for name, _ in declared}  # held, never referenced


def test_a_compaction_shows_the_tools_whose_references_it_summarized() -> None:
    ch = _searching(HoldingProvider())
    loop_ctx = _ToolLoopContext(room_id="r1")
    loop_ctx.first_shown = frozenset({"lookup", "find_tools"})
    found = AIToolResultPart(
        tool_call_id="c0", name="find_tools", result="{}", references=["track_shipment"]
    )
    token = _current_loop_ctx.set(loop_ctx)
    try:
        ch._show_summarized_references([AIMessage(role="tool", content=[found])])
    finally:
        _current_loop_ctx.reset(token)

    assert "track_shipment" in loop_ctx.first_shown


def test_anthropic_behind_a_base_url_holds_no_tool() -> None:
    config = AnthropicConfig(api_key="k", model="claude-sonnet-5", base_url="https://gw.example")
    assert not AnthropicAIProvider(config).supports_deferred_tools


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ("", []),
        ([], []),
        (
            [AITextPart(text="body")],
            [
                {"type": "text", "text": "[Result of activate_skill]"},
                {"type": "text", "text": "<tool_result>\nbody\n</tool_result>"},
            ],
        ),
    ],
    ids=["empty-text", "empty-parts", "parts"],
)
def test_anthropic_renders_a_referencing_result_of_any_shape(
    result: Any, expected: list[dict[str, Any]]
) -> None:
    part = AIToolResultPart(
        tool_call_id="c0", name="activate_skill", result=result, references=["inventory"]
    )
    messages = [AIMessage(role="user", content="hi"), AIMessage(role="tool", content=[part])]

    blocks = _anthropic_kwargs([_tool("x")], messages)["messages"][1]["content"]

    assert blocks[0]["content"] == [{"type": "tool_reference", "tool_name": "inventory"}]
    unmarked = [{k: v for k, v in b.items() if k != "cache_control"} for b in blocks[1:]]
    assert unmarked == expected
