"""A hidden tool called by its exact name is revealed only once the tool
answered (RMK-461, RFC §6.4).

Tool Search recovers an exact-name call to a catalogue tool it hides, and
reveals the tool as the room's tool memory keeps any tool used: a call the
tool answered (served, failed, withheld by an ON_TOOL_CALL hook) reveals it
for the turn's next rounds and later turns; a call refused before it ran
(BEFORE_TOOL_USE, its handler's refusal) or that nothing served reveals
nothing. Invalid model arguments retain the schema for this turn alone,
without recording an executed call in the room's memory.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from roomkit import HookExecution, HookResult, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.core.exceptions import ToolFailedError, ToolRefusedError, UnservedToolCallError
from roomkit.models.channel import ChannelBinding
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.providers.ai.base import AIResponse, AIToolCall, AIToolResultPart
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.test_deferred_tools import HoldingProvider
from tests.tool_loop_modes import respond

_SMS_TOOL = {
    "name": "send_sms",
    "description": "Send an SMS text message to a phone number.",
    "parameters": {
        "type": "object",
        "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
        "required": ["to", "body"],
    },
}
_LOOKUP_TOOL = {"name": "lookup", "description": "Look a record up.", "parameters": {}}
_CATALOGUE = [
    *({"name": f"widget_{i}", "description": f"Operate widget number {i}."} for i in range(5)),
    _SMS_TOOL,
]
_SMS = {"to": "+15551234567", "body": "hi"}
_DONE = AIResponse(content="done", finish_reason="stop")


def _round(*calls: AIToolCall) -> AIResponse:
    return AIResponse(content="", finish_reason="tool_calls", tool_calls=list(calls))


def _sms(call_id: str = "t1", arguments: dict[str, Any] | None = None) -> AIToolCall:
    return AIToolCall(
        id=call_id, name="send_sms", arguments=_SMS if arguments is None else arguments
    )


def _provider(streaming: bool, *first_round: AIToolCall, holding: bool = False) -> MockAIProvider:
    """One round of *first_round* (a direct send_sms by default), its answer,
    then a second turn that only answers."""
    kind = HoldingProvider if holding else MockAIProvider
    return kind(
        ai_responses=[_round(*(first_round or (_sms(),))), _DONE, _DONE], streaming=streaming
    )


async def _two_turns(
    provider: MockAIProvider, handler: Any, hooks: Any = None, **channel: Any
) -> AIChannel:
    ai = AIChannel("ai1", provider=provider, tool_search=True, tool_handler=handler, **channel)
    kit = RoomKit()
    kit.register_channel(ai)
    if hooks is not None:
        hooks(kit)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "ai1")
    binding = ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata={
            "tools": [*_CATALOGUE, _LOOKUP_TOOL] if "tool_search_pinned" in channel else _CATALOGUE
        },
    )
    for body in ("go", "again"):
        event = make_event(room_id="r1", body=body, channel_id="sms1")
        await respond(ai, event, binding, await kit._build_context("r1"))
    await kit.close()
    return ai


def _revealed(provider: MockAIProvider) -> tuple[bool, bool]:
    """Whether send_sms is declared on the turn's next round, and on the next
    turn's first round."""
    next_round = {tool.name for tool in provider.calls[1].tools}
    next_turn = {tool.name for tool in provider.calls[2].tools}
    return "send_sms" in next_round, "send_sms" in next_turn


async def _served(name: str, arguments: dict[str, Any]) -> str:
    return '{"sent": true}'


async def _refusing(name: str, arguments: dict[str, Any]) -> str:
    raise ToolRefusedError("Not to that number.")


async def _failing(name: str, arguments: dict[str, Any]) -> str:
    raise ToolFailedError("The SMS gateway is down.")


async def _unserved(name: str, arguments: dict[str, Any]) -> str:
    raise UnservedToolCallError(name)


def _deny_before_use(kit: RoomKit) -> None:
    @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="deny")
    async def _deny(event: Any, ctx: Any) -> HookResult:
        return HookResult.block("not now")


def _withhold_result(kit: RoomKit) -> None:
    @kit.hook(HookTrigger.ON_TOOL_CALL, execution=HookExecution.SYNC, name="withhold")
    async def _withhold(event: Any, ctx: Any) -> HookResult:
        return HookResult.block("withheld")


@pytest.mark.parametrize(
    ("handler", "hooks", "arguments"),
    [
        (_refusing, None, None),
        (_served, _deny_before_use, None),
        (_unserved, None, None),
    ],
    ids=["handler-refusal", "before-tool-use-block", "nothing-served"],
)
async def test_a_call_refused_before_it_ran_reveals_nothing(
    streaming: bool, handler: Any, hooks: Any, arguments: dict[str, Any] | None
) -> None:
    provider = _provider(streaming, _sms(arguments=arguments))
    await _two_turns(provider, handler, hooks)

    assert _revealed(provider) == (False, False)


async def test_invalid_arguments_retain_the_schema_but_not_an_executed_call(
    streaming: bool,
) -> None:
    provider = _provider(streaming, _sms(arguments={"to": "+15551234567"}))
    await _two_turns(provider, _served)

    assert _revealed(provider) == (True, False)


@pytest.mark.parametrize(
    ("handler", "hooks"),
    [(_served, None), (_failing, None), (_served, _withhold_result)],
    ids=["served", "failed", "on-tool-call-block"],
)
async def test_a_call_the_tool_answered_keeps_it_revealed(
    streaming: bool, handler: Any, hooks: Any
) -> None:
    provider = _provider(streaming)
    await _two_turns(provider, handler, hooks)

    assert _revealed(provider) == (True, True)


async def test_a_refused_recovery_leaves_a_find_tools_reveal_of_the_same_round(
    streaming: bool,
) -> None:
    search = AIToolCall(
        id="s1", name="find_tools", arguments={"query": "send an sms text message"}
    )
    provider = _provider(streaming, search, _sms())
    await _two_turns(provider, _refusing)

    assert _revealed(provider)[0] is True


async def test_a_refused_recovery_leaves_a_served_one_of_the_same_round(streaming: bool) -> None:
    calls = 0

    async def first_served(name: str, arguments: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ToolRefusedError("Only one message.")
        return '{"sent": true}'

    provider = _provider(streaming, _sms("t1"), _sms("t2"))
    await _two_turns(provider, first_served)

    assert _revealed(provider) == (True, True)


async def test_a_sibling_answered_while_a_recovery_waits_does_not_reference_it(
    streaming: bool,
) -> None:
    """On a provider that holds tools unseen, a pinned sibling that answers
    while the recovered call waits on its gate references nothing of it."""

    async def slow_lookup(name: str, arguments: dict[str, Any]) -> str:
        await asyncio.sleep(0.01)  # answers while the recovery waits on its gate
        return '{"found": true}'

    def deny_after_a_wait(kit: RoomKit) -> None:
        @kit.hook(HookTrigger.BEFORE_TOOL_USE, execution=HookExecution.SYNC, name="approve")
        async def _approve(event: Any, ctx: Any) -> HookResult:
            if event.name == "send_sms":
                await asyncio.sleep(0.05)
                return HookResult.block("denied by the approver")
            return HookResult.allow()

    lookup = AIToolCall(id="c1", name="lookup", arguments={})
    provider = _provider(streaming, lookup, _sms(), holding=True)
    await _two_turns(provider, slow_lookup, deny_after_a_wait, tool_search_pinned={"lookup"})

    parts = [
        part
        for message in provider.calls[1].messages
        if message.role == "tool" and isinstance(message.content, list)
        for part in message.content
        if isinstance(part, AIToolResultPart)
    ]
    assert [part.name for part in parts] == ["lookup", "send_sms"]
    assert all("send_sms" not in (part.references or []) for part in parts)


@pytest.mark.parametrize("search_first", [True, False], ids=["search-first", "recovery-first"])
async def test_a_served_recovery_survives_a_find_tools_of_its_round(
    streaming: bool, search_first: bool
) -> None:
    """A find_tools of the same round swaps the reveal window; the tool a
    served recovery used stays shown, whichever call settles first."""
    search = AIToolCall(id="s1", name="find_tools", arguments={"query": "operate widget number"})
    calls = (search, _sms()) if search_first else (_sms(), search)
    provider = _provider(streaming, *calls)
    await _two_turns(provider, _served)

    assert _revealed(provider) == (True, True)
