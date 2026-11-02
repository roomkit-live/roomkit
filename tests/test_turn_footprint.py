"""An AI channel measures the turn's footprint before reading its memory (RFC §20).

What the history may take is what the window leaves once the rest of the turn
is in it. The channel measures that rest as round 0 sends it (the system prompt
with the agent's identity, the tools declared under Tool Search and the tool
policy, the channel's own notes) and the reply budget apart; a budget-aware
memory reserves the larger of its floor and that input, and the reply once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from roomkit.channels._ai_cuts import CUT_NOTE
from roomkit.channels._speaker import SPEAKER_ATTRIBUTION_NOTE
from roomkit.channels._turn_notes import turn_notes
from roomkit.channels.agent import Agent
from roomkit.core.framework import RoomKit
from roomkit.memory import (
    BudgetAwareMemory,
    MemoryProvider,
    MemoryResult,
    SlidingWindowMemory,
    TurnFootprint,
    current_turn_footprint,
)
from roomkit.memory.token_estimator import (
    estimate_message_tokens,
    estimate_tokens,
    estimate_tool_tokens,
)
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import InboundMessage
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import RoomEvent, TextContent
from roomkit.models.room import Room
from roomkit.orchestration.state import ConversationState, set_conversation_state
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills import SkillRegistry
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext
from roomkit.tools.policy import ToolPolicy
from tests.conftest import make_event
from tests.test_framework import SimpleChannel
from tests.test_skills import _make_skill_dir_full
from tests.test_skills_integration import MockScriptExecutor
from tests.tool_loop_modes import respond


class _Measured(MemoryProvider):
    """Records the footprint the channel measured when it reads the room."""

    def __init__(self) -> None:
        self.footprints: list[TurnFootprint | None] = []

    async def retrieve(
        self, room_id: str, current_event: RoomEvent, context: RoomContext, **kwargs: Any
    ) -> MemoryResult:
        self.footprints.append(current_turn_footprint())
        return MemoryResult()


def _binding(**metadata: Any) -> ChannelBinding:
    return ChannelBinding(
        channel_id="agent",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata=metadata,
    )


async def test_the_footprint_is_the_turn_as_round_0_sends_it(
    tmp_path: Path, streaming: bool
) -> None:
    _make_skill_dir_full(tmp_path, "reports", scripts=["build.py"])
    skills = SkillRegistry()
    skills.discover(tmp_path)
    provider = MockAIProvider(responses=["ok"], streaming=streaming)
    memory = _Measured()
    agent = Agent(
        "agent",
        provider=provider,
        role="Billing advisor",
        system_prompt="Answer billing questions.",
        max_tokens=400,
        tools=[
            AITool(name=f"tool_{i}", description="looks things up " * 30, parameters={})
            for i in range(40)
        ],
        tool_handler=AsyncMock(return_value="ok"),
        tool_search=True,
        tool_search_pinned=["tool_0"],
        skills=skills,
        script_executor=MockScriptExecutor(),
        tool_policy=ToolPolicy(deny=["run_skill_script"]),
        memory=memory,
    )

    await respond(
        agent,
        make_event(body="hello", channel_id="member", room_id="r1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )

    sent = provider.calls[0]
    declared = [tool.name for tool in sent.tools or []]
    assert "Agent Identity" in (sent.system_prompt or "")
    assert "run_skill_script" not in declared and "tool_1" not in declared
    assert memory.footprints == [
        TurnFootprint(input_tokens=_input_tokens(sent, notes=[]), reply_tokens=400)
    ]


def _input_tokens(sent: AIContext, *, notes: list[str]) -> int:
    """The input round 0 sent, with room for the speaker attribution and the
    note a cut answer brings."""
    return (
        estimate_tokens(sent.system_prompt or "")
        + sum(estimate_tool_tokens(tool) for tool in sent.tools or [])
        + estimate_tokens(turn_notes([SPEAKER_ATTRIBUTION_NOTE, CUT_NOTE, *notes]) or "")
    )


async def test_a_later_turn_is_measured_again_with_the_rounds_it_replays(
    streaming: bool,
) -> None:
    """The tools already used ride the next turn's history: the footprint
    counts the rounds it replays, and a tool handler, whose loop is the
    call's own, reads no footprint."""
    seen_in_handler: list[TurnFootprint | None] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        seen_in_handler.append(current_turn_footprint())
        return "result " * 1_000

    class _MeasuredWindow(SlidingWindowMemory):
        footprints: list[TurnFootprint | None] = []

        async def retrieve(self, *args: Any, **kwargs: Any) -> MemoryResult:
            self.footprints.append(current_turn_footprint())
            return await super().retrieve(*args, **kwargs)

    call = AIToolCall(id="c1", name="lookup", arguments={"q": "a"})
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", tool_calls=[call]),
            AIResponse(content="done"),
            AIResponse(content="again"),
        ],
        streaming=streaming,
    )
    memory = _MeasuredWindow()
    agent = Agent(
        "agent",
        provider=provider,
        tools=[AITool(name="lookup", description="Looks up", parameters={})],
        tool_handler=lookup,
        memory=memory,
    )
    kit = RoomKit()
    kit.register_channel(SimpleChannel("member"))
    kit.register_channel(agent)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "member")
    await kit.attach_channel("r1", "agent", category=ChannelCategory.INTELLIGENCE)
    for body in ("first", "second"):
        await kit.process_inbound(
            InboundMessage(channel_id="member", sender_id="u1", content=TextContent(body=body))
        )
    await kit.close()

    replayed = [m for m in provider.calls[-1].messages if isinstance(m.content, list)]
    assert [m.role for m in replayed] == ["assistant", "tool"]
    first, second = (f.input_tokens for f in (memory.footprints[0], memory.footprints[-1]))  # type: ignore[union-attr]
    # The rest of the turn is the same as the first's: the difference is the round.
    assert second - first == sum(estimate_message_tokens(m) for m in replayed)
    assert seen_in_handler == [None]


async def test_the_identity_is_measured_in_the_rooms_language(streaming: bool) -> None:
    """``handler.set_language()`` sets the room's language: the identity the
    prompt carries, and the channel measures, is in it."""
    provider = MockAIProvider(responses=["ok"], streaming=streaming)
    agent = Agent("agent", provider=provider, role="Advisor", language="English")
    room = set_conversation_state(Room(id="r1"), ConversationState(context={"language": "French"}))

    await respond(
        agent,
        make_event(body="bonjour", channel_id="member", room_id="r1"),
        _binding(),
        RoomContext(room=room),
    )

    assert "Always respond in French" in (provider.calls[0].system_prompt or "")


async def test_a_binding_s_max_tokens_is_measured_as_the_turn_reads_it() -> None:
    """A room's ``max_tokens: "100"`` is the turn's 100 (RFC §6.4, Appendix
    A.9): a budget-aware memory reserves 100 and the model is asked, as with
    any other memory (RMK-1255)."""
    provider = MockAIProvider(responses=["ok"])
    measured = _Measured()
    agent = Agent(
        "agent",
        provider=provider,
        memory=BudgetAwareMemory(measured, max_context_tokens=8_000),
    )

    await respond(
        agent,
        make_event(body="hello", channel_id="member", room_id="r1"),
        _binding(max_tokens="100"),
        RoomContext(room=Room(id="r1")),
    )

    assert [footprint.reply_tokens for footprint in measured.footprints if footprint] == [100]
    assert [call.max_tokens for call in provider.calls] == [100]


async def test_a_refused_max_tokens_fails_the_turn_as_with_any_memory() -> None:
    """A budget-aware memory reserves nothing for a ``max_tokens`` the turn
    refuses: the turn fails with pydantic's error naming every setting it
    refuses, as with any other memory, its model not asked."""
    provider = MockAIProvider(responses=["ok"])
    agent = Agent(
        "agent",
        provider=provider,
        memory=BudgetAwareMemory(SlidingWindowMemory(), max_context_tokens=8_000),
    )

    with pytest.raises(ValidationError, match="2 validation errors") as refused:
        await respond(
            agent,
            make_event(body="hello", channel_id="member", room_id="r1"),
            _binding(temperature="hot", max_tokens="many"),
            RoomContext(room=Room(id="r1")),
        )
    assert [error["loc"] for error in refused.value.errors()] == [
        ("temperature",),
        ("max_tokens",),
    ]
    assert provider.calls == []


async def test_outside_a_turn_there_is_no_footprint() -> None:
    assert current_turn_footprint() is None


async def test_a_budget_aware_memory_reserves_at_least_the_measured_footprint() -> None:
    history = [make_event(body="history " * 200, room_id="r1") for _ in range(30)]
    current = make_event(body="now", room_id="r1")
    context = RoomContext(room=Room(id="r1"), recent_events=[*history, current])

    async def kept(memory: BudgetAwareMemory, footprint: TurnFootprint | None) -> int:
        token = _current_loop_ctx.set(_ToolLoopContext(turn_footprint=footprint))
        try:
            return len((await memory.retrieve("r1", current, context)).events)
        finally:
            _current_loop_ctx.reset(token)

    def memory(reserved: int) -> BudgetAwareMemory:
        return BudgetAwareMemory(
            SlidingWindowMemory(max_events=100),
            max_context_tokens=20_000,
            reserved_tokens=reserved,
        )

    unmeasured = await kept(memory(0), None)
    measured = await kept(memory(0), TurnFootprint(input_tokens=12_000, reply_tokens=0))
    declared_larger = await kept(memory(12_000), TurnFootprint(input_tokens=2_000, reply_tokens=0))

    assert measured < unmeasured
    assert declared_larger == measured


async def test_the_reply_is_reserved_once_the_larger_of_the_margin_and_its_budget() -> None:
    """The 15 % margin is the reply's headroom: a reply budget under it changes
    nothing, one over it is reserved instead of it, never on top."""
    history = [make_event(body="history " * 200, room_id="r1") for _ in range(80)]
    current = make_event(body="now", room_id="r1")
    context = RoomContext(room=Room(id="r1"), recent_events=[*history, current])
    memory = BudgetAwareMemory(SlidingWindowMemory(max_events=100), max_context_tokens=20_000)

    async def kept(reply_tokens: int) -> int:
        footprint = TurnFootprint(input_tokens=1_000, reply_tokens=reply_tokens)
        token = _current_loop_ctx.set(_ToolLoopContext(turn_footprint=footprint))
        try:
            return len((await memory.retrieve("r1", current, context)).events)
        finally:
            _current_loop_ctx.reset(token)

    within_margin = await kept(2_000)  # the margin is 3,000 of 20,000
    over_margin = await kept(9_000)

    assert within_margin == await kept(0)
    assert over_margin < within_margin
