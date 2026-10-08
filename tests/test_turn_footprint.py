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

from roomkit import split_turn_notes
from roomkit.channels._speaker import SPEAKER_ATTRIBUTION_NOTE
from roomkit.channels._turn_notes import turn_notes
from roomkit.channels.agent import Agent
from roomkit.memory import (
    BudgetAwareMemory,
    MemoryProvider,
    MemoryResult,
    SlidingWindowMemory,
    TurnFootprint,
    current_turn_footprint,
)
from roomkit.memory.token_estimator import estimate_tokens, estimate_tool_tokens
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.event import RoomEvent
from roomkit.models.room import Room
from roomkit.orchestration.state import ConversationState, set_conversation_state
from roomkit.providers.ai.base import AIContext, AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills import SkillRegistry
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext
from roomkit.tools.policy import ToolPolicy
from tests.conftest import make_event
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
    """The input round 0 sent, with room for the speaker attribution."""
    return (
        estimate_tokens(sent.system_prompt or "")
        + sum(estimate_tool_tokens(tool) for tool in sent.tools or [])
        + estimate_tokens(turn_notes([SPEAKER_ATTRIBUTION_NOTE, *notes]) or "")
    )


async def test_a_later_turn_is_measured_again_with_the_digest_of_the_tools_used(
    streaming: bool,
) -> None:
    """The digest of the tools already used rides the next turn's input: the
    footprint counts it, and a tool handler, whose loop is the call's own,
    reads no footprint."""
    seen_in_handler: list[TurnFootprint | None] = []

    async def lookup(name: str, arguments: dict[str, Any]) -> str:
        seen_in_handler.append(current_turn_footprint())
        return "result " * 1_000

    call = AIToolCall(id="c1", name="lookup", arguments={"q": "a"})
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(content="", tool_calls=[call]),
            AIResponse(content="done"),
            AIResponse(content="again"),
        ],
        streaming=streaming,
    )
    memory = _Measured()
    agent = Agent(
        "agent",
        provider=provider,
        tools=[AITool(name="lookup", description="Looks up", parameters={})],
        tool_handler=lookup,
        memory=memory,
    )
    for body in ("first", "second"):
        await respond(
            agent,
            make_event(body=body, channel_id="member", room_id="r1"),
            _binding(),
            RoomContext(room=Room(id="r1")),
        )

    second = provider.calls[-1]
    _input, sent_notes = split_turn_notes(str(second.messages[-1].content))
    digest = sent_notes.split("\n\n", 1)[1]
    assert "lookup" in digest
    assert memory.footprints[-1] == TurnFootprint(
        input_tokens=_input_tokens(second, notes=[digest]), reply_tokens=0
    )
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
