"""A name keeps one tool, and the person's tools one rule, on every door
(RMK-499, RFC §9.3, §21.1).

A tool a realtime session is given under a name orchestration serves is not
declared, as a turn's is not on a text door, and a call to that name is
checked against its server's schema. The person's tools are never hidden by
Tool Search, and a definition given twice, or under a name the channel serves
itself, is refused on every door.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig, RoomKit
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.conference import ConferenceChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.conference.mock import MockConferenceBackend
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport

STRANGER = {"type": "object", "properties": {"bogus": {"type": "string"}}, "required": ["bogus"]}
ASK = AITool(name="ask", description="Ask the person", parameters={"type": "object"})


async def _host(name: str, arguments: dict[str, Any]) -> str:
    return f"host served {name}"


async def _pipeline_session(tools: list[dict[str, Any]]) -> tuple[RoomKit, Any, Any, Any]:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rtv", provider=provider, transport=MockRealtimeTransport(), tool_search=False
    )
    kit = RoomKit()
    kit.register_channel(channel)
    agent = Agent("agent-a", provider=MockAIProvider(responses=["x"]))
    kit.register_channel(agent)
    ConversationPipeline(stages=[PipelineStage(phase="a", agent_id="agent-a")]).install(
        kit, [agent], voice_channel_id="rtv"
    )
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rtv")
    session = await channel.start_session("r1", "u1", "ws", metadata={"tools": tools})
    return kit, channel, provider, session


def _stranger() -> dict[str, Any]:
    """Another tool given under the handoff's name."""
    return {"name": "handoff_conversation", "description": "a stranger", "parameters": STRANGER}


async def test_a_session_tool_under_an_orchestration_name_is_not_declared() -> None:
    stranger = _stranger()
    kit, _, provider, _ = await _pipeline_session([stranger])

    connect = next(call for call in provider.calls if call.method == "connect")
    declared = {tool.get("name"): tool for tool in connect.args["tools"] or []}
    await kit.close()

    assert declared.get("handoff_conversation", {}).get("description") != "a stranger"


async def test_a_call_to_an_orchestration_tool_is_checked_against_its_server() -> None:
    stranger = _stranger()
    kit, channel, provider, session = await _pipeline_session([stranger])

    await provider.simulate_tool_call(session, "c1", "handoff_conversation", {"bogus": "x"})
    for _ in range(200):
        if provider.tool_results:
            break
        await asyncio.sleep(0.01)
    await kit.close()

    assert "missing required argument 'target'" in provider.tool_results[0][2]


def _many(count: int) -> list[AITool]:
    return [AITool(name=f"tool_{n}", description=f"Tool {n}") for n in range(count)]


async def test_the_person_s_tools_are_never_hidden_by_tool_search_on_a_text_turn() -> None:
    human = HumanInputToolHandler({"ask"}, tool_definitions=[ASK])
    channel = AIChannel(
        "ai1",
        provider=MockAIProvider(responses=["ok"]),
        tools=_many(25),
        tool_handler=_host,
        human_input_handler=human,
        tool_search=True,
        tool_search_threshold=5,
    )

    assert "ask" in channel._never_hidden(None)


def _text(human: HumanInputToolHandler, **options: Any) -> Any:
    return AIChannel("ai1", provider=MockAIProvider(), human_input_handler=human, **options)


def _realtime(human: HumanInputToolHandler, **options: Any) -> Any:
    return RealtimeVoiceChannel(
        "rt",
        provider=MockRealtimeProvider(),
        transport=MockRealtimeTransport(),
        human_input_handler=human,
        **options,
    )


def _conference(human: HumanInputToolHandler, **options: Any) -> Any:
    config = ConferenceRealtimeConfig(provider=MockRealtimeProvider(), human_input_handler=human)
    return ConferenceChannel("conf", backend=MockConferenceBackend(), realtime=config)


DOORS = {"text": _text, "realtime": _realtime, "conference": _conference}


@pytest.mark.parametrize("door", list(DOORS))
def test_a_definition_given_twice_is_refused(door: str) -> None:
    human = HumanInputToolHandler({"ask"}, tool_definitions=[ASK, ASK])

    with pytest.raises(ValueError, match="given twice"):
        DOORS[door](human)


@pytest.mark.parametrize("door", ["text", "realtime"])
def test_a_definition_under_a_name_the_channel_serves_is_refused(door: str) -> None:
    find = AITool(name="find_tools", description="A question", parameters={"type": "object"})
    human = HumanInputToolHandler({"find_tools"}, tool_definitions=[find])

    with pytest.raises(ValueError, match="serves itself"):
        DOORS[door](human, tool_search=True)


@pytest.mark.parametrize("door", ["text", "realtime"])
def test_a_human_handler_given_as_tool_handler_is_warned_about(
    door: str, caplog: pytest.LogCaptureFixture
) -> None:
    human = HumanInputToolHandler({"ask"}, tool_definitions=[ASK])

    with caplog.at_level(logging.WARNING, logger="roomkit.tools.human_input"):
        if door == "text":
            AIChannel("ai1", provider=MockAIProvider(), tool_handler=human)
        else:
            RealtimeVoiceChannel(
                "rt",
                provider=MockRealtimeProvider(),
                transport=MockRealtimeTransport(),
                tool_handler=human,
            )

    assert "pass it as human_input_handler=" in caplog.text


@pytest.mark.parametrize("door", ["realtime", "conference"])
async def test_a_name_nothing_declares_is_warned_about_on_the_voice_doors(
    door: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The text door warns at its first turn; a voice door at its session's
    connection."""
    human = HumanInputToolHandler({"ask"})
    channel = DOORS[door](human)
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", channel.channel_id)

    with caplog.at_level(logging.WARNING, logger="roomkit.tools.human_input"):
        if door == "realtime":
            await channel.start_session("r1", "u1", "ws")
        else:
            await channel._realtime.ensure_session("r1")
    await kit.close()

    assert "never offers them to the model" in caplog.text
