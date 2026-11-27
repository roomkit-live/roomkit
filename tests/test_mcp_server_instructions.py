"""What an MCP server publishes for the model reaches its system prompt (RFC §21.2).

The provider keeps ``InitializeResult.instructions``; a channel the host hands
them to adds them to every system prompt it composes, set apart under the
host's label for the server. Every channel test runs on both paths that
compose a prompt: the AI channel's turn and the realtime session.
"""

from __future__ import annotations

import logging
import sys
import textwrap
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from roomkit import AIChannelTurnConfig, RoomKit
from roomkit.channels._server_instructions import MAX_SERVER_INSTRUCTIONS_CHARS
from roomkit.channels._tool_search_constants import TOOL_SEARCH_PREAMBLE
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.delivery import STANDALONE
from roomkit.models.enums import ChannelCategory, ChannelType, EventType
from roomkit.models.room import Room
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.mcp import MCPToolProvider
from roomkit.voice.base import VoiceSession, VoiceSessionState
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from roomkit.voice.realtime.reasoning import (
    AgentReasoningBackend,
    ReasoningRequest,
    TranscriptLine,
)
from tests.channel_tasks import settled
from tests.conftest import make_event
from tests.tool_loop_modes import respond

pytest.importorskip("mcp.server.mcpserver")

GM = "Resolve every roll yourself: never ask the players for a total."
NOTES = "Dates are ISO 8601."
HOST = "You are the game master."

_SERVER = textwrap.dedent(
    """\
    import sys
    from mcp.server import MCPServer

    instructions = sys.argv[1] if len(sys.argv) > 1 else None
    server = MCPServer("gamemaster", instructions=instructions, log_level="WARNING")

    @server.tool()
    def roll(dice: str) -> int:
        \"\"\"Roll dice.\"\"\"
        return 4

    server.run()
    """
)


@pytest.fixture
def server_script(tmp_path: Path) -> str:
    path = tmp_path / "server.py"
    path.write_text(_SERVER)
    return str(path)


async def _tool(name: str, arguments: dict[str, Any]) -> str:
    return "ok"


# -- The provider --------------------------------------------------------------


async def test_the_provider_keeps_what_the_server_published(server_script: str) -> None:
    async with MCPToolProvider.from_command(sys.executable, [server_script, GM]) as mcp:
        assert mcp.instructions == GM
        assert mcp.server_info["name"] == "gamemaster"


async def test_a_server_that_published_nothing_reads_none(server_script: str) -> None:
    async with MCPToolProvider.from_command(sys.executable, [server_script]) as mcp:
        assert mcp.instructions is None


def test_reading_them_before_connecting_is_refused() -> None:
    provider = MCPToolProvider.from_command(sys.executable, ["server.py"])

    with pytest.raises(RuntimeError, match="not connected"):
        _ = provider.instructions
    with pytest.raises(RuntimeError, match="not connected"):
        _ = provider.server_info


async def test_a_stdio_servers_instructions_reach_the_model(server_script: str) -> None:
    async with MCPToolProvider.from_command(sys.executable, [server_script, GM]) as mcp:
        provider = MockAIProvider(responses=["ok"])
        channel = AIChannel(
            "ai1",
            provider=provider,
            system_prompt=HOST,
            tools=mcp.get_tools(),
            tool_handler=mcp.as_tool_handler(),
            server_instructions={"gm": mcp.instructions},
        )
        await respond(channel, make_event(body="I attack the goblin"), _binding(), _context())

    prompt = provider.calls[0].system_prompt or ""
    assert prompt.count(GM) == 1
    assert "## MCP server: gm\n<server_instructions>\n" in prompt


# -- Both paths ----------------------------------------------------------------


def _context() -> RoomContext:
    return RoomContext(room=Room(id="r1"))


def _binding(metadata: dict[str, Any] | None = None) -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
        metadata=metadata or {},
    )


async def _ai_prompt(room_prompt: str | None = None, **channel: Any) -> str:
    """The system prompt of an AI channel's turn."""
    provider = MockAIProvider(responses=["ok"])
    ai = AIChannel("ai1", provider=provider, system_prompt=HOST, tool_handler=_tool, **channel)
    metadata = {"system_prompt": room_prompt} if room_prompt else {}
    await respond(ai, make_event(body="go"), _binding(metadata), _context())
    return provider.calls[0].system_prompt or ""


@asynccontextmanager
async def _in_session(
    rt: RealtimeVoiceChannel, metadata: dict[str, Any] | None = None
) -> AsyncIterator[VoiceSession]:
    """A session of *rt* in a room of its own; the kit closes after."""
    kit = RoomKit()
    kit.register_channel(rt)
    room = await kit.create_room()
    await kit.attach_channel(room.id, rt.channel_id)
    try:
        yield await rt.start_session(room.id, "user-1", "ws", metadata=metadata)
    finally:
        await kit.close()


def _connected_with(provider: MockRealtimeProvider) -> str:
    """The system prompt the session connected with."""
    connect = next(call for call in provider.calls if call.method == "connect")
    return connect.args["system_prompt"] or ""


async def _realtime_prompt(room_prompt: str | None = None, **channel: Any) -> str:
    """The system prompt a realtime session connects with."""
    provider = MockRealtimeProvider()
    rt = RealtimeVoiceChannel(
        "rt1",
        provider=provider,
        transport=MockRealtimeTransport(),
        system_prompt=HOST,
        tool_handler=_tool,
        **channel,
    )
    async with _in_session(rt, {"system_prompt": room_prompt} if room_prompt else None):
        return _connected_with(provider)


PromptOf = Callable[..., Awaitable[str]]
PATHS: dict[str, PromptOf] = {"ai channel": _ai_prompt, "realtime session": _realtime_prompt}
CHANNELS: dict[str, Callable[..., object]] = {
    "ai channel": lambda **kw: AIChannel("ai1", provider=MockAIProvider(), **kw),
    "realtime session": lambda **kw: RealtimeVoiceChannel(
        "rt1", provider=MockRealtimeProvider(), transport=MockRealtimeTransport(), **kw
    ),
}
_TOOLS = [{"name": "roll", "description": "Roll dice."}, {"name": "hp", "description": "HP."}]
_DECLARED: dict[PromptOf, list[Any]] = {
    _ai_prompt: [AITool(**tool) for tool in _TOOLS],
    _realtime_prompt: _TOOLS,
}


def _skills(tmp_path: Path) -> SkillRegistry:
    skill = tmp_path / "combat"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: combat\ndescription: Fights\n---\nFight well.")
    registry = SkillRegistry()
    registry.discover(tmp_path)
    return registry


@pytest.mark.parametrize("prompt_of", PATHS.values(), ids=PATHS.keys())
async def test_each_servers_instructions_follow_the_hosts_prompt_once(
    prompt_of: PromptOf,
) -> None:
    prompt = await prompt_of(server_instructions={"gm": GM, "notes": NOTES})

    assert prompt.count(GM) == 1 and prompt.count(NOTES) == 1
    assert prompt.count("<server_instructions>") == 2
    assert prompt.index(HOST) < prompt.index("## MCP server: gm") < prompt.index(GM)
    assert prompt.index(GM) < prompt.index("## MCP server: notes") < prompt.index(NOTES)


@pytest.mark.parametrize("prompt_of", PATHS.values(), ids=PATHS.keys())
async def test_they_come_before_the_skills(prompt_of: PromptOf, tmp_path: Path) -> None:
    prompt = await prompt_of(server_instructions={"gm": GM}, skills=_skills(tmp_path))

    assert prompt.index(GM) < prompt.index("<available_skills>")


@pytest.mark.parametrize("prompt_of", PATHS.values(), ids=PATHS.keys())
async def test_tool_search_leaves_them_in_the_prompt(prompt_of: PromptOf) -> None:
    prompt = await prompt_of(
        server_instructions={"gm": GM}, tool_search=True, tools=_DECLARED[prompt_of]
    )

    assert TOOL_SEARCH_PREAMBLE in prompt
    assert prompt.count(GM) == 1


@pytest.mark.parametrize("prompt_of", PATHS.values(), ids=PATHS.keys())
async def test_a_prompt_set_for_the_room_keeps_them(prompt_of: PromptOf) -> None:
    prompt = await prompt_of(room_prompt="Room prompt.", server_instructions={"gm": GM})

    assert HOST not in prompt
    assert prompt.index("Room prompt.") < prompt.index(GM)
    assert prompt.count(GM) == 1


@pytest.mark.parametrize("prompt_of", PATHS.values(), ids=PATHS.keys())
async def test_a_server_that_published_nothing_adds_nothing(prompt_of: PromptOf) -> None:
    prompt = await prompt_of(server_instructions={"gm": None, "notes": "  \n"})

    assert prompt == HOST


@pytest.mark.parametrize("prompt_of", PATHS.values(), ids=PATHS.keys())
async def test_instructions_past_the_bound_are_cut_with_a_warning(
    prompt_of: PromptOf, caplog: pytest.LogCaptureFixture
) -> None:
    long = "word " * MAX_SERVER_INSTRUCTIONS_CHARS + "TAIL"

    with caplog.at_level(logging.WARNING, logger="roomkit.channels.server_instructions"):
        prompt = await prompt_of(server_instructions={"gm": long})

    body = prompt.split("<server_instructions>\n", 1)[1].rsplit("\n</server_instructions>", 1)[0]
    assert len(body) <= MAX_SERVER_INSTRUCTIONS_CHARS
    assert body.endswith("…") and "TAIL" not in prompt
    assert any("gm" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("make", CHANNELS.values(), ids=CHANNELS.keys())
@pytest.mark.parametrize("label", ["", "game master", "gm\n", "gm:1", "x" * 65, "<gm>"])
def test_a_label_that_is_not_an_identifier_is_refused(
    make: Callable[..., object], label: str
) -> None:
    with pytest.raises(ValueError, match="not an identifier"):
        make(server_instructions={label: None})


@pytest.mark.parametrize("make", CHANNELS.values(), ids=CHANNELS.keys())
def test_a_server_mapped_to_something_other_than_text_is_refused(
    make: Callable[..., object],
) -> None:
    provider = MCPToolProvider.from_command(sys.executable, ["server.py"])

    with pytest.raises(TypeError, match="pass the provider's .instructions"):
        make(server_instructions={"gm": provider})


@pytest.mark.parametrize("make", CHANNELS.values(), ids=CHANNELS.keys())
@pytest.mark.parametrize("given", ["text", "blank text", "provider"])
def test_instructions_not_mapped_to_a_label_are_refused(
    make: Callable[..., object], given: str
) -> None:
    # The mapping forgotten: refused whatever the server published, blank
    # text included, rather than failing on the day it publishes some.
    value = {
        "text": "Roll dice yourself.",
        "blank text": "",
        "provider": MCPToolProvider.from_command(sys.executable, ["server.py"]),
    }[given]

    with pytest.raises(TypeError, match="maps a label to each server's instructions"):
        make(server_instructions=value)


# -- One path's own -------------------------------------------------------------


async def test_every_turn_carries_them_once_a_standalone_one_too() -> None:
    provider = MockAIProvider(responses=["ok", "ok", "ok"])
    ai = AIChannel("ai1", provider=provider, system_prompt=HOST, server_instructions={"gm": GM})
    instruction = make_event(
        body="Greet the table.", type=EventType.INSTRUCTION, metadata={STANDALONE: True}
    )
    for event in (make_event(body="one"), make_event(body="two"), instruction):
        await respond(ai, event, _binding(), _context())

    assert [(call.system_prompt or "").count(GM) for call in provider.calls] == [1, 1, 1]


async def test_a_realtime_provider_without_tools_reads_none() -> None:
    class Toolless(MockRealtimeProvider):
        @property
        def supports_tools(self) -> bool:
            return False

    provider = Toolless()
    rt = RealtimeVoiceChannel(
        "rt1",
        provider=provider,
        transport=MockRealtimeTransport(),
        system_prompt=HOST,
        server_instructions={"gm": GM},
    )
    async with _in_session(rt):
        assert _connected_with(provider) == HOST


async def test_a_realtime_skill_activation_keeps_them_once(tmp_path: Path) -> None:
    provider = MockRealtimeProvider()
    rt = RealtimeVoiceChannel(
        "rt1",
        provider=provider,
        transport=MockRealtimeTransport(),
        system_prompt=HOST,
        skills=_skills(tmp_path),
        skill_delivery_mode="on_demand",
        server_instructions={"gm": GM},
    )
    async with _in_session(rt) as session:
        provider.reconfigure = AsyncMock()  # type: ignore[method-assign]
        await provider.simulate_tool_call(session, "c1", "activate_skill", {"name": "combat"})
        await settled(rt)  # the reconfigure the activation schedules after its result

        sent = provider.reconfigure.call_args.kwargs["system_prompt"]
        assert sent.count(GM) == 1 and "Fight well." in sent
        assert sent.index(GM) < sent.index("Fight well.")


async def test_a_prompt_a_turn_config_sets_keeps_them() -> None:
    async def turn(_binding: ChannelBinding, _context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(system_prompt="Turn prompt.")

    prompt = await _ai_prompt(config_provider=turn, server_instructions={"gm": GM})

    assert HOST not in prompt
    assert prompt.index("Turn prompt.") < prompt.index(GM)


async def test_a_reasoning_backends_agent_reads_its_own() -> None:
    """The agent a reasoning backend drives composes its model's prompt: the
    voice session's tools it calls are the servers' (RFC §12.4.1, §21.2)."""
    model = MockAIProvider(responses=["The goblin is hit."])
    agent = Agent("reasoner", provider=model, system_prompt=HOST, server_instructions={"gm": GM})
    session = VoiceSession(
        id="s1",
        room_id="r1",
        participant_id="u1",
        channel_id="rt1",
        state=VoiceSessionState.ACTIVE,
    )
    request = ReasoningRequest(session, "d1", [TranscriptLine("user", "I attack.")], True)

    _ = [output async for output in AgentReasoningBackend(agent).run(request)]

    prompt = model.calls[0].system_prompt or ""
    assert prompt.index(HOST) < prompt.index(GM)
    assert prompt.count(GM) == 1


def test_a_pipeline_agent_carrying_them_is_refused_for_the_channels() -> None:
    """A realtime pipeline's session prompt is the channel's: an agent's
    instructions would be dropped while its tools are served (RFC §19.5)."""
    teller = Agent("teller", provider=MockAIProvider(), server_instructions={"gm": GM})
    quiet = Agent("quiet", provider=MockAIProvider(), server_instructions={"gm": None})
    kit = RoomKit()
    kit.register_channel(
        RealtimeVoiceChannel(
            "rtv", provider=MockRealtimeProvider(), transport=MockRealtimeTransport()
        )
    )
    for agent in (teller, quiet):
        kit.register_channel(agent)

    def install(agents: list[Agent]) -> None:
        stages = [PipelineStage(phase=a.channel_id, agent_id=a.channel_id) for a in agents]
        ConversationPipeline(stages=stages).install(kit, agents, voice_channel_id="rtv")

    with pytest.raises(ValueError) as refused:
        install([teller])
    assert "Agent 'teller' carries MCP server instructions," in str(refused.value)
    assert "server_instructions=..." in str(refused.value)
    install([quiet])  # published nothing: nothing to drop
