"""``current_tool_allowed_names()`` lists what the turn's policy admits, on
every door (RMK-420, RFC §21.4).

A tool the policy denies the turn's actor is left out on a text turn, a
realtime session and a conference alike: the gate refuses it before any
handler. A tool a skill keeps closed stays in, since activating the skill
opens it, and so does a tool that escapes the policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from roomkit import ConferenceRealtimeConfig
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
from roomkit.tools import current_tool_allowed_names
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext
from roomkit.tools.policy import RoleOverride, ToolPolicy
from roomkit.voice.realtime.mock import MockRealtimeProvider
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.conftest import make_event
from tests.test_realtime_tool_policy import TOOLS, _Calls, _channel
from tests.tool_loop_modes import respond

DENY_DELETE = ToolPolicy(deny=["delete_*"])
OBSERVER_DENIES_DELETE = ToolPolicy(role_overrides={"observer": RoleOverride(deny=["delete_*"])})


class _Seen(_Calls):
    """A host handler that reads the names the call may reach."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[set[str] | None] = []

    async def handler(self, name: str, arguments: dict[str, Any]) -> str:
        self.names.append(current_tool_allowed_names())
        return await super().handler(name, arguments)


def _skill(tmp_path: Path) -> SkillRegistry:
    folder = tmp_path / "publisher"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: publisher\ndescription: Publishes\n"
        "allowed_tools: publish_site\n---\nPublish carefully.",
        encoding="utf-8",
    )
    registry = SkillRegistry()
    registry.discover(tmp_path)
    return registry


def _lookup_then_done(streaming: bool) -> MockAIProvider:
    return MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name="lookup_account", arguments={})],
            ),
            AIResponse(content="done"),
        ],
        streaming=streaming,
    )


def _binding() -> ChannelBinding:
    return ChannelBinding(
        channel_id="ai1",
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )


async def test_a_text_turn_lists_what_its_policy_admits(tmp_path: Path, streaming: bool) -> None:
    seen = _Seen()
    provider = _lookup_then_done(streaming)
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[AITool(**tool) for tool in TOOLS] + [AITool(name="publish_site", description="d")],
        tool_handler=seen.handler,
        # An allow-list: the skill's own door escapes it, as it escapes any policy.
        tool_policy=ToolPolicy(allow=["lookup_*", "publish_site"]),
        skills=_skill(tmp_path),
        tool_search=False,
    )

    await respond(
        channel,
        make_event(body="go", channel_id="sms1"),
        _binding(),
        RoomContext(room=Room(id="r1")),
    )

    [names] = seen.names
    assert names is not None
    assert "delete_account" not in names
    # Gated by a skill not yet activated, and the skill's own door.
    assert {"lookup_account", "publish_site", "activate_skill"} <= names


@pytest.mark.parametrize(("role", "denied"), [("observer", True), ("member", False)])
async def test_a_text_turn_resolves_the_policy_for_its_participant(
    role: str, denied: bool, streaming: bool
) -> None:
    seen = _Seen()
    channel = AIChannel(
        "ai1",
        provider=_lookup_then_done(streaming),
        tools=[AITool(**tool) for tool in TOOLS],
        tool_handler=seen.handler,
        tool_policy=OBSERVER_DENIES_DELETE,
        tool_search=False,
    )
    event = make_event(body="go", channel_id="sms1", participant_id="u1")
    context = RoomContext(
        room=Room(id="r1"),
        participants=[Participant(id="u1", room_id="r1", channel_id="sms1", role=role)],
    )

    await respond(channel, event, _binding(), context)

    [names] = seen.names
    assert names is not None
    assert ("delete_account" in names) is not denied
    assert "lookup_account" in names


@pytest.mark.parametrize(
    ("role", "denied"),
    [(None, True), ("observer", True), ("member", False)],
    ids=["policy", "role-denied", "role-admitted"],
)
async def test_a_realtime_call_lists_what_the_sessions_policy_admits(
    role: str | None, denied: bool
) -> None:
    policy = DENY_DELETE if role is None else OBSERVER_DENIES_DELETE
    seen = _Seen()
    kit, channel, provider, session = await _channel(seen, policy=policy, role=role)

    await provider.simulate_tool_call(session, "c1", "lookup_account", {})
    await until(lambda: bool(seen.names))

    assert seen.names == [{"lookup_account"} if denied else {"lookup_account", "delete_account"}]
    await kit.close()


async def test_a_conference_call_lists_what_its_policy_admits() -> None:
    seen = _Seen()
    provider = MockRealtimeProvider()
    config = ConferenceRealtimeConfig(
        provider=provider, tools=TOOLS, tool_handler=seen.conference, tool_policy=DENY_DELETE
    )
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    session = await channel._realtime.ensure_session(ROOM)
    assert session is not None

    await provider.simulate_tool_call(session, "c1", "lookup_account", {})
    await until(lambda: bool(seen.names))

    assert seen.names == [{"lookup_account"}]
    await kit.close()


async def test_a_realtime_call_lists_the_tools_that_escape_the_policy(tmp_path: Path) -> None:
    seen = _Seen()
    kit, channel, provider, session = await _channel(
        seen, policy=ToolPolicy(allow=["lookup_*"]), skills=_skill(tmp_path)
    )

    await provider.simulate_tool_call(session, "c1", "lookup_account", {})
    await until(lambda: bool(seen.names))

    [names] = seen.names
    assert names is not None
    assert "delete_account" not in names
    assert {"lookup_account", "activate_skill"} <= names
    await kit.close()


async def test_a_driven_turn_lists_what_the_agents_policy_admits() -> None:
    """The toolset a reasoning backend drives its agent with (RFC §12.4.1)."""
    agent = Agent(
        "reasoner",
        provider=MockAIProvider(responses=["done"]),
        tool_policy=DENY_DELETE,
    )
    loop_ctx = _ToolLoopContext(room_id="r1")
    await agent._driven_turn(_binding(), loop_ctx, [], [AITool(**tool) for tool in TOOLS])

    token = _current_loop_ctx.set(loop_ctx)
    try:
        assert current_tool_allowed_names() == {"lookup_account"}
    finally:
        _current_loop_ctx.reset(token)
