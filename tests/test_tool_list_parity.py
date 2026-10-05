"""One tool list, one set of rules, on a text turn and a realtime session (RMK-397).

The same configuration declares the same tools and applies the same rules on
both paths: skills a host marked unavailable, a skill's ``requires``, a
pipeline that reads the channel's tools as they are, ``find_tools``'
related names, a never-hidden tool's origin, the ``list_tools`` inventory, an
infrastructure tool the turn does not offer, and the names a handler reads
from ``current_tool_allowed_names()`` (RFC §6.4, §12.4, §19.5, §21.1, §21.4,
§24.3).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from roomkit import ConferenceRealtimeConfig, RoomKit
from roomkit.channels._tool_search_constants import FIND_TOOLS_SCHEMA
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.core.exceptions import ToolNameCollisionError
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.models.tool_call import AIResponseEvent
from roomkit.orchestration.pipeline import ConversationPipeline, PipelineStage
from roomkit.providers.ai.base import AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
from roomkit.tasks.delegate import DelegateHandler, setup_realtime_delegation
from roomkit.tools import current_tool_allowed_names
from roomkit.tools.human_input import HumanInputToolHandler
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.conftest import make_event
from tests.test_tool_search_orchestration import (
    CRM,
    _calling,
    _crm,
    _FixedProvider,
    _model,
    _orchestrated,
    _turn,
)
from tests.tool_loop_modes import respond

REASON = "requires tool(s) not available in this context: calendar"


def _dicts(tools: list[AITool]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools
    ]


ASK = AITool(name="ask", description="the channel's", parameters={})


def _schema(name: str, description: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "description": description or name,
        "parameters": {"type": "object", "properties": {}},
    }


async def _session(
    channel: RealtimeVoiceChannel, provider: MockRealtimeProvider
) -> tuple[RoomKit, Any]:
    kit = RoomKit()
    kit.register_channel(channel)
    room = await kit.create_room()
    await kit.attach_channel(room.id, channel.channel_id)
    return kit, await channel.start_session(room.id, "u", "ws")


async def _call(
    channel: RealtimeVoiceChannel,
    provider: MockRealtimeProvider,
    session: Any,
    name: str,
    arguments: dict[str, Any],
) -> str:
    await provider.simulate_tool_call(session, f"c-{name}", name, arguments)
    await asyncio.wait_for(asyncio.gather(*list(channel._scheduled_tasks)), 5)
    return provider.tool_results[-1][2]


def _text_result(channel: AIChannel, first: int) -> str:
    messages = _model(channel).calls[first + 1].messages
    return next(str(p.result) for m in messages if m.role == "tool" for p in m.content)


class TestUnavailableSkills:
    """A registry whose every skill is unavailable still says why (§24.3)."""

    @staticmethod
    def _registry(tmp: Path) -> SkillRegistry:
        folder = tmp / "test-skill"
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\nBody.", encoding="utf-8"
        )
        registry = SkillRegistry()
        registry.discover(tmp)
        registry.mark_unavailable("test-skill", REASON)
        return registry

    async def test_a_text_turn_gives_the_reason(self, tmp_path: Path) -> None:
        provider = MockAIProvider(responses=["ok"])
        channel = AIChannel("ai1", provider=provider, skills=self._registry(tmp_path))
        binding = ChannelBinding(
            channel_id="ai1",
            room_id="r1",
            channel_type=ChannelType.AI,
            category=ChannelCategory.INTELLIGENCE,
        )

        await respond(
            channel,
            make_event(body="go", channel_id="sms1"),
            binding,
            RoomContext(room=Room(id="r1")),
        )

        [context] = provider.calls
        assert REASON in (context.system_prompt or "")
        assert {"activate_skill", "read_skill_reference"} <= {t.name for t in context.tools}

    async def test_a_realtime_session_gives_the_reason(self, tmp_path: Path) -> None:
        provider = MockRealtimeProvider()
        channel = RealtimeVoiceChannel(
            "rt",
            provider=provider,
            transport=MockRealtimeTransport(),
            skills=self._registry(tmp_path),
        )
        kit, _ = await _session(channel, provider)

        connect = next(c for c in provider.calls if c.method == "connect")
        assert REASON in (connect.args["system_prompt"] or "")
        assert {"activate_skill", "read_skill_reference"} <= {
            t["name"] for t in connect.args["tools"] or []
        }
        await kit.close()


async def test_a_skill_requires_what_orchestration_set_up(tmp_path: Path) -> None:
    """``requires`` is checked against what the session declares (§24.3)."""
    folder = tmp_path / "deleg-skill"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: deleg-skill\ndescription: Delegates\nrequires: delegate_task\n---\n"
        "Call delegate_task.",
        encoding="utf-8",
    )
    registry = SkillRegistry()
    registry.discover(tmp_path)
    provider = MockRealtimeProvider()
    provider.reconfigure = AsyncMock()  # type: ignore[method-assign]
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        skills=registry,
        tool_handler=AsyncMock(return_value="ok"),
        skill_delivery_mode="on_demand",
    )
    setup_realtime_delegation(channel, DelegateHandler(MagicMock()))
    kit, session = await _session(channel, provider)

    result = await _call(channel, provider, session, "activate_skill", {"name": "deleg-skill"})

    assert "error" not in json.loads(result)
    await kit.close()


class TestPipelineTools:
    """A pipeline declares the channel's tools as they are now (§19.5)."""

    @staticmethod
    def _install(rtv: RealtimeVoiceChannel, agent: Agent) -> RoomKit:
        kit = MagicMock()
        kit.hook.return_value = MagicMock()
        kit.channels = {"rtv": rtv}
        kit.get_room = AsyncMock(return_value=Room(id="r1"))
        pipeline = ConversationPipeline(stages=[PipelineStage(phase="a", agent_id="agent-a")])
        pipeline.install(kit, [agent], voice_channel_id="rtv")
        return kit

    async def test_a_channel_configured_after_the_install_keeps_its_tools(self) -> None:
        rtv = RealtimeVoiceChannel(
            "rtv",
            provider=MockRealtimeProvider(),
            transport=MockRealtimeTransport(),
            tools=[_schema("old_lookup")],
        )
        self._install(rtv, Agent("agent-a", role="A", system_prompt="Be A."))

        rtv.configure(tools=[_schema("new_lookup")])
        config = await rtv._room_session_config("r1")

        assert config is not None
        assert [t["name"] for t in config.tools or []] == ["new_lookup", "handoff_conversation"]

    @pytest.mark.parametrize(
        ("options", "name", "declared"),
        [
            pytest.param(
                {"tools": [_schema("lookup", "the channel's")]},
                "lookup",
                "the channel's",
                id="host",
            ),
            pytest.param(
                {
                    "human_input_handler": HumanInputToolHandler(
                        tool_names={"ask"}, tool_definitions=[ASK]
                    )
                },
                "ask",
                "the channel's",
                id="human-input",
            ),
            pytest.param(
                {"human_input_handler": HumanInputToolHandler(tool_names={"ask"})},
                "ask",
                None,
                id="human-input-undeclared",
            ),
            pytest.param(
                {"tools": [_schema("other")], "tool_search": True},
                "find_tools",
                FIND_TOOLS_SCHEMA["description"],
                id="channel-own",
            ),
        ],
    )
    async def test_a_name_the_channel_carries_is_the_channels(
        self,
        options: dict[str, Any],
        name: str,
        declared: str | None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """One schema, one server (§21.1): the agent's tool of that name is
        neither declared nor served, a warning names it, and the agent's other
        tools are installed and served, whatever tool of the channel's
        carries the name, declared in the session or not (RMK-517)."""
        provider = MockRealtimeProvider()
        rtv = RealtimeVoiceChannel(
            "rtv",
            provider=provider,
            transport=MockRealtimeTransport(),
            tool_handler=AsyncMock(return_value="channel"),
            **options,
        )
        agent = Agent(
            "agent-a",
            role="A",
            tools=[
                AITool(name=name, description="the agent's", parameters={}),
                AITool(name="balance", description="the agent's own", parameters={}),
            ],
            tool_handler=AsyncMock(return_value="agent"),
            tool_search=False,
        )
        kit = RoomKit()
        kit.register_channel(rtv)
        kit.register_channel(agent)
        pipeline = ConversationPipeline(stages=[PipelineStage(phase="a", agent_id="agent-a")])

        with caplog.at_level(logging.WARNING, logger="roomkit.orchestration.pipeline"):
            pipeline.install(kit, [agent], voice_channel_id="rtv")
        room = await kit.create_room()
        await kit.attach_channel(room.id, "rtv")
        session = await rtv.start_session(room.id, "u", "ws")

        connect = next(c for c in provider.calls if c.method == "connect")
        descriptions = {t["name"]: t["description"] for t in connect.args["tools"] or []}
        assert descriptions.get(name) == declared
        # Declared, or behind find_tools when Tool Search holds the catalogue.
        assert descriptions.get("balance", "the agent's own") == "the agent's own"
        assert "shares its name with a tool of channel rtv" in caplog.text
        assert await _call(rtv, provider, session, "balance", {}) == "agent"
        await kit.close()

    async def test_the_channel_cannot_take_an_agents_name_later(self) -> None:
        """The registry refuses it, so no call is left between two servers."""
        rtv = RealtimeVoiceChannel(
            "rtv",
            provider=MockRealtimeProvider(),
            transport=MockRealtimeTransport(),
            tools=[_schema("other")],
            tool_handler=AsyncMock(return_value="channel"),
        )
        agent = Agent(
            "agent-a",
            role="A",
            tools=[AITool(name="lookup", description="the agent's", parameters={})],
            tool_handler=AsyncMock(return_value="agent"),
        )
        self._install(rtv, agent)

        with pytest.raises(ToolNameCollisionError):
            rtv.configure(tools=[_schema("lookup", "the channel's")])


class TestFindTools:
    """``find_tools`` never names what it never returns (§21.1)."""

    PINNED = AITool(
        name="crm_pinned_summary",
        description="Summary of the customer.",
        parameters={"type": "object", "properties": {}},
    )

    async def test_a_text_search_names_no_pinned_tool_as_related(self) -> None:
        channel = _orchestrated(
            False,
            [self.PINNED, *CRM],
            _calling("find_tools", query="crm operation number 3"),
            tool_search=True,
            tool_search_pinned=["crm_pinned_summary"],
        )

        result = json.loads(_text_result(channel, await _turn(channel, "r1")))

        assert result["matches"]
        assert "crm_pinned_summary" not in (result.get("related_tools_same_source") or [])

    async def test_a_realtime_search_names_no_pinned_tool_as_related(self) -> None:
        provider = MockRealtimeProvider()
        channel = RealtimeVoiceChannel(
            "voice",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=_dicts([self.PINNED, *CRM]),
            tool_handler=_crm,
            tool_search=True,
            tool_search_pinned=["crm_pinned_summary"],
        )
        kit, session = await _session(channel, provider)

        result = json.loads(
            await _call(channel, provider, session, "find_tools", {"query": "crm operation 3"})
        )

        assert result["matches"]
        assert "crm_pinned_summary" not in (result.get("related_tools_same_source") or [])
        await kit.close()


async def test_a_tool_tool_search_never_hides_is_always_declared() -> None:
    """``plan_tasks`` is reported ``always``, its first use aside (§6.4)."""
    plan = _calling("plan_tasks", tasks=[{"title": "one", "status": "pending"}])
    channel = _orchestrated(False, CRM, plan, tool_search=True, enable_planning=True)
    seen: list[AIResponseEvent] = []

    async def observe(event: AIResponseEvent) -> None:
        seen.append(event)

    channel._after_response_hook = observe
    await _turn(channel, "r1")
    await _turn(channel, "r1")

    origins = [{t.name: t.origin for t in event.declared_tools}["plan_tasks"] for event in seen]
    assert origins == ["always", "always"]


class TestListTools:
    """``list_tools`` lists every tool the turn or session can call (§21.1)."""

    async def test_a_text_turn_lists_the_tools_always_declared(self) -> None:
        channel = _orchestrated(False, CRM, _calling("list_tools"), tool_search=True)

        result = json.loads(_text_result(channel, await _turn(channel, "r1")))

        names = {tool["name"] for tool in result["tools"]}
        assert "delegate_task" in names
        assert not names & {"find_tools", "list_tools"}

    async def test_a_realtime_session_lists_the_tools_always_declared(self) -> None:
        provider = MockRealtimeProvider()
        channel = RealtimeVoiceChannel(
            "voice",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=_dicts(CRM),
            tool_handler=_crm,
            tool_search=True,
        )
        setup_realtime_delegation(channel, DelegateHandler(MagicMock()))
        kit, session = await _session(channel, provider)

        result = json.loads(await _call(channel, provider, session, "list_tools", {}))

        names = {tool["name"] for tool in result["tools"]}
        assert "delegate_task" in names
        assert {t.name for t in CRM} <= names
        assert not names & {"find_tools", "list_tools"}
        await kit.close()


class TestAnUndeclaredInfrastructureTool:
    """Tool Search hiding nothing, ``find_tools`` is refused as undeclared (§6.4, §12.4)."""

    async def test_a_text_turn_refuses_it(self) -> None:
        channel = _orchestrated(False, CRM[:2], _calling("find_tools", query="crm"))

        result = json.loads(_text_result(channel, await _turn(channel, "r1")))

        assert result == {"error": "Tool 'find_tools' is not declared."}

    async def test_a_realtime_session_refuses_it(self) -> None:
        provider = MockRealtimeProvider()
        channel = RealtimeVoiceChannel(
            "voice",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=[_schema("lookup")],
            tool_handler=AsyncMock(return_value="ok"),
        )
        kit, session = await _session(channel, provider)

        result = json.loads(await _call(channel, provider, session, "find_tools", {"query": "x"}))

        assert result == {"error": "Tool 'find_tools' is not declared."}
        await kit.close()

    async def test_a_realtime_session_refuses_a_script_with_no_executor(
        self, tmp_path: Path
    ) -> None:
        folder = tmp_path / "scripted"
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            "---\nname: scripted\ndescription: Runs scripts\n---\nBody.", encoding="utf-8"
        )
        registry = SkillRegistry()
        registry.discover(tmp_path)
        provider = MockRealtimeProvider()
        channel = RealtimeVoiceChannel(
            "voice",
            provider=provider,
            transport=MockRealtimeTransport(),
            skills=registry,
            tool_handler=AsyncMock(return_value="ok"),
        )
        kit, session = await _session(channel, provider)

        result = json.loads(
            await _call(
                channel,
                provider,
                session,
                "run_skill_script",
                {"skill_name": "scripted", "script_name": "x.sh"},
            )
        )

        assert result == {"error": "Tool 'run_skill_script' is not declared."}
        await kit.close()


async def test_a_realtime_handler_reads_the_sessions_tools() -> None:
    """``current_tool_allowed_names()`` on a realtime door (§21.4)."""
    seen: list[set[str] | None] = []

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        seen.append(current_tool_allowed_names())
        return "ok"

    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "voice",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=[_schema("lookup"), _schema("book")],
        tool_handler=handler,
    )
    setup_realtime_delegation(channel, DelegateHandler(MagicMock()))
    kit, session = await _session(channel, provider)

    await _call(channel, provider, session, "lookup", {})

    assert seen == [{"lookup", "book", "delegate_task"}]
    await kit.close()


class TestReviewedEdges:
    """Edges the review of RMK-397 found."""

    async def test_a_provider_native_tool_leaves_every_call_served(self) -> None:
        """A tool with no name (a provider's own, ``google_search``) is kept,
        and the calls beside it still run (§21.4)."""
        seen: list[set[str] | None] = []

        async def handler(name: str, arguments: dict[str, Any]) -> str:
            seen.append(current_tool_allowed_names())
            return "ok"

        provider = MockRealtimeProvider()
        channel = RealtimeVoiceChannel(
            "voice",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=[{"google_search": {}}, _schema("lookup")],
            tool_handler=handler,
        )
        kit, session = await _session(channel, provider)

        result = await _call(channel, provider, session, "lookup", {})

        assert result == "ok"
        assert seen == [{"lookup"}]
        await kit.close()

    @pytest.mark.parametrize(
        ("tools", "expected"),
        [([{"google_search": {}}, {"name": "hr_lookup"}], {"hr_lookup"}), ([], None)],
        ids=["native-tool", "no-catalogue"],
    )
    async def test_a_conference_handler_reads_the_sessions_tools(
        self, tools: list[dict[str, Any]], expected: set[str] | None
    ) -> None:
        """A conference declaring no catalogue admits any name, and has none."""
        seen: list[set[str] | None] = []

        async def handler(room_id: str, name: str, arguments: dict[str, Any]) -> str:
            seen.append(current_tool_allowed_names())
            return "ok"

        provider = MockRealtimeProvider()
        config = ConferenceRealtimeConfig(provider=provider, tools=tools, tool_handler=handler)
        kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
        session = await channel._realtime.ensure_session(ROOM)
        assert session is not None

        await provider.simulate_tool_call(session, "c1", "hr_lookup", {})
        await until(lambda: bool(seen))

        assert seen == [expected]
        await kit.close()

    async def test_a_realtime_inventory_lists_the_skill_tools(self, tmp_path: Path) -> None:
        folder = tmp_path / "guide"
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            "---\nname: guide\ndescription: A guide\n---\nBody.", encoding="utf-8"
        )
        registry = SkillRegistry()
        registry.discover(tmp_path)
        provider = MockRealtimeProvider()
        channel = RealtimeVoiceChannel(
            "voice",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=_dicts(CRM),
            tool_handler=_crm,
            tool_search=True,
            skills=registry,
        )
        kit, session = await _session(channel, provider)

        result = json.loads(await _call(channel, provider, session, "list_tools", {}))

        names = {tool["name"] for tool in result["tools"]}
        assert {"activate_skill", "read_skill_reference"} <= names
        await kit.close()

    async def test_a_fixed_declaration_session_reaches_what_it_lists(self) -> None:
        """``list_tools(name=...)`` and ``call_tool`` reach every tool the
        inventory lists (§21.1)."""
        provider = _FixedProvider()
        channel = RealtimeVoiceChannel(
            "voice",
            provider=provider,
            transport=MockRealtimeTransport(),
            tools=_dicts(CRM),
            tool_handler=_crm,
            tool_search=True,
        )
        setup_realtime_delegation(channel, DelegateHandler(MagicMock()))
        kit, session = await _session(channel, provider)
        search = channel._tool_search_support
        assert search is not None and search.uses_call_tool

        listed = json.loads(
            await _call(channel, provider, session, "list_tools", {"name": "delegate_task"})
        )
        _, _, error = search.unwrap_call(
            {"name": "delegate_task", "arguments_json": "{}"}, session.id
        )

        assert listed["tool"]["name"] == "delegate_task"
        assert error is None
        await kit.close()

    async def test_a_shadowed_agent_tool_stays_undeclared_once_the_channel_drops_it(
        self,
    ) -> None:
        """Nothing of the agent's serves it, so its schema is not declared."""
        rtv = RealtimeVoiceChannel(
            "rtv",
            provider=MockRealtimeProvider(),
            transport=MockRealtimeTransport(),
            tools=[_schema("lookup", "the channel's")],
            tool_handler=AsyncMock(return_value="channel"),
        )
        agent = Agent(
            "agent-a",
            role="A",
            tools=[AITool(name="lookup", description="the agent's", parameters={})],
            tool_handler=AsyncMock(return_value="agent"),
        )
        TestPipelineTools._install(rtv, agent)

        rtv.configure(tools=[_schema("other")])
        config = await rtv._room_session_config("r1")

        assert config is not None
        assert [t["name"] for t in config.tools or []] == ["other", "handoff_conversation"]

    async def test_a_handoff_reads_the_channels_tools_as_they_are(self) -> None:
        rtv = RealtimeVoiceChannel(
            "rtv",
            provider=MockRealtimeProvider(),
            transport=MockRealtimeTransport(),
            tools=[_schema("old_lookup")],
        )
        TestPipelineTools._install(rtv, Agent("agent-a", role="A", system_prompt="Be A."))
        handoff = rtv._registry.lookup("handoff_conversation", "r1")
        assert handoff is not None and handoff.serve is not None
        wiring = handoff.serve.__self__
        session = MagicMock()
        rtv.get_room_sessions = MagicMock(return_value=[session])  # type: ignore[method-assign]
        rtv.reconfigure_session = AsyncMock()  # type: ignore[method-assign]
        wiring._greet_on_handoff = False

        rtv.configure(tools=[_schema("new_lookup")])
        await wiring.on_handoff_complete("r1", SimpleNamespace(new_agent_id="agent-a"))

        tools = rtv.reconfigure_session.await_args.kwargs["tools"]
        assert [t["name"] for t in tools] == ["new_lookup", "handoff_conversation"]
