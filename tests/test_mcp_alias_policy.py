"""A call under an MCP alias is judged under the tool it runs (RMK-483, RFC §21.1).

``MCPToolProvider.as_tool_handler()`` serves ``mcp__<server>__<tool>`` as
``<tool>``. The tool policy and a skill's gate judge both names, on every
door, so a tool they refuse never runs under its alias; the alias itself is
still served for a tool they admit. A provider given a ``label`` serves only
the aliases naming its server, so a policy that names a server holds when
every provider of the chain is labelled (RMK-493); a bare name runs on the
first labelled server whose alias the policy admits (RMK-957).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mcp.types as mt
import pytest

from roomkit import ConferenceRealtimeConfig, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.channels.realtime_voice import RealtimeVoiceChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AITool, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.compose import ToolHandler, compose_tool_handlers
from roomkit.tools.external import PolicyExternalToolHandler
from roomkit.tools.mcp import MCPToolProvider
from roomkit.tools.policy import (
    RoleOverride,
    ToolPolicy,
    alias_admitted,
    judged_names,
    policy_refusal,
    served_tool_name,
)
from roomkit.voice.realtime.mock import MockRealtimeProvider, MockRealtimeTransport
from tests.conference.test_conference_realtime import ROOM, realtime_kit, until
from tests.conftest import make_event
from tests.test_realtime_skills import _registry_with_skill
from tests.tool_doors import DOORS as EVERY_DOOR
from tests.tool_doors import run_door
from tests.tool_loop_modes import respond

ALIAS = "mcp__crm__delete_records"


class _Server:
    def __init__(self) -> None:
        self.ran: list[str] = []

    async def send_request(self, request: Any, result_type: Any) -> Any:
        name = request.params.name
        self.ran.append(name)
        return mt.CallToolResult(
            content=[mt.TextContent(type="text", text=f"{name} done")], isError=False
        )


def _mcp_handler(
    server: _Server, label: str | None = None, *, gate_discovery: bool = True
) -> ToolHandler:
    provider = MCPToolProvider("http://crm.invalid/mcp", label=label)
    provider._connected = True
    provider._session = server  # type: ignore[assignment]
    provider._tools = [
        AITool(name=n, description=n, parameters={}) for n in ("search_records", "delete_records")
    ]
    provider._tool_set = {"search_records", "delete_records"}
    return provider.as_tool_handler(gate_discovery=gate_discovery)


def test_an_alias_is_judged_under_both_names() -> None:
    assert served_tool_name(ALIAS) == "delete_records"
    assert served_tool_name("delete_records") == "delete_records"
    assert judged_names(ALIAS) == (ALIAS, "delete_records")
    assert judged_names("search_records") == ("search_records",)


@pytest.mark.parametrize(
    ("policy", "kept_out"),
    [
        (ToolPolicy(deny=["mcp__b__*"]), {"b"}),
        (ToolPolicy(allow=["mcp__a__*", "lookup"]), {"b"}),
        (ToolPolicy(allow=["lookup"]), set()),
        (
            ToolPolicy(
                role_overrides={
                    "member": RoleOverride(mode="restrict", allow=["mcp__a__*", "lookup"])
                }
            ).resolve("member"),
            {"b"},
        ),
    ],
    ids=["deny", "allow-aliases", "allow-bare", "role-restrict"],
)
def test_a_policy_keeps_out_only_the_servers_it_names(
    policy: ToolPolicy, kept_out: set[str]
) -> None:
    """Which labelled servers a policy keeps a bare name off (RMK-957): one a
    deny pattern or an allow list of aliases names, a role's restricting
    list included; an allow list of bare names alone names none."""
    assert {s for s in ("a", "b") if not alias_admitted(policy, f"mcp__{s}__lookup")} == kept_out


async def _on_session(
    policy: ToolPolicy | None,
    handler: ToolHandler,
    name: str,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tools=tools,
        tool_handler=handler,
        tool_policy=policy,
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")
    await provider.simulate_tool_call(session, "c1", name, {})
    await until(lambda: bool(provider.tool_results))
    await kit.close()
    return provider.tool_results[0][2]


async def _on_conference(policy: ToolPolicy | None, handler: ToolHandler, name: str) -> str:
    provider = MockRealtimeProvider()

    async def room_handler(room_id: str, tool: str, arguments: dict[str, Any]) -> Any:
        return await handler(tool, arguments)

    config = ConferenceRealtimeConfig(
        provider=provider, tool_handler=room_handler, tool_policy=policy
    )
    kit, channel, _, _ = await realtime_kit(provider=provider, config=config)
    session = await channel._realtime.ensure_session(ROOM)
    await provider.simulate_tool_call(session, "c1", name, {})
    await until(lambda: bool(provider.tool_results))
    await kit.close()
    return provider.tool_results[0][2]


DOORS = {"session": _on_session, "conference": _on_conference}


@pytest.mark.parametrize("door", list(DOORS))
@pytest.mark.parametrize(
    "policy",
    [ToolPolicy(deny=["delete_*"]), ToolPolicy(allow=["search_*"])],
    ids=["deny", "allow"],
)
async def test_a_tool_the_policy_refuses_never_runs_under_its_alias(
    door: str, policy: ToolPolicy
) -> None:
    server = _Server()

    answer = await DOORS[door](policy, _mcp_handler(server), ALIAS)

    assert "not permitted by the agent's tool policy" in answer
    assert server.ran == []


@pytest.mark.parametrize("door", list(DOORS))
async def test_an_alias_of_an_admitted_tool_is_still_served(door: str) -> None:
    server = _Server()

    answer = await DOORS[door](
        ToolPolicy(deny=["delete_*"]), _mcp_handler(server), "mcp__crm__search_records"
    )

    assert "search_records done" in answer
    assert server.ran == ["search_records"]


def _two_servers(*, labelled: bool) -> tuple[_Server, _Server, ToolHandler]:
    """Servers a and b, each serving ``delete_records``, composed with b first."""
    a, b = _Server(), _Server()
    handler = compose_tool_handlers(
        _mcp_handler(b, "b" if labelled else None), _mcp_handler(a, "a" if labelled else None)
    )
    return a, b, handler


SCOPED_TO_A = [
    ToolPolicy(deny=["mcp__b__*"]),
    ToolPolicy(allow=["mcp__a__*", "delete_records"]),
]


@pytest.mark.parametrize("door", list(DOORS))
@pytest.mark.parametrize("policy", SCOPED_TO_A, ids=["deny-b", "allow-a"])
async def test_a_policy_scoped_to_one_server_keeps_the_call_off_the_other(
    door: str, policy: ToolPolicy
) -> None:
    a, b, handler = _two_servers(labelled=True)

    on_a = await DOORS[door](policy, handler, "mcp__a__delete_records")
    on_b = await DOORS[door](policy, handler, "mcp__b__delete_records")

    assert "delete_records done" in on_a
    assert "not permitted by the agent's tool policy" in on_b
    assert (a.ran, b.ran) == (["delete_records"], [])


async def test_a_policy_that_names_no_server_applies_to_both() -> None:
    a, b, handler = _two_servers(labelled=True)

    refused = [
        await _on_session(ToolPolicy(deny=["delete_*"]), handler, f"mcp__{server}__delete_records")
        for server in ("a", "b")
    ]
    assert all("not permitted by the agent's tool policy" in answer for answer in refused)
    assert (a.ran, b.ran) == ([], [])

    admitting = ToolPolicy(allow=["mcp__*", "delete_records"])
    for server in ("a", "b"):
        await _on_session(admitting, handler, f"mcp__{server}__delete_records")
    assert (a.ran, b.ran) == (["delete_records"], ["delete_records"])


@pytest.mark.parametrize("policy", SCOPED_TO_A, ids=["deny-b", "allow-a"])
async def test_a_catalogue_of_aliases_refuses_a_bare_name(policy: ToolPolicy) -> None:
    """A policy that names a server holds for a session that declares the
    tools under their aliases alone: a bare name names no server, and is
    refused as undeclared rather than run on the chain's first server."""
    a, b, handler = _two_servers(labelled=True)
    aliases = [
        {"name": f"mcp__{server}__delete_records", "description": "delete", "parameters": {}}
        for server in ("a", "b")
    ]

    answer = await _on_session(policy, handler, "delete_records", tools=aliases)

    assert "Tool 'delete_records' is not declared." in answer
    assert (a.ran, b.ran) == ([], [])


def _lookup_servers() -> tuple[_Server, _Server, ToolHandler]:
    """Labelled servers a and b, each serving ``lookup``, composed with b first."""
    a, b = _Server(), _Server()
    handlers = []
    for server, label in ((b, "b"), (a, "a")):
        provider = MCPToolProvider("http://records.invalid/mcp", label=label)
        provider._connected = True
        provider._session = server  # type: ignore[assignment]
        provider._tools = [AITool(name="lookup", description="Look up", parameters={})]
        provider._tool_set = {"lookup"}
        handlers.append(provider.as_tool_handler())
    return a, b, compose_tool_handlers(*handlers)


@pytest.mark.parametrize("door", EVERY_DOOR)
@pytest.mark.parametrize(
    ("policy", "ran"),
    [
        (ToolPolicy(deny=["mcp__b__*"]), (["lookup"], [])),
        (ToolPolicy(allow=["mcp__a__*", "lookup"]), (["lookup"], [])),
        (ToolPolicy(deny=["mcp__a__*", "mcp__b__*"]), ([], [])),
        (ToolPolicy(allow=["lookup"]), ([], ["lookup"])),
    ],
    ids=["deny-b", "allow-a", "deny-both", "allow-bare"],
)
async def test_a_bare_name_runs_on_the_first_server_the_policy_admits(
    door: str, policy: ToolPolicy, ran: tuple[list[str], list[str]]
) -> None:
    """A labelled server serves a bare name unless the policy names it to
    keep it out (RMK-957): b, first in the chain, is kept out, so a takes the
    call, or nothing does when both are kept out; a policy that names no
    server leaves the call on the first server, b."""
    a, b, handler = _lookup_servers()

    await run_door(door, handler, channel={"tool_policy": policy})

    assert (a.ran, b.ran) == ran


async def test_a_reasoning_backend_agent_s_policy_keeps_a_server_out() -> None:
    """The policy of the agent a voice channel delegates to judges a bare name
    beside the channel's (RMK-957, RFC §12.4.1)."""
    a, b, handler = _lookup_servers()

    await run_door(
        "rt-agent-backend", handler, agent={"tool_policy": ToolPolicy(deny=["mcp__b__*"])}
    )

    assert (a.ran, b.ran) == (["lookup"], [])


async def test_an_unlabelled_provider_serves_an_alias_naming_any_server() -> None:
    a, b, handler = _two_servers(labelled=False)

    answer = await _on_session(None, handler, "mcp__a__delete_records")

    assert "delete_records done" in answer
    assert (a.ran, b.ran) == ([], ["delete_records"])


@pytest.mark.parametrize("gate_discovery", [True, False], ids=["gated", "forwarding"])
async def test_a_labelled_provider_serves_no_alias_of_another_server(gate_discovery: bool) -> None:
    server = _Server()
    handler = _mcp_handler(server, "crm", gate_discovery=gate_discovery)

    answer = await _on_session(None, handler, "mcp__erp__delete_records")

    assert answer == '{"error": "No handler for tool mcp__erp__delete_records"}'
    assert server.ran == []


@pytest.mark.parametrize("label", ["", "a__b", "crm_", "crm server", "crm\n", "x" * 65])
@pytest.mark.parametrize(
    "make",
    [
        lambda label: MCPToolProvider("http://crm.invalid/mcp", label=label),
        lambda label: MCPToolProvider.from_url("http://crm.invalid/mcp", label=label),
        lambda label: MCPToolProvider.from_command("crm-server", label=label),
    ],
    ids=["init", "from_url", "from_command"],
)
def test_a_label_that_is_not_an_identifier_is_refused(make: Any, label: str) -> None:
    with pytest.raises(ValueError, match="label"):
        make(label)


async def test_a_tool_a_skill_gates_never_runs_under_its_alias(tmp_path: Path) -> None:
    server = _Server()
    provider = MockRealtimeProvider()
    channel = RealtimeVoiceChannel(
        "rt",
        provider=provider,
        transport=MockRealtimeTransport(),
        tool_handler=_mcp_handler(server),
        skills=_registry_with_skill(tmp_path, allowed_tools="delete_*"),
    )
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    await kit.attach_channel("r1", "rt")
    session = await channel.start_session("r1", "u", "ws")

    await provider.simulate_tool_call(session, "c1", ALIAS, {})
    await until(lambda: bool(provider.tool_results))
    await kit.close()

    assert "gated by a skill" in provider.tool_results[0][2]
    assert server.ran == []


def _text_channel(server: _Server, **kwargs: Any) -> tuple[AIChannel, MockAIProvider]:
    """A text channel declaring the tool under its alias, as a host that names
    MCP tools that way does; its model calls the alias once."""
    provider = MockAIProvider(
        ai_responses=[
            AIResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[AIToolCall(id="c1", name=ALIAS, arguments={})],
            ),
            AIResponse(content="done"),
        ]
    )
    channel = AIChannel(
        "ai1",
        provider=provider,
        tools=[AITool(name=ALIAS, description="delete", parameters={})],
        tool_handler=_mcp_handler(server),
        tool_search=False,
        **kwargs,
    )
    return channel, provider


async def _text_answer(channel: AIChannel, provider: MockAIProvider) -> str:
    binding = ChannelBinding(channel_id="ai1", room_id="r1", channel_type=ChannelType.AI)
    await respond(
        channel, make_event(room_id="r1", body="go"), binding, RoomContext(room=Room(id="r1"))
    )
    [answer] = [
        str(part.result)
        for message in provider.calls[1].messages
        if message.role == "tool"
        for part in message.content
    ]
    return answer


async def test_a_text_turn_judges_an_alias_under_both_names() -> None:
    server = _Server()
    channel, provider = _text_channel(server, tool_policy=ToolPolicy(deny=["delete_*"]))

    answer = await _text_answer(channel, provider)

    assert "not permitted by the agent's tool policy" in answer
    assert server.ran == []


async def test_a_text_turn_gates_an_alias_as_its_tool(tmp_path: Path) -> None:
    server = _Server()
    skills = _registry_with_skill(tmp_path, allowed_tools="delete_*")
    channel, provider = _text_channel(server, skills=skills)

    answer = await _text_answer(channel, provider)

    assert "gated by a skill" in answer
    assert server.ran == []


async def test_an_external_handler_judges_an_alias_under_both_names() -> None:
    handler = PolicyExternalToolHandler(policy=ToolPolicy(deny=["delete_*"]))

    decision = await handler.process_tool_call(ALIAS, {})

    assert decision.approved is False
    assert decision.reason == policy_refusal(ALIAS)
