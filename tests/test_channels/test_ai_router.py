"""An AIChannel whose router picks each turn's provider (RFC §6.4; RMK-1638).

The shared scenarios under ``ai-router/`` pin what a room observes: which
model serves each turn and round, what the router is asked, the context built
for the model it picks. These tests pin the channel's own surface: what it is
built with, what it names, what it closes, and the turns no room drives.
"""

from __future__ import annotations

from typing import Any

import pytest

from roomkit.channels import AIRouter, FallbackRouter, RouteRequest
from roomkit.channels.agent import Agent
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIContext, AIMessage, AIProvider, AIResponse, AITool
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.tools.context import _current_loop_ctx, _ToolLoopContext
from tests.conftest import make_event
from tests.tool_loop_modes import respond, run_tool_loop

CATALOGUE = [
    AITool(
        name=f"crm_action_{i}",
        description=f"Perform CRM operation number {i} on a customer record, "
        "with filters, pagination and audit metadata.",
        parameters={"type": "object", "properties": {"id": {"type": "string"}}},
    )
    for i in range(30)
]


class _Picks(AIRouter):
    """Names *provider* for every request, and keeps the requests."""

    def __init__(self, provider: AIProvider | None) -> None:
        self.provider = provider
        self.requests: list[RouteRequest] = []
        self.closed = False

    async def route(self, request: RouteRequest) -> AIProvider | None:
        self.requests.append(request)
        return self.provider

    async def close(self) -> None:
        self.closed = True


class _Window(MockAIProvider):
    """A model whose context window is *window* tokens."""

    def __init__(self, window: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._window = window

    @property
    def context_window(self) -> int | None:
        return self._window


class _Closing(MockAIProvider):
    closed = False

    async def close(self) -> None:
        self.closed = True


async def _served(name: str, arguments: dict[str, Any]) -> str:
    return "ok"


async def _answer(channel: AIChannel, body: str = "go") -> None:
    binding = ChannelBinding(
        channel_id=channel.channel_id,
        room_id="r1",
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    await respond(
        channel,
        make_event(room_id="r1", body=body, channel_id="sms1"),
        binding,
        RoomContext(room=Room(id="r1")),
    )


@pytest.mark.parametrize("serves", ["both", "neither"])
def test_a_channel_takes_a_provider_or_a_router_one_of_the_two(serves: str) -> None:
    kwargs: dict[str, Any] = {}
    if serves == "both":
        kwargs = {"provider": MockAIProvider(), "router": _Picks(MockAIProvider())}
    with pytest.raises(ValueError, match="a provider or a router"):
        AIChannel("ai1", **kwargs)


def test_a_channel_with_a_router_names_the_router_not_a_provider() -> None:
    router = _Picks(MockAIProvider())
    channel = AIChannel("ai1", router=router)

    assert channel.provider is None
    assert channel.router is router
    assert channel.provider_name == "_Picks"
    assert channel.info["provider"] == "_Picks"


def test_an_agent_with_a_router_is_not_config_only() -> None:
    agent = Agent("a1", router=_Picks(MockAIProvider()))

    assert not agent.is_config_only
    assert Agent("a2").is_config_only


async def test_a_router_that_names_no_provider_for_a_turn_fails_it() -> None:
    channel = AIChannel("ai1", router=_Picks(None))

    with pytest.raises(ValueError, match="named no provider for a turn"):
        await _answer(channel)


@pytest.mark.parametrize(("window", "searched"), [(4_000, True), (1_000_000, False)])
async def test_the_window_of_the_model_the_router_picks_decides_tool_search(
    window: int, searched: bool
) -> None:
    # Thirty schemas pass 10 % of a 4,000-token window, never of a million's.
    model = _Window(window, ai_responses=[AIResponse(content="ok")])
    channel = AIChannel("ai1", router=_Picks(model), tools=CATALOGUE, tool_handler=_served)

    await _answer(channel)

    declared = {tool.name for tool in model.calls[0].tools or []}
    assert ("find_tools" in declared) == searched
    assert ("crm_action_0" in declared) != searched


async def test_a_turn_no_room_drives_is_routed_on_its_last_user_message() -> None:
    # A voice delegation runs the loop on a context it built itself.
    model = MockAIProvider(ai_responses=[AIResponse(content="ok")])
    router = _Picks(model)
    channel = AIChannel("ai1", router=router)
    context = AIContext(
        messages=[
            AIMessage(role="user", content="earlier"),
            AIMessage(role="assistant", content="noted"),
            AIMessage(role="user", content="what now?"),
        ]
    )

    run = await run_tool_loop(channel, context)

    assert run.reason == "completed"
    [request] = router.requests
    assert (request.reason, request.text, request.event) == ("turn", "what now?", None)
    assert len(model.calls) == 1


async def test_a_loop_run_inside_another_agent_s_turn_runs_on_its_own_provider() -> None:
    # A handler of one agent's turn drives another agent's loop: the loop
    # inherits the enclosing turn's context, never the provider it routed.
    theirs = MockAIProvider(ai_responses=[AIResponse(content="theirs")])
    own = MockAIProvider(ai_responses=[AIResponse(content="own")])
    enclosing = _ToolLoopContext(provider=theirs, provider_of="other-agent")
    token = _current_loop_ctx.set(enclosing)
    try:
        run = await run_tool_loop(AIChannel("ai1", provider=own), AIContext(messages=[]))
    finally:
        _current_loop_ctx.reset(token)

    assert run.text == "own"
    assert (len(own.calls), len(theirs.calls)) == (1, 0)


async def test_the_channel_closes_its_router_and_a_fallback_router_both_providers() -> None:
    primary, fallback = _Closing(), _Closing()
    router = _Picks(MockAIProvider())

    await AIChannel("ai1", router=router).close()
    await AIChannel("ai2", router=FallbackRouter(primary, fallback)).close()

    assert router.closed
    assert primary.closed and fallback.closed
