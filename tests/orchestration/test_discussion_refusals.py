"""What a room a discussion holds cannot share it with (RFC §19.7.5 rule 1).

One rule decides who speaks, never two: the install refuses a room that holds
something else that would, and while the discussion is installed, binding it
or installing it is refused.
"""

from __future__ import annotations

import pytest

from roomkit.channels.ai import AIChannel
from roomkit.core.framework import RoomKit
from roomkit.core.hooks import HookRegistration
from roomkit.models.enums import ChannelCategory, ChannelType, HookExecution, HookTrigger
from roomkit.orchestration.router import ConversationRouter
from roomkit.orchestration.strategies import Discussion, Swarm
from roomkit.providers.ai.base import AIResponse
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.speaking.mock import MockSpeakPolicy, MockThinker
from tests.test_framework import SimpleChannel


def _agent(channel_id: str, **kwargs: object) -> AIChannel:
    provider = MockAIProvider(ai_responses=[AIResponse(content="ok")])
    return AIChannel(channel_id, provider=provider, **kwargs)  # type: ignore[arg-type]


def _router_hook(name: str = "router") -> HookRegistration:
    return HookRegistration(
        trigger=HookTrigger.BEFORE_BROADCAST,
        execution=HookExecution.SYNC,
        fn=ConversationRouter(default_agent_id="a").as_hook(),
        name=name,
    )


async def _discussion_room(kit: RoomKit) -> None:
    await kit.create_room(room_id="r1", orchestration=Discussion([_agent("a"), _agent("b")]))


# -- At install --


async def test_an_agent_that_thinks_while_it_listens_is_refused() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1")

    strategy = Discussion([_agent("a", thinker=MockThinker(), speak_policy=MockSpeakPolicy())])
    with pytest.raises(ValueError, match="thinks while it listens"):
        await strategy.install(kit, "r1")
    await kit.close()


async def test_a_room_with_a_router_is_refused() -> None:
    kit = RoomKit()
    await kit.create_room(room_id="r1")
    kit.hook_engine.add_room_hook("r1", _router_hook())

    with pytest.raises(ValueError, match="a router is installed"):
        await Discussion([_agent("a")]).install(kit, "r1")
    await kit.close()


async def test_an_agent_another_strategy_runs_is_refused() -> None:
    kit = RoomKit()
    a = _agent("a")
    await kit.create_room(room_id="r1")
    a._registry.set_turn_runner("r1", lambda *args: None, owner=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="another strategy runs"):
        await Discussion([a]).install(kit, "r1")
    await kit.close()


# -- Once installed --


async def test_binding_another_intelligence_channel_is_refused() -> None:
    kit = RoomKit()
    await _discussion_room(kit)
    kit.register_channel(_agent("other"))

    with pytest.raises(ValueError, match="not one of its agents"):
        await kit.attach_channel("r1", "other", category=ChannelCategory.INTELLIGENCE)
    assert await kit.store.get_binding("r1", "other") is None
    await kit.close()


async def test_binding_a_voice_channel_is_refused_and_a_transport_is_not() -> None:
    kit = RoomKit()
    await _discussion_room(kit)
    kit.register_channel(SimpleChannel("phone", channel_type=ChannelType.VOICE))
    kit.register_channel(SimpleChannel("sms"))

    with pytest.raises(ValueError, match="voice or realtime"):
        await kit.attach_channel("r1", "phone")
    binding = await kit.attach_channel("r1", "sms")
    assert binding.category == ChannelCategory.TRANSPORT
    await kit.close()


async def test_installing_a_router_is_refused_in_the_room_and_globally() -> None:
    kit = RoomKit()
    await _discussion_room(kit)

    with pytest.raises(ValueError, match="holds a discussion"):
        kit.hook_engine.add_room_hook("r1", _router_hook())
    with pytest.raises(ValueError, match="holds a discussion"):
        kit.hook_engine.register(_router_hook("global_router"))
    # Another room, and any hook that is no router, stay free.
    kit.hook_engine.add_room_hook("r2", _router_hook())
    await kit.close()


async def test_installing_another_strategy_is_refused() -> None:
    kit = RoomKit()
    await _discussion_room(kit)

    with pytest.raises(ValueError, match="holds a (strategy|discussion)"):
        await Swarm(agents=[]).install(kit, "r1")  # type: ignore[arg-type]
    assert not kit.hook_engine.has_router_hook("r1")
    await kit.close()


async def test_once_uninstalled_the_room_takes_a_router_again() -> None:
    kit = RoomKit()
    strategy = Discussion([_agent("a")])
    await kit.create_room(room_id="r1", orchestration=strategy)

    await strategy.uninstall(kit, "r1")
    kit.hook_engine.add_room_hook("r1", _router_hook())
    assert kit.hook_engine.has_router_hook("r1")
    await kit.close()
