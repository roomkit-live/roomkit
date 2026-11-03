"""A turn's skills are the registry its configuration gave it (RFC §24.3).

One channel serves an agent in every room, built once; the skills a turn may
activate can belong to the person it is for. The configuration of each turn
gives its registry: the manifest, the activation, reference and script tools
and the skill gating read it in place of the channel's, which is never
replaced and never read by another turn.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from roomkit import AIChannel, AIChannelTurnConfig
from roomkit.models.channel import ChannelBinding
from roomkit.models.context import RoomContext
from roomkit.models.enums import ChannelCategory, ChannelType
from roomkit.models.room import Room
from roomkit.providers.ai.base import AIResponse, AIToolCall
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.skills.registry import SkillRegistry
from tests.conftest import make_event
from tests.tool_loop_modes import LoopRun, respond


def _registry(root: Path, *names: str) -> SkillRegistry:
    for name in names:
        skill_dir = root / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: The {name} procedure\n---\nRules for {name}.",
            encoding="utf-8",
        )
    registry = SkillRegistry()
    registry.discover(root)
    return registry


def _activate(name: str) -> AIResponse:
    return AIResponse(
        content="",
        tool_calls=[
            AIToolCall(id=f"call-{name}", name="activate_skill", arguments={"name": name})
        ],
        finish_reason="tool_calls",
    )


async def _turn(channel: AIChannel, room_id: str) -> LoopRun:
    binding = ChannelBinding(
        room_id=room_id,
        channel_id=channel.channel_id,
        channel_type=ChannelType.AI,
        category=ChannelCategory.INTELLIGENCE,
    )
    context = RoomContext(room=Room(id=room_id), bindings=[binding])
    return await respond(channel, make_event(room_id=room_id), binding, context)


def _prompt(provider: MockAIProvider, index: int = 0) -> str:
    return provider.calls[index].system_prompt or ""


async def test_a_turn_lists_and_activates_its_own_skills(streaming: bool, tmp_path: Path) -> None:
    shared = _registry(tmp_path / "agent", "shared")
    ada = _registry(tmp_path / "ada", "shared", "ada-procedure")

    async def resolve(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(skills=ada)

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[_activate("ada-procedure"), AIResponse(content="done")],
    )
    channel = AIChannel("agent", provider=provider, skills=shared, config_provider=resolve)
    try:
        run = await _turn(channel, "room-ada")
    finally:
        await channel.close()

    assert "ada-procedure" in _prompt(provider)
    (call,) = run.calls
    assert "Rules for ada-procedure." in str(call.result)


async def test_concurrent_turns_each_read_their_own_registry(
    streaming: bool, tmp_path: Path
) -> None:
    """Two people talk to the same agent at once: neither reaches the other's."""
    registries = {
        "room-ada": _registry(tmp_path / "ada", "ada-procedure"),
        "room-bob": _registry(tmp_path / "bob", "bob-procedure"),
    }
    entered = asyncio.Event()
    release = asyncio.Event()

    async def resolve(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        if binding.room_id == "room-ada":
            entered.set()
            await asyncio.wait_for(release.wait(), 1)
        else:
            release.set()
        return AIChannelTurnConfig(skills=registries[binding.room_id])

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            _activate("bob-procedure"),
            _activate("bob-procedure"),
            AIResponse(content="done"),
            AIResponse(content="done"),
        ],
    )
    channel = AIChannel(
        "agent", provider=provider, skills=SkillRegistry(), config_provider=resolve
    )
    try:
        async with asyncio.TaskGroup() as group:
            ada = group.create_task(_turn(channel, "room-ada"))
            await asyncio.wait_for(entered.wait(), 1)
            bob = group.create_task(_turn(channel, "room-bob"))
    finally:
        await channel.close()

    (bob_call,) = bob.result().calls
    (ada_call,) = ada.result().calls
    assert "Rules for bob-procedure." in str(bob_call.result)
    assert "Rules for bob-procedure." not in str(ada_call.result)


@pytest.mark.parametrize("config", [None, AIChannelTurnConfig()], ids=["no config", "no skills"])
async def test_a_turn_that_gives_none_reads_the_channels(
    streaming: bool, tmp_path: Path, config: AIChannelTurnConfig | None
) -> None:
    shared = _registry(tmp_path / "agent", "shared")

    async def resolve(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig | None:
        return config

    provider = MockAIProvider(
        streaming=streaming, ai_responses=[_activate("shared"), AIResponse(content="done")]
    )
    channel = AIChannel("agent", provider=provider, skills=shared, config_provider=resolve)
    try:
        run = await _turn(channel, "room")
    finally:
        await channel.close()

    (call,) = run.calls
    assert "Rules for shared." in str(call.result)


async def test_a_turns_registry_is_never_left_on_the_channel(
    streaming: bool, tmp_path: Path
) -> None:
    """The next turn that gives none reads the channel's, not the last turn's."""
    shared = _registry(tmp_path / "agent", "shared")
    ada = _registry(tmp_path / "ada", "ada-procedure")
    configs = iter([AIChannelTurnConfig(skills=ada), None])

    async def resolve(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig | None:
        return next(configs)

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            AIResponse(content="first"),
            _activate("ada-procedure"),
            AIResponse(content="done"),
        ],
    )
    channel = AIChannel("agent", provider=provider, skills=shared, config_provider=resolve)
    try:
        await _turn(channel, "room-ada")
        second = await _turn(channel, "room-bob")
    finally:
        await channel.close()

    assert "ada-procedure" not in _prompt(provider, 1)
    (call,) = second.calls
    assert "Rules for ada-procedure." not in str(call.result)


async def test_a_channel_built_without_skills_ignores_a_turns_registry(
    streaming: bool, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ada = _registry(tmp_path / "ada", "ada-procedure")

    async def resolve(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(skills=ada)

    provider = MockAIProvider(streaming=streaming, ai_responses=[AIResponse(content="done")])
    channel = AIChannel("agent", provider=provider, config_provider=resolve)
    try:
        with caplog.at_level(logging.WARNING, logger="roomkit.channels.ai"):
            await _turn(channel, "room")
    finally:
        await channel.close()

    declared: list[Any] = provider.calls[0].tools or []
    assert "activate_skill" not in {tool.name for tool in declared}
    assert "ada-procedure" not in _prompt(provider)
    assert any("built without skills" in record.getMessage() for record in caplog.records)


async def test_a_skill_the_next_turns_registry_lacks_leaves_the_room(
    streaming: bool, tmp_path: Path
) -> None:
    """In one room, Ada's turn activates her skill and Bob's turn follows: her
    rules cannot be in front of his model, so the skill leaves the room's
    active ones, and her next activation delivers its body again (RFC §24.4)."""
    registries = iter(
        [
            _registry(tmp_path / "ada-1", "ada-procedure"),
            _registry(tmp_path / "bob", "bob-procedure"),
            _registry(tmp_path / "ada-2", "ada-procedure"),
        ]
    )

    async def resolve(binding: ChannelBinding, context: RoomContext) -> AIChannelTurnConfig:
        return AIChannelTurnConfig(skills=next(registries))

    provider = MockAIProvider(
        streaming=streaming,
        ai_responses=[
            _activate("ada-procedure"),
            AIResponse(content="done"),
            AIResponse(content="bob's answer"),
            _activate("ada-procedure"),
            AIResponse(content="done"),
        ],
    )
    channel = AIChannel(
        "agent", provider=provider, skills=SkillRegistry(), config_provider=resolve
    )
    try:
        await _turn(channel, "team")
        await _turn(channel, "team")
        again = await _turn(channel, "team")
    finally:
        await channel.close()

    bobs_prompt = _prompt(provider, 2)
    assert "Rules for ada-procedure." not in bobs_prompt
    (call,) = again.calls
    assert "Rules for ada-procedure." in str(call.result)
