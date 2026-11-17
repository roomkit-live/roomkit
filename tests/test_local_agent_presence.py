"""Local placement and automatic presence, without a remote executor (RFC §10.6.7)."""

from __future__ import annotations

from typing import Any

import pytest

from roomkit import AIChannel, RoomKit
from roomkit.core.exceptions import RoomNotFoundError
from roomkit.models import ChannelCategory, HookTrigger
from roomkit.providers.ai import MockAIProvider
from tests.test_agent_presence import _heard_all, _published
from tests.test_framework import SimpleChannel


def _agent(name: str) -> AIChannel:
    return AIChannel(name, provider=MockAIProvider(responses=["hi"]))


async def test_the_local_executor_is_excluded_from_the_public_executor_list() -> None:
    kit = RoomKit()
    try:
        assert await kit.executors() == []
        kit.register_channel(_agent("helper"))
        await kit.create_room(room_id="r1")
        await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        assert await kit.executors() == []
        await kit.turn_executor.close()
        assert await kit.executors() == []
    finally:
        await kit.close()


async def test_local_presence_is_sorted_and_filters_by_binding_category() -> None:
    kit = RoomKit()
    for name in ("zulu", "alpha", "not-intelligence"):
        kit.register_channel(_agent(name))
    kit.register_channel(SimpleChannel("ops"))
    try:
        await kit.create_room(room_id="r1")
        for name in ("zulu", "alpha"):
            await kit.attach_channel("r1", name, category=ChannelCategory.INTELLIGENCE)
        for name in ("ops", "not-intelligence"):
            await kit.attach_channel("r1", name, category=ChannelCategory.TRANSPORT)
        presence = await kit.agent_presence("r1")
        own = kit.turn_executor.executor_id
        assert [p.model_dump() for p in presence] == [
            {"channel_id": "alpha", "online": True, "executors": [own]},
            {"channel_id": "zulu", "online": True, "executors": [own]},
        ]
        presence[0].executors.append("caller-only")
        assert (await kit.agent_presence("r1"))[0].executors == [own]
    finally:
        await kit.close()


async def test_local_presence_reads_room_hooks_and_executor_closure() -> None:
    kit = RoomKit()
    kit.register_channel(_agent("helper"))
    try:
        for room in ("r1", "r2"):
            await kit.create_room(room_id=room)
            await kit.attach_channel(room, "helper", category=ChannelCategory.INTELLIGENCE)
        own = kit.turn_executor.executor_id
        await kit.enable_room_hook("r1", "audit")
        assert (await kit.agent_presence("r1"))[0].model_dump() == {
            "channel_id": "helper",
            "online": False,
            "executors": [],
        }
        assert (await kit.agent_presence("r2"))[0].executors == [own]

        @kit.turn_executor.room_hook("audit", HookTrigger.BEFORE_TOOL_USE)
        async def audit(_event: Any, _context: Any) -> None:
            pass

        assert (await kit.agent_presence("r1"))[0].executors == [own]
        await kit.turn_executor.close()
        for room in ("r1", "r2"):
            assert (await kit.agent_presence(room))[0].model_dump() == {
                "channel_id": "helper",
                "online": False,
                "executors": [],
            }
    finally:
        await kit.close()


@pytest.mark.parametrize("organization", [None, "tenant-a"])
async def test_local_presence_scopes_the_read_to_the_rooms_organization(
    organization: str | None,
) -> None:
    kit = RoomKit()
    kit.register_channel(_agent("helper"))
    try:
        await kit.create_room(room_id="r1", organization_id=organization)
        await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        own = kit.turn_executor.executor_id
        assert (await kit.agent_presence("r1", organization_id=organization))[0].executors == [own]
        with pytest.raises(RoomNotFoundError):
            await kit.agent_presence("r1", organization_id="other-tenant")
        with pytest.raises(RoomNotFoundError):
            await kit.agent_presence("missing", organization_id=organization)
        await kit.create_room(room_id="empty", organization_id=organization)
        assert await kit.agent_presence("empty", organization_id=organization) == []
    finally:
        await kit.close()


async def test_local_presence_publication_tracks_attach_hooks_detach_and_room_close() -> None:
    kit = RoomKit()
    kit.register_channel(_agent("helper"))
    try:
        await kit.create_room(room_id="r1")
        await kit.create_room(room_id="r2")
        heard = await _published(kit, "r1", "r2")
        await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        await kit.enable_room_hook("r1", "missing-hook")
        await kit.disable_room_hook("r1", "missing-hook")
        await kit.detach_channel("r1", "helper")
        await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        await kit.close_room("r1")
        await kit.reopen_room("r1")
        await kit.attach_channel("r1", "helper", category=ChannelCategory.INTELLIGENCE)
        await _heard_all(kit)
        assert heard == {
            "r1": [
                ("online", "helper"),
                ("offline", "helper"),
                ("online", "helper"),
                ("online", "helper"),
                ("online", "helper"),
            ],
            "r2": [],
        }
    finally:
        await kit.close()
