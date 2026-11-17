"""Read and edit host definitions through the public extra_tools accessor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import AIChannel, RoomKit
from roomkit.providers.ai import AITool

if TYPE_CHECKING:
    from tests.conformance._harness import Harness


async def read(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    channel = kit.get_channel(spec["channel_id"])
    assert isinstance(channel, AIChannel)
    return [tool.model_dump(mode="json") for tool in channel.extra_tools]


async def replace(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    channel = kit.get_channel(spec["channel_id"])
    assert isinstance(channel, AIChannel)
    definition = AITool.model_validate(spec["definition"])
    selected = channel.extra_tools[spec["index"]]
    for name in AITool.model_fields:
        setattr(selected, name, getattr(definition, name))
