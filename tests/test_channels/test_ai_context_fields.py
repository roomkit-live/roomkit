"""Provider room snapshots and target metadata survive real tool rounds."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from roomkit import HookResult, HookTrigger, RoomKit
from roomkit.channels.ai import AIChannel
from roomkit.models.channel import ChannelBinding, ChannelCapabilities
from roomkit.models.context import RoomContext
from roomkit.models.enums import Access, ChannelCategory, ChannelDirection, ChannelMediaType
from roomkit.models.participant import Participant
from roomkit.models.room import Room
from roomkit.providers.ai.mock import MockAIProvider
from tests.conftest import make_event
from tests.test_tool_context import _binding, _tool_round_responses
from tests.tool_loop_modes import respond


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("replace_context", [False, True])
async def test_full_room_and_target_fields_reach_every_round_after_the_hook(
    streaming: bool, replace_context: bool
) -> None:
    provider = MockAIProvider(ai_responses=_tool_round_responses(), streaming=streaming)

    async def handler(name: str, arguments: dict[str, Any]) -> str:
        return "found"

    channel = AIChannel("ai1", provider=provider, tool_handler=handler)
    kit = RoomKit()
    kit.register_channel(channel)
    await kit.create_room(room_id="r1")
    own = _binding("r1")
    restrictive = ChannelCapabilities(
        media_types=[ChannelMediaType.TEXT, ChannelMediaType.RICH],
        max_length=91,
        supports_buttons=True,
        custom={"target": "snapshot"},
    )
    target = ChannelBinding(
        channel_id="out",
        room_id="r1",
        channel_type="websocket",
        access=Access.NONE,
        muted=True,
        output_muted=True,
        visibility="internal",
        capabilities=restrictive,
    )
    context = RoomContext(
        room=Room(id="r1", metadata={"topic": "source"}),
        bindings=[
            # Even a transport binding for this channel is not its own target.
            own.model_copy(update={"category": ChannelCategory.TRANSPORT}),
            target,
            target.model_copy(
                update={
                    "channel_id": "in",
                    "direction": ChannelDirection.INBOUND,
                    "capabilities": ChannelCapabilities(max_length=1),
                }
            ),
            target.model_copy(
                update={
                    "channel_id": "other-ai",
                    "category": ChannelCategory.INTELLIGENCE,
                    "capabilities": ChannelCapabilities(max_length=2),
                }
            ),
        ],
        participants=[
            Participant(
                id="alice",
                room_id="r1",
                channel_id="in",
                display_name="Alice",
                metadata={"team": "support"},
            )
        ],
        recent_events=[
            make_event(room_id="r1", channel_id="in", body="earlier").model_copy(
                update={"index": 7}
            )
        ],
        history_start_index=7,
        author_register={"version": 3, "names": {}},
    )
    original = context.model_dump()
    replacement = context.model_copy(
        update={"room": context.room.model_copy(update={"metadata": {"topic": "hook"}})}
    )
    changed_caps = ChannelCapabilities(media_types=[ChannelMediaType.LOCATION], max_length=17)
    seen: list[dict[str, Any]] = []

    @kit.hook(HookTrigger.BEFORE_AI_GENERATION)
    async def before(event: Any, _: Any) -> HookResult:
        ai = event.ai_context
        assert ai.room is not None
        seen.append(ai.room.model_dump())
        update: dict[str, Any] = {"system_prompt": "Hook kept."}
        if replace_context:
            update.update(
                room=replacement,
                target_capabilities=changed_caps,
                target_media_types=[ChannelMediaType.AUDIO],
            )
        return HookResult.modify(replace(event, ai_context=ai.model_copy(update=update)))

    try:
        await respond(
            channel, make_event(room_id="r1", channel_id="in", body="read"), own, context
        )
        assert len(provider.calls) == 2
        assert len(seen) == 1
        assert seen[0] == original
        expected_room = replacement.model_dump() if replace_context else original
        for request in provider.calls:
            assert request.room is not None
            assert request.room.model_dump() == expected_room
            expected_caps = changed_caps if replace_context else restrictive
            assert request.target_capabilities is not None
            assert request.target_capabilities.model_dump(exclude={"media_types"}) == (
                expected_caps.model_dump(exclude={"media_types"})
            )
            assert set(request.target_capabilities.media_types) == set(expected_caps.media_types)
            assert set(request.target_media_types) == (
                {ChannelMediaType.AUDIO} if replace_context else set(restrictive.media_types)
            )
            assert request.system_prompt == "Hook kept."
        assert context.model_dump() == original
    finally:
        await kit.close()
