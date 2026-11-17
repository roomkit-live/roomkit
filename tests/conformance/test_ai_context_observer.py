"""The fixture snapshots context values and normalizes room-bound indexes."""

from __future__ import annotations

import pytest

from roomkit.models.context import RoomContext
from roomkit.models.room import Room
from roomkit.providers.ai import AIContext
from tests.conformance._agent import ScriptedModel
from tests.conformance._normalize import Normalizer, Observed
from tests.conformance._script import Gates


async def test_model_observer_keeps_absence_and_copies_the_room_at_request_time() -> None:
    model = ScriptedModel([{"text": "first"}, {"text": "next"}], Gates(), streaming=False)
    await model.generate(AIContext())
    assert {
        key: model.requests[0][key]
        for key in ("room", "target_capabilities", "target_media_types")
    } == {"room": None, "target_capabilities": None, "target_media_types": []}
    context = RoomContext(room=Room(id="r", metadata={"at": "request"}))
    await model.generate(AIContext(room=context))
    context.room.metadata["at"] = "later"
    assert model.requests[1]["room"]["room"]["metadata"] == {"at": "request"}


@pytest.mark.parametrize(("boundary", "expected"), [(None, None), (0, 0), (3, 1)])
def test_context_history_boundary_uses_its_room_lifecycle_indexes(
    boundary: int | None, expected: int | None
) -> None:
    lifecycle = [
        {
            "id": f"life-{index}",
            "room_id": "r",
            "index": index,
            "source": {"channel_id": "system"},
            "visibility": "internal",
            "content": {"type": "system"},
        }
        for index in (0, 1)
    ]
    normalizer = Normalizer(Observed(literal_rooms={"r", "other"}, timelines={"r": lifecycle}))
    snapshot = {"room": {"id": "r"}, "history_start_index": boundary}
    assert normalizer.value(snapshot)["history_start_index"] == expected
    other = {"room": {"id": "other"}, "history_start_index": boundary}
    assert normalizer.value(other)["history_start_index"] == boundary
