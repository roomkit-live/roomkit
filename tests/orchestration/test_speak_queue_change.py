"""The public queue-change value and carrier, independent of executor presence."""

from __future__ import annotations

import json

from roomkit import SpeakQueue, SpeakQueueChange, SpeakQueueEvent
from roomkit.orchestration import SpeakQueueChange as OrchestrationChange
from roomkit.orchestration.strategies.discussion import SpeakQueueChange as DiscussionChange


def test_turn_skipped_is_a_public_queue_change_and_json_string() -> None:
    change = SpeakQueueChange.TURN_SKIPPED
    assert OrchestrationChange.TURN_SKIPPED is change
    assert DiscussionChange.TURN_SKIPPED is change
    assert change.name == "TURN_SKIPPED"
    assert change.value == str(change) == "turn_skipped"
    assert json.dumps(change) == '"turn_skipped"'
    assert SpeakQueueChange(json.loads(json.dumps(change))) is change


def test_a_queue_event_keeps_turn_skipped_and_its_default_fields() -> None:
    queue = SpeakQueue(None, (), frozenset(), (), waiting=False, over=False)
    event = SpeakQueueEvent("r1", queue, SpeakQueueChange.TURN_SKIPPED)
    assert event.room_id == "r1" and event.queue is queue
    assert event.change is SpeakQueueChange.TURN_SKIPPED
    assert event.channel_ids == () and event.event_id is None
    assert json.loads(json.dumps({"change": event.change})) == {"change": "turn_skipped"}
