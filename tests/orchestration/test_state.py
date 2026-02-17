"""Tests for ConversationState models and helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from roomkit.models.room import Room
from roomkit.orchestration.state import (
    ConversationPhase,
    ConversationState,
    PhaseTransition,
    get_conversation_state,
    set_conversation_state,
)

# -- ConversationPhase --------------------------------------------------------


class TestConversationPhase:
    def test_built_in_phases_are_strings(self):
        assert ConversationPhase.INTAKE == "intake"
        assert ConversationPhase.QUALIFICATION == "qualification"
        assert ConversationPhase.HANDLING == "handling"
        assert ConversationPhase.ESCALATION == "escalation"
        assert ConversationPhase.RESOLUTION == "resolution"
        assert ConversationPhase.FOLLOWUP == "followup"

    def test_phase_usable_as_dict_key(self):
        d = {ConversationPhase.INTAKE: 1}
        assert d["intake"] == 1


# -- ConversationState defaults -----------------------------------------------


class TestConversationStateDefaults:
    def test_defaults(self):
        state = ConversationState()
        assert state.phase == ConversationPhase.INTAKE
        assert state.active_agent_id is None
        assert state.previous_agent_id is None
        assert state.handoff_count == 0
        assert state.phase_history == []
        assert state.context == {}

    def test_phase_started_at_is_recent(self):
        before = datetime.now(UTC)
        state = ConversationState()
        after = datetime.now(UTC)
        assert before <= state.phase_started_at <= after


# -- transition() immutability ------------------------------------------------


class TestTransition:
    def test_returns_new_instance(self):
        original = ConversationState()
        new = original.transition(to_phase="handling", to_agent="agent-a")
        assert new is not original
        assert original.phase == ConversationPhase.INTAKE
        assert original.active_agent_id is None

    def test_updates_phase_and_agent(self):
        state = ConversationState()
        new = state.transition(to_phase="handling", to_agent="agent-a")
        assert new.phase == "handling"
        assert new.active_agent_id == "agent-a"

    def test_records_previous_agent(self):
        state = ConversationState(active_agent_id="agent-a")
        new = state.transition(to_phase="review", to_agent="agent-b")
        assert new.previous_agent_id == "agent-a"
        assert new.active_agent_id == "agent-b"

    def test_handoff_count_increments_on_agent_change(self):
        state = ConversationState(active_agent_id="agent-a")
        new = state.transition(to_phase="review", to_agent="agent-b")
        assert new.handoff_count == 1

    def test_handoff_count_stable_on_same_agent(self):
        state = ConversationState(active_agent_id="agent-a")
        new = state.transition(to_phase="review", to_agent="agent-a")
        assert new.handoff_count == 0

    def test_handoff_count_increments_from_none(self):
        state = ConversationState()  # active_agent_id=None
        new = state.transition(to_phase="handling", to_agent="agent-a")
        assert new.handoff_count == 1

    def test_handoff_count_increments_to_none(self):
        state = ConversationState(active_agent_id="agent-a")
        new = state.transition(to_phase="escalation", to_agent=None)
        assert new.handoff_count == 1

    def test_phase_history_appended(self):
        state = ConversationState()
        s1 = state.transition(to_phase="handling", to_agent="agent-a", reason="qualified")
        s2 = s1.transition(to_phase="review", to_agent="agent-b", reason="code complete")

        assert len(s2.phase_history) == 2

        t0 = s2.phase_history[0]
        assert t0.from_phase == "intake"
        assert t0.to_phase == "handling"
        assert t0.from_agent is None
        assert t0.to_agent == "agent-a"
        assert t0.reason == "qualified"

        t1 = s2.phase_history[1]
        assert t1.from_phase == "handling"
        assert t1.to_phase == "review"
        assert t1.from_agent == "agent-a"
        assert t1.to_agent == "agent-b"
        assert t1.reason == "code complete"

    def test_transition_metadata(self):
        state = ConversationState()
        new = state.transition(
            to_phase="handling",
            to_agent="agent-a",
            metadata={"summary": "user wants help"},
        )
        assert new.phase_history[0].metadata == {"summary": "user wants help"}

    def test_phase_started_at_updated(self):
        state = ConversationState()
        original_time = state.phase_started_at
        new = state.transition(to_phase="handling")
        assert new.phase_started_at >= original_time

    def test_custom_phase_strings(self):
        state = ConversationState()
        new = state.transition(to_phase="my-custom-phase", to_agent="agent-x")
        assert new.phase == "my-custom-phase"
        assert new.phase_history[0].to_phase == "my-custom-phase"

    def test_context_preserved_through_transition(self):
        state = ConversationState(context={"key": "value"})
        new = state.transition(to_phase="handling")
        assert new.context == {"key": "value"}


# -- Serialization round-trip -------------------------------------------------


class TestSerialization:
    def test_model_dump_and_validate(self):
        state = ConversationState(active_agent_id="agent-a")
        s1 = state.transition(to_phase="handling", to_agent="agent-b", reason="test")

        data = s1.model_dump(mode="json")
        restored = ConversationState.model_validate(data)

        assert restored.phase == s1.phase
        assert restored.active_agent_id == s1.active_agent_id
        assert restored.previous_agent_id == s1.previous_agent_id
        assert restored.handoff_count == s1.handoff_count
        assert restored.phase_started_at == s1.phase_started_at
        assert len(restored.phase_history) == len(s1.phase_history)
        assert restored.phase_history[0].reason == "test"
        assert restored.context == s1.context

    def test_datetime_survives_json_roundtrip(self):
        state = ConversationState()
        data = state.model_dump(mode="json")
        restored = ConversationState.model_validate(data)
        assert restored.phase_started_at == state.phase_started_at

    def test_transition_timestamp_survives_roundtrip(self):
        state = ConversationState()
        s1 = state.transition(to_phase="handling")
        data = s1.model_dump(mode="json")
        restored = ConversationState.model_validate(data)
        assert restored.phase_history[0].timestamp == s1.phase_history[0].timestamp

    def test_context_dict_survives_roundtrip(self):
        state = ConversationState(
            context={
                "handoff_summary": "user needs portfolio help",
                "handoff_from": "agent-triage",
                "nested": {"a": [1, 2, 3]},
            }
        )
        data = state.model_dump(mode="json")
        restored = ConversationState.model_validate(data)
        assert restored.context == state.context


# -- Room.metadata round-trip -------------------------------------------------


def _make_room(**kwargs: object) -> Room:
    return Room(id="room-1", **kwargs)  # type: ignore[arg-type]


class TestRoomMetadataRoundTrip:
    def test_set_and_get(self):
        room = _make_room()
        state = ConversationState(active_agent_id="agent-a")
        s1 = state.transition(to_phase="handling", to_agent="agent-b", reason="qualified")

        updated_room = set_conversation_state(room, s1)
        restored = get_conversation_state(updated_room)

        assert restored.phase == "handling"
        assert restored.active_agent_id == "agent-b"
        assert restored.previous_agent_id == "agent-a"
        assert restored.handoff_count == 1
        assert len(restored.phase_history) == 1
        assert restored.phase_history[0].reason == "qualified"

    def test_get_from_empty_room(self):
        room = _make_room()
        state = get_conversation_state(room)
        assert state.phase == ConversationPhase.INTAKE
        assert state.active_agent_id is None
        assert state.handoff_count == 0

    def test_preserves_existing_metadata(self):
        room = _make_room(metadata={"custom_key": "custom_value"})
        state = ConversationState()
        updated = set_conversation_state(room, state)
        assert updated.metadata["custom_key"] == "custom_value"
        assert "_conversation_state" in updated.metadata

    def test_original_room_unchanged(self):
        room = _make_room()
        state = ConversationState()
        set_conversation_state(room, state)
        assert "_conversation_state" not in room.metadata

    def test_multiple_transitions_roundtrip(self):
        room = _make_room()

        s0 = ConversationState()
        s1 = s0.transition(to_phase="handling", to_agent="agent-a")
        s2 = s1.transition(to_phase="review", to_agent="agent-b")
        s3 = s2.transition(to_phase="coding", to_agent="agent-a", reason="fixes needed")

        room = set_conversation_state(room, s3)
        restored = get_conversation_state(room)

        assert restored.phase == "coding"
        assert restored.active_agent_id == "agent-a"
        assert restored.previous_agent_id == "agent-b"
        assert restored.handoff_count == 3
        assert len(restored.phase_history) == 3


# -- PhaseTransition model ---------------------------------------------------


class TestPhaseTransition:
    def test_defaults(self):
        t = PhaseTransition(from_phase="a", to_phase="b")
        assert t.from_agent is None
        assert t.to_agent is None
        assert t.reason == ""
        assert t.metadata == {}
        assert isinstance(t.timestamp, datetime)

    def test_timestamp_is_utc(self):
        t = PhaseTransition(from_phase="a", to_phase="b")
        assert t.timestamp.tzinfo is not None
