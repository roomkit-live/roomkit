"""A discussion's speak queue as pure state (RFC §19.7.5 rules 7, 8, 10, 11, 12)."""

from __future__ import annotations

from roomkit.orchestration.strategies.discussion._queue import Ask, SpeakQueueState

MAX_DEPTH = 5


def _agent_ask(event: str, depth: int, asker: str) -> Ask:
    return Ask(event_id=event, depth=depth, asker=asker)


def _person_ask(event: str, person: str = "ops") -> Ask:
    return Ask(event_id=event, depth=0, asker=person, person=True)


def _next(state: SpeakQueueState) -> str | None:
    pick = state.next_turn(MAX_DEPTH)
    return pick.entry.agent if pick.entry is not None else None


def _give(state: SpeakQueueState) -> str | None:
    pick = state.next_turn(MAX_DEPTH)
    if pick.entry is None:
        return None
    state.take(pick.entry)
    state.ended(pick.entry.agent)
    return pick.entry.agent


def test_an_agent_named_twice_is_owed_one_turn_answering_the_latest() -> None:
    state = SpeakQueueState()
    state.ask("sre", _agent_ask("e1", 1, "investigator"))
    state.ask("sre", _agent_ask("e2", 2, "dev"))

    entry = state.next_turn(MAX_DEPTH).entry
    assert entry is not None and entry.agent == "sre"
    assert entry.answered() == _agent_ask("e2", 2, "dev")
    assert entry.askers() == ["investigator", "dev"]
    assert state.view().queue == ("sre",)


def test_a_merged_turn_answers_the_persons_message_over_a_later_agents() -> None:
    state = SpeakQueueState()
    state.ask("sre", _person_ask("p1"), front=True)
    state.ask("sre", _agent_ask("e2", 3, "dev"))

    entry = state.next_turn(MAX_DEPTH).entry
    assert entry is not None and entry.answered() == _person_ask("p1")


def test_people_go_first_first_come_first_served_agents_at_the_back() -> None:
    state = SpeakQueueState()
    state.ask("dev", _agent_ask("e1", 1, "investigator"))
    state.ask("comms", _person_ask("p1"), front=True)
    state.ask("sre", _person_ask("p2"), front=True)
    state.ask("dev", _person_ask("p3"), front=True)

    assert state.view().queue == ("comms", "sre", "dev")


def test_the_agent_that_just_spoke_waits_while_another_can_take_the_turn() -> None:
    state = SpeakQueueState()
    state.ask("investigator", _person_ask("p1"), front=True)
    state.ask("sre", _agent_ask("e1", 1, "investigator"))
    assert _give(state) == "investigator"

    # A person's answer puts investigator first again; sre, owed a turn since
    # before, is served first (the POC's starvation run).
    state.ask("investigator", _person_ask("p2"), front=True)
    assert _next(state) == "sre"
    assert _give(state) == "sre"
    assert _give(state) == "investigator"


def test_an_agent_alone_in_the_queue_may_speak_twice_in_a_row() -> None:
    state = SpeakQueueState()
    state.ask("sre", _person_ask("p1"), front=True)
    assert _give(state) == "sre"
    state.ask("sre", _person_ask("p2"), front=True)
    assert _give(state) == "sre"


def test_an_agent_that_only_listens_answers_only_a_person_who_names_it() -> None:
    state = SpeakQueueState(listening=["sre"])
    state.ask("sre", _agent_ask("e1", 1, "dev"))
    assert _next(state) is None
    assert state.owes_a_person()

    state.ask("sre", _person_ask("p1"), front=True)
    assert _give(state) == "sre"
    assert state.listening == ["sre"]


def test_an_instruction_turn_is_given_even_to_an_agent_that_only_listens() -> None:
    state = SpeakQueueState(listening=["sre"])
    state.ask("dev", _agent_ask("e1", 1, "investigator"))
    state.queue_instruction("sre", "i1", depth=0)

    entry = state.next_turn(MAX_DEPTH).entry
    assert entry is not None and (entry.agent, entry.instruction) == ("sre", "i1")


def test_the_depth_limit_stops_a_turn_once_and_a_shallower_ask_resumes_it() -> None:
    state = SpeakQueueState()
    state.ask("sre", _agent_ask("e4", 4, "dev"))

    pick = state.next_turn(MAX_DEPTH)
    assert pick.entry is None and [e.agent for e in pick.stopped] == ["sre"]
    assert state.next_turn(MAX_DEPTH).stopped == []  # recorded once
    assert state.owes_a_person()

    state.ask("sre", _person_ask("p1"), front=True)
    entry = state.next_turn(MAX_DEPTH).entry
    assert entry is not None and entry.answered() == _person_ask("p1")


def test_asked_agents_are_found_and_cleared_per_person() -> None:
    state = SpeakQueueState()
    state.record_asked("dev", "ops")
    state.record_asked("sre", "alice")
    assert state.asking("ops") == ["dev"]
    assert state.asking(None) == ["dev", "sre"]

    state.clear_asked("ops")
    assert state.asked == [("sre", "alice")]
    state.clear_asked(None, agents=["sre"])
    assert state.asked == []


def test_dropping_the_queue_reports_the_instruction_turns_it_held() -> None:
    state = SpeakQueueState()
    state.ask("dev", _agent_ask("e1", 1, "sre"))
    state.queue_instruction("sre", "i1", depth=0)

    dropped = state.drop()
    assert [e.instruction for e in dropped] == ["i1"]
    assert state.view().queue == ()


def test_what_is_stored_leaves_out_who_speaks() -> None:
    state = SpeakQueueState()
    state.ask("sre", _person_ask("p1"), front=True)
    entry = state.next_turn(MAX_DEPTH).entry
    assert entry is not None
    state.take(entry)
    state.waiting = True

    stored = state.model_dump()
    assert "speaking" not in stored
    restored = SpeakQueueState.model_validate(stored)
    assert (restored.speaking, restored.turns_given, restored.waiting) == (None, 1, True)


def test_while_it_waits_for_a_person_only_a_turn_of_its_own_is_given() -> None:
    state = SpeakQueueState(waiting=True)
    state.ask("dev", _agent_ask("e1", 1, "sre"))
    assert _next(state) is None

    state.queue_regeneration("sre", _person_ask("p0"))
    entry = state.next_turn(MAX_DEPTH).entry
    assert entry is not None and (entry.agent, entry.regenerate) == ("sre", True)
    state.take(entry)
    assert state.waiting

    state.person_wrote()
    assert _next(state) == "dev"


def test_a_regenerated_answer_is_not_merged_into_a_pending_turn() -> None:
    state = SpeakQueueState()
    state.ask("sre", _agent_ask("e1", 1, "dev"))
    state.queue_regeneration("sre", _person_ask("p0"))
    state.ask("sre", _agent_ask("e2", 1, "dev"))

    assert [(e.regenerate, [a.event_id for a in e.asks]) for e in state.ordered()] == [
        (True, ["p0"]),
        (False, ["e1", "e2"]),
    ]


def test_a_restart_drops_the_instruction_turns_and_keeps_the_rest() -> None:
    state = SpeakQueueState()
    state.ask("dev", _agent_ask("e1", 1, "sre"))
    state.queue_instruction("sre", "i1", depth=0)

    dropped = state.drop_instructions()
    assert [e.instruction for e in dropped] == ["i1"]
    assert state.view().queue == ("dev",)
