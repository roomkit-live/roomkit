"""A scenario's top-level members: the checker's citations are accepted, not run."""

from __future__ import annotations

import pytest

from tests.conformance import test_conformance as runner
from tests.conformance._shape import unknown_members


def test_a_missing_feature_skips_before_reading_the_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "PROVIDED_FEATURES", {"message_session"})
    scenario = {"requires_features": ["message_session", "voice"], "setup": {"channels": [{}]}}
    marks = runner._marks("media/example", scenario)
    assert len(marks) == 1
    assert marks[0].name == "skip"
    assert marks[0].kwargs["reason"] == "missing scenario features: voice"


@pytest.mark.parametrize(
    ("member", "problems"),
    [("requirements", []), ("requirement", ["$: requirement"])],
)
def test_requirement_citations_pass_and_unknown_members_do_not(
    member: str, problems: list[str]
) -> None:
    scenario = {"id": "x", "rules": ["§10.4"], member: ["RK-10.4-003"]}
    assert unknown_members(scenario) == problems


@pytest.mark.parametrize(
    ("spawn", "problems"),
    [
        ({"gate": "go", "delay_ms": 5, "do": [], "after_do_gate": "done"}, []),
        ({"fail": "boom"}, ["setup.hooks[0].reactions[0].spawn: fail"]),
        (
            {"do": [{"call": "send_event", "room_id": "$room", "typo": 1}]},
            ["setup.hooks[0].reactions[0].spawn.do[0]: typo"],
        ),
    ],
    ids=["its-members", "a-reaction-s-failure", "an-action-s-typo"],
)
def test_a_spawned_task_takes_a_gate_a_delay_actions_and_an_after_gate(
    spawn: dict[str, object], problems: list[str]
) -> None:
    hook = {"name": "h", "trigger": "before_broadcast", "reactions": [{"spawn": spawn}]}
    assert unknown_members({"setup": {"hooks": [hook]}}) == problems


@pytest.mark.parametrize(
    ("call", "problems"),
    [
        ("register_hook", []),
        ("add_room_hook", []),
        ("hook_engine.add_room_hook", []),
        ("kit_executor.add_owner_room_hook", []),
        ("kit_executor.hook", ["steps[0].hook: as"]),
    ],
)
def test_a_hook_left_unnamed_is_labelled_where_it_is_registered_for_the_kit_or_a_room(
    call: str, problems: list[str]
) -> None:
    step = {"call": call, "hook": {"as": "first", "trigger": "after_broadcast"}}
    if call != "register_hook" and not call.startswith("kit_executor.hook"):
        step["room_id"] = "room-1"
    assert unknown_members({"steps": [step]}) == problems


@pytest.mark.parametrize("member", ["hooks", "room_hooks"])
def test_an_executor_defines_its_hooks_by_name(member: str) -> None:
    hook = {"as": "audit", "trigger": "on_tool_call"}
    scenario = {"setup": {"executors": [{"id": "w1", member: [hook]}]}}
    assert unknown_members(scenario) == [f"setup.executors[0].{member}[0]: as"]


@pytest.mark.parametrize(
    ("finish_reason", "problems"),
    [
        ("length", []),
        (None, []),
        (
            "max_tokens",
            ["setup.channels[0].model.rounds[0].finish_reason: 'max_tokens' is not a stop reason"],
        ),
    ],
)
def test_a_round_ends_on_a_stop_reason_of_the_contract(
    finish_reason: str | None, problems: list[str]
) -> None:
    round_ = {"text": "Half an ans", "finish_reason": finish_reason}
    scenario = {"setup": {"channels": [{"id": "ai", "type": "ai", "model": {"rounds": [round_]}}]}}
    assert unknown_members(scenario) == problems
