"""A scenario's top-level members: the checker's citations are accepted, not run."""

from __future__ import annotations

import pytest

from tests.conformance._shape import unknown_members


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
