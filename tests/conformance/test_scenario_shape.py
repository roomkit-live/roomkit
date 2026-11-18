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
