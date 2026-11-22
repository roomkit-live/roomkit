"""Expectation operators must reject a superficially similar observation."""

from __future__ import annotations

import pytest

from tests.conformance._match import MISSING, mismatches
from tests.conformance._normalize import Normalizer, Observed


@pytest.mark.parametrize(
    ("actual", "matches"),
    [
        ("Résultat", True),
        ("Résultat plus notes", True),
        ("prefix Résultat", False),
        ("", False),
        (None, False),
        (MISSING, False),
        ({"text": "Résultat"}, False),
    ],
)
def test_starts_with_requires_text_at_the_start(actual: object, matches: bool) -> None:
    assert (not mismatches({"$starts_with": "Résultat"}, actual)) is matches


@pytest.mark.parametrize(
    ("actual", "matches"),
    [
        ("running for 3 s; at “12/20 s” (1 s ago)", True),
        ("running for 3 s; at “12/20 s”", False),
        ("at “12/20 s” (1 s ago); running for 3 s", False),
        ("running for; at", False),
        (None, False),
    ],
)
def test_contains_a_list_holds_each_text_in_its_order(actual: object, matches: bool) -> None:
    expected = {"$contains": ["running for ", " s; at “12/20 s” (", " s ago)"]}
    assert (not mismatches(expected, actual)) is matches


@pytest.mark.parametrize("malformed", [[], ["a", 1], 1])
def test_contains_refuses_what_is_no_text_nor_list_of_texts(malformed: object) -> None:
    assert mismatches({"$contains": malformed}, "anything")


def test_acp_option_ids_are_not_event_ids() -> None:
    options = [
        {"id": "model", "type": "select", "currentValue": "large"},
        {"id": "fast", "type": "boolean", "currentValue": True},
    ]
    normalizer = Normalizer(Observed(literal_rooms=set(), steps=[{"result": options}]))
    assert normalizer.value(options) == options
    assert normalizer.value({"configId": "model"}) == {"configId": "model"}
    assert normalizer.value({"event_id": "unrecorded"}) == {"event_id": "$e1"}
