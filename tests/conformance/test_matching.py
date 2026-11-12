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


def test_acp_option_ids_are_not_event_ids() -> None:
    options = [
        {"id": "model", "type": "select", "currentValue": "opus"},
        {"id": "fast", "type": "boolean", "currentValue": True},
    ]
    normalizer = Normalizer(Observed(literal_rooms=set(), steps=[{"result": options}]))
    assert normalizer.value(options) == options
    assert normalizer.value({"configId": "model"}) == {"configId": "model"}
    assert normalizer.value({"event_id": "unrecorded"}) == {"event_id": "$e1"}
