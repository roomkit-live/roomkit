"""Expectation operators must reject a superficially similar observation."""

from __future__ import annotations

import pytest

from tests.conformance._match import MISSING, mismatches


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
