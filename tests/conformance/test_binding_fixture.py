"""Malformed synchronous binding controls cannot silently run a no-op."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conformance._shape import unknown_members


@pytest.mark.parametrize("control", [None, [], {"fail": False}, {"gate": "wait"}])
def test_binding_callback_controls_are_synchronous_and_explicit(control: Any) -> None:
    scenario = {"setup": {"channels": [{"id": "sms", "type": "sms", "update_binding": control}]}}
    assert unknown_members(scenario)
