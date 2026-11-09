"""Malformed receipts fail the fixture before any delivery runs."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conformance._shape import unknown_members


@pytest.mark.parametrize(
    "receipt",
    [
        {},
        {"success": "yes"},
        {"success": 1},
        {"success": True, "provider_id": "typo"},
        {"success": True, "metadata": []},
        {"success": True, "error": 5},
        {"success": True, "provider_message_id": False},
        [],
        "bad",
    ],
)
def test_a_malformed_provider_receipt_is_refused_before_execution(receipt: Any) -> None:
    channel = {"id": "out", "type": "sms", "deliver": [{"output": {"provider_result": receipt}}]}
    assert unknown_members({"setup": {"channels": [channel]}})


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        {"success": True},
        {
            "success": True,
            "provider_message_id": "m1",
            "error": None,
            "metadata": {"accepted": True},
        },
    ],
)
def test_a_provider_receipt_is_an_explicit_supported_output(receipt: Any) -> None:
    channel = {"id": "out", "type": "sms", "deliver": [{"output": {"provider_result": receipt}}]}
    assert unknown_members({"setup": {"channels": [channel]}}) == []
