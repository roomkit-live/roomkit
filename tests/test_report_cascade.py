"""The public cascade report copies its lists and fills the caller's own record."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from roomkit import DeliveryHandle, DeliveryResult, InboundResult, ResponseMetadata, RoomEvent
from tests.conftest import make_event


@dataclass
class _Report:
    """Only the data a report needs, with no cascade waiting or cancellation."""

    delivery_results: dict[str, DeliveryResult] = field(default_factory=dict)
    unavailable_targets: list[str] = field(default_factory=list)
    response_metadata: ResponseMetadata = field(default_factory=ResponseMetadata)
    response_events: list[RoomEvent] = field(default_factory=list)
    error: Exception | None = None
    cancelled: str | None = None


def test_delivery_results_are_replaced_and_share_the_reported_mapping() -> None:
    previous = DeliveryResult(channel_id="old", status="sent")
    delivered = DeliveryResult(channel_id="sms", status="sent")
    result = InboundResult(delivery_results={"old": previous})
    original = result.delivery_results
    report = _Report(delivery_results={"sms": delivered})

    result.report_cascade(cast(Any, report))

    assert result.delivery_results is report.delivery_results
    assert result.delivery_results["sms"] is delivered
    assert list(result.delivery_results) == ["sms"]
    assert original == {"old": previous}
    report.delivery_results["late"] = DeliveryResult(channel_id="late", status="queued")
    delivered.provider_message_id = "provider-1"
    assert result.delivery_results["late"].status == "queued"
    assert result.delivery_results["sms"].provider_message_id == "provider-1"

    report.delivery_results = {}
    assert "sms" in result.delivery_results  # A replacement needs another report.
    result.report_cascade(cast(Any, report))
    assert result.delivery_results is report.delivery_results
    assert result.delivery_results == {}


@pytest.mark.parametrize("duplicate", [False, True], ids=["new", "duplicate"])
@pytest.mark.parametrize("initial", [None, ["previous"]], ids=["none", "previous"])
def test_availability_is_copied_except_on_a_duplicate(
    duplicate: bool, initial: list[str] | None
) -> None:
    result = InboundResult(duplicate=duplicate, unavailable_targets=initial)
    original = result.unavailable_targets
    report = _Report(unavailable_targets=["missing", "missing", "absent"])

    result.report_cascade(cast(Any, report))

    if duplicate:
        assert result.unavailable_targets is original
        assert result.unavailable_targets == initial
    else:
        assert result.unavailable_targets == ["missing", "missing", "absent"]
        assert result.unavailable_targets is not report.unavailable_targets
    report.unavailable_targets.clear()
    assert result.unavailable_targets == (
        initial if duplicate else ["missing", "missing", "absent"]
    )
    result.report_cascade(cast(Any, report))
    assert result.unavailable_targets == (initial if duplicate else [])


def test_reporting_merges_into_the_record_the_caller_already_holds() -> None:
    record = ResponseMetadata({"keep": 1, "replace": "old"})
    incoming = ResponseMetadata({"replace": "new", "added": [1]})
    result = InboundResult(response_metadata=record)
    report = _Report(response_metadata=incoming)
    before = record.writes

    result.report_cascade(cast(Any, report))

    assert result.response_metadata is record
    assert result.response_metadata is not incoming
    assert dict(record) == {"keep": 1, "replace": "new", "added": [1]}
    assert record.writes - before == 2
    incoming["replace"] = "later"
    assert record["replace"] == "new"
    # Values are kept, not deep-copied: changing a held value is no key write.
    incoming["added"].append(2)
    assert record["added"] == [1, 2]
    assert record.writes - before == 2


@pytest.mark.parametrize(
    ("initial", "incoming", "expected", "writes"),
    [
        ({}, {}, {}, 0),
        ({"turns": {"a": {"end": "old"}}}, {}, {"turns": {"a": {"end": "old"}}}, 1),
        ({}, {"turns": {}}, {"turns": {}}, 1),
        ({}, {"turns": 3}, {"turns": 3}, 1),
        ({"turns": "old"}, {}, {"turns": "old"}, 0),
        ({"turns": "old"}, {"turns": None}, {"turns": None}, 1),
        (
            {"turns": {"a": {"end": "old"}}},
            {"turns": 3},
            {"turns": {"a": {"end": "old"}}},
            2,
        ),
        (
            {"turns": {"a": {"end": "old"}}},
            {"turns": {"b": {"end": "new"}}},
            {"turns": {"a": {"end": "old"}, "b": {"end": "new"}}},
            2,
        ),
        (
            {"turns": {"a": {"end": "old", "usage": 1}, "b": {"end": "kept"}}},
            {"turns": {"a": {"end": "new"}}},
            {"turns": {"a": {"end": "new"}, "b": {"end": "kept"}}},
            2,
        ),
    ],
    ids=[
        "absent",
        "retained-channel",
        "empty-turns",
        "nonobject-turns",
        "retained-nonobject",
        "null-over-nonobject",
        "channel-over-nonobject",
        "distinct-channels",
        "replace-whole-channel-entry",
    ],
)
def test_turn_entries_merge_per_channel_and_each_report_counts_its_writes(
    initial: dict[str, Any], incoming: dict[str, Any], expected: dict[str, Any], writes: int
) -> None:
    record = ResponseMetadata(initial)
    result = InboundResult(response_metadata=record)
    report = _Report(response_metadata=ResponseMetadata(incoming))
    before = record.writes

    result.report_cascade(cast(Any, report))

    assert result.response_metadata is record
    assert dict(record) == expected
    assert record.writes - before == writes
    result.report_cascade(cast(Any, report))
    assert dict(record) == expected
    assert record.writes - before == 2 * writes
    assert dict(report.response_metadata) == incoming


def test_response_events_copy_the_list_but_keep_its_events() -> None:
    previous = make_event(body="previous")
    first, second = make_event(body="first"), make_event(body="second")
    result = InboundResult(response_events=[previous])
    original = result.response_events
    report = _Report(response_events=[second, first, second])

    result.report_cascade(cast(Any, report))

    assert result.response_events is not report.response_events
    assert len(result.response_events) == 3
    assert all(
        actual is expected
        for actual, expected in zip(result.response_events, [second, first, second], strict=True)
    )
    assert original == [previous]
    report.response_events.clear()
    second.metadata["reported-later"] = True
    assert len(result.response_events) == 3
    assert result.response_events[0].metadata["reported-later"] is True
    result.report_cascade(cast(Any, report))
    assert result.response_events == []
    assert result.response_events is not report.response_events


@pytest.mark.parametrize("existing", [False, True], ids=["no-error", "earlier-error"])
@pytest.mark.parametrize("incoming", [False, True], ids=["no-failure", "cascade-failed"])
@pytest.mark.parametrize("cancelled", [None, "cut"], ids=["not-cancelled", "cancelled"])
def test_the_first_error_stays_while_cancellation_is_always_replaced(
    existing: bool, incoming: bool, cancelled: str | None
) -> None:
    earlier = ValueError("earlier") if existing else None
    failure = RuntimeError("cascade") if incoming else None
    result = InboundResult(error=earlier, cancellation_reason="previous")
    report = _Report(error=failure, cancelled=cancelled)

    result.report_cascade(cast(Any, report))

    expected = earlier if existing else failure
    assert result.error is expected
    assert result.cancellation_reason == cancelled
    report.error = RuntimeError("later")
    report.cancelled = None
    result.report_cascade(cast(Any, report))
    assert result.error is (expected if expected is not None else report.error)
    assert result.cancellation_reason is None


async def test_reporting_is_synchronous_and_keeps_the_decision_and_handle() -> None:
    report = _Report()
    event = make_event()
    result = InboundResult(event=event, blocked=True, reason="decided", duplicate=True)

    async def finished() -> None:
        return

    consumer = asyncio.create_task(finished())
    await consumer
    handle = DeliveryHandle(cast(Any, report), consumer, result)
    result.delivery = handle

    returned = result.report_cascade(cast(Any, report))

    assert returned is None
    assert result.event is event
    assert (result.blocked, result.reason, result.duplicate) == (True, "decided", True)
    assert result.delivery is handle
