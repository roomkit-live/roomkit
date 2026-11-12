"""The turn's record across an owner-served call, its rule alone (RFC §10.6.5).

The handler on the owner works on a copy of the record its call carried; what
it changed, measured against that copy, is applied to the executor's record.
"""

from __future__ import annotations

from typing import Any

from roomkit.models.response_metadata import ResponseMetadata
from roomkit.tools.context import ToolCallContext, current_response_metadata, owner_served_call
from roomkit.turns._call_record import apply_changes, changes_of


def _served(carried: dict[str, Any], write: Any) -> Any:
    """What a handler running *write* on the owner changed of *carried*."""
    with owner_served_call(
        ToolCallContext(room_id="r1", channel_id="agent"),
        room=None,
        actor_id=None,
        requester=None,
        chain_depth=1,
        allowed_names=None,
        response_metadata=carried,
    ) as worked:
        record = current_response_metadata()
        assert record is worked
        write(record)
    return changes_of(carried, worked)


def test_a_handler_works_on_a_copy_nothing_of_which_is_the_carried_record() -> None:
    carried = {"names": {"a": 1}}

    changes = _served(carried, lambda record: record["names"].update(b=2))

    assert carried == {"names": {"a": 1}}
    assert changes is not None
    assert changes.written == {"names": {"a": 1, "b": 2}}


def test_a_call_made_outside_the_turn_reads_no_record() -> None:
    with owner_served_call(
        ToolCallContext(room_id="r1", channel_id="agent"),
        room=None,
        actor_id=None,
        requester=None,
        chain_depth=0,
        allowed_names=None,
    ) as worked:
        assert current_response_metadata() is None
    assert worked is None


def test_a_handler_that_changed_nothing_reports_nothing() -> None:
    assert _served({"seen": 1}, lambda record: record.update(seen=1)) is None


def test_a_key_removed_is_one_the_call_carried_never_one_written_since() -> None:
    record = ResponseMetadata({"carried": 1, "written_meanwhile": "x"})
    carried = {"carried": 1}

    apply_changes(record, carried, _served(carried, lambda r: r.pop("carried")))

    assert dict(record) == {"written_meanwhile": "x"}


def test_two_handlers_extending_one_mapping_keep_both_extensions() -> None:
    record = ResponseMetadata({"names": {"a": 1}})
    carried = {"names": {"a": 1}}
    first = _served(carried, lambda r: r["names"].update(b=2))
    second = _served(carried, lambda r: r["names"].update(c=3))

    apply_changes(record, carried, first)
    apply_changes(record, carried, second)

    assert record["names"] == {"a": 1, "b": 2, "c": 3}


def test_a_mapping_the_call_did_not_carry_adds_its_entries_to_the_records() -> None:
    record = ResponseMetadata({})
    first = _served({}, lambda r: r.update(names={"b": 2}))
    second = _served({}, lambda r: r.update(names={"c": 3}))

    apply_changes(record, {}, first)
    apply_changes(record, {}, second)

    assert record["names"] == {"b": 2, "c": 3}


def test_an_entry_removed_from_a_mapping_is_one_the_call_carried() -> None:
    record = ResponseMetadata({"names": {"a": 1, "b": 2, "late": 9}})
    carried = {"names": {"a": 1, "b": 2}}

    apply_changes(record, carried, _served(carried, lambda r: r["names"].pop("a")))

    assert record["names"] == {"b": 2, "late": 9}


def test_a_value_other_than_a_mapping_is_replaced_by_the_one_answered_last() -> None:
    record = ResponseMetadata({"read": ["one"]})
    carried = {"read": ["one"]}
    first = _served(carried, lambda r: r["read"].append("two"))
    second = _served(carried, lambda r: r["read"].append("three"))

    apply_changes(record, carried, first)
    apply_changes(record, carried, second)

    assert record["read"] == ["one", "three"]
