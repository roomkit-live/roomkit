"""Public presence model data and matching, without serving remote executors."""

from __future__ import annotations

import pytest

from roomkit.turns import AgentPresence, ExecutorInfo


def test_executor_info_defaults_are_explicit_and_not_an_every_channel_declaration() -> None:
    info = ExecutorInfo(executor_id="worker")
    assert info.model_dump(mode="json") == {
        "executor_id": "worker",
        "node": None,
        "hosts": None,
        "organizations": None,
        "room_hooks": [],
        "last_beat": None,
        "online": True,
    }
    assert not info.covers("helper")
    assert AgentPresence(channel_id="", online=False, executors=[]).model_dump() == {
        "channel_id": "",
        "online": False,
        "executors": [],
    }


@pytest.mark.parametrize(
    ("hosts", "organizations", "channel", "organization", "expected"),
    [
        ({"helper"}, None, "helper", None, True),
        ({"helper"}, None, "other", None, False),
        ({"acme-*"}, None, "acme-helper", None, True),
        ({"acme-*"}, None, "acme", None, False),
        ({"*"}, None, "any", None, True),
        ({"*"}, None, "any", "tenant-a", False),
        ({"*"}, {"tenant-a"}, "any", "tenant-a", True),
        ({"*"}, {"tenant-a"}, "any", None, False),
        ({"*"}, {"*"}, "any", None, True),
        ({"*"}, {"*"}, "any", "tenant-b", True),
        (set(), {"*"}, "any", None, False),
    ],
)
def test_public_executor_coverage_respects_channels_and_organizations(
    hosts: set[str],
    organizations: set[str] | None,
    channel: str,
    organization: str | None,
    expected: bool,
) -> None:
    info = ExecutorInfo(executor_id="worker", hosts=hosts, organizations=organizations)
    assert info.covers(channel, organization_id=organization) is expected


def test_coverage_gates_online_hooks_and_organization_before_validating_host_globs() -> None:
    bad = ExecutorInfo(executor_id="worker", hosts={"a*b"})
    assert not bad.model_copy(update={"online": False}).covers("ab")
    assert not bad.covers("ab", room_hooks=["missing"])
    assert not bad.covers("ab", organization_id="other")
    with pytest.raises(ValueError) as error:
        bad.covers("ab")
    assert str(error.value) == (
        "hosts entry 'a*b': a '*' ends an entry to declare a prefix "
        "('org-*'), and stands nowhere else"
    )
    ready = ExecutorInfo(executor_id="worker", hosts={"*"}, room_hooks={"audit", "trace"})
    assert ready.covers("any", room_hooks=["audit"])
    assert not ready.covers("any", room_hooks=["unknown"])
