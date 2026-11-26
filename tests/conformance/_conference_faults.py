"""A conference's scripted faults, credentials and close (RFC §12.10.3, §12.10.4).

``conference.fault`` arms a fault on the scripted SFU (roomkit-specs
``conformance/README.md``, "Scripted conference faults"): the call of a
method it takes holds at a gate once received, then fails as the backend's
own failure. ``conference.open`` arms its ``faults`` before the attach and
gives the sessions the SFU returns its ``bot_joined_at``.
``conference.mint`` asks the channel for a participant's credential, and
``conference.close`` closes the channel itself, apart from the kit.

Kept free of the other runner modules: ``_calls`` maps this module's error,
and the scripted SFU (``_conference_sfu``) consults its faults.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic_core import to_jsonable_python

from roomkit import ConferenceGrants

if TYPE_CHECKING:
    from roomkit import RoomKit
    from roomkit.conference import MockConferenceCall
    from tests.conformance._conference import ConferenceFixture
    from tests.conformance._harness import Harness
    from tests.conformance._script import Gates

# The calls a fault may take: those ``conference.read`` lists.
_METHODS = frozenset(
    {
        "ensure_room",
        "close_room",
        "list_participants",
        "join_as_bot",
        "update_bot_grants",
        "leave",
        "stop_playback",
        "subscribe_track",
        "unsubscribe_track",
        "publish_audio",
        "publish_video",
        "mint_access",
        "remove_participant",
        "mute_track",
        "unmute_track",
        "close",
    }
)
_FAULT = frozenset({"method", "fail", "gate", "after", "ignores_cancel"})


class ScriptedBackendError(Exception):
    """What a call a fault fails raises: the backend's own failure
    (README, ``backend_failed``)."""


class UnarmableFaultError(Exception):
    """A fault the scenario names that no call of the scripted SFU could take:
    the scenario's mistake, which no error name covers."""


@dataclass
class _Fault:
    method: str
    fail: str | None
    gate: str | None
    after: int
    ignores_cancel: bool
    taken: bool = False


class SFUFaults:
    """The faults the scenario armed on one scripted SFU, each taken by one
    call; those on one method take its calls in the order they were armed."""

    def __init__(self, gates: Gates) -> None:
        self._gates = gates
        self._armed: list[_Fault] = []

    def arm(self, spec: Mapping[str, Any]) -> None:
        if unknown := sorted(set(spec) - _FAULT):
            raise UnarmableFaultError(f"a conference fault takes no {', '.join(unknown)}")
        method = spec["method"]
        if method not in _METHODS:
            raise UnarmableFaultError(f"the scripted SFU has no call {method!r}")
        if spec.get("fail") is None and spec.get("gate") is None:
            raise UnarmableFaultError("a conference fault needs a fail, a gate, or both")
        self._armed.append(
            _Fault(
                method,
                spec.get("fail"),
                spec.get("gate"),
                int(spec.get("after", 0)),
                bool(spec.get("ignores_cancel", False)),
            )
        )

    async def take(self, method: str, record: MockConferenceCall) -> None:
        """What the call *record*, just received, does under the fault it
        takes, if any: hold at its gate, then fail."""
        fault = self._next(method)
        if fault is None:
            return
        if fault.gate is not None:
            await self._hold(fault.gate, fault.ignores_cancel, record)
        if fault.fail is not None:
            raise ScriptedBackendError(fault.fail)

    def _next(self, method: str) -> _Fault | None:
        """The fault this call of *method* takes: the first one armed on the
        method and still untaken lets it through while its ``after`` lasts."""
        for fault in self._armed:
            if fault.method != method or fault.taken:
                continue
            if fault.after > 0:
                fault.after -= 1
                return None
            fault.taken = True
            return fault
        return None

    async def _hold(self, gate: str, ignores_cancel: bool, record: MockConferenceCall) -> None:
        """Hold at *gate*. A cancellation marks the call ``cancelled``; it ends
        the call, or, with *ignores_cancel*, the call holds on until the gate
        opens, as a request that does not honour its cancellation."""
        held = asyncio.ensure_future(self._gates.hold(gate))
        while True:
            try:
                await asyncio.shield(held)
                return
            except asyncio.CancelledError:
                # The call reads it among conference.read's ``calls``.
                record.args["cancelled"] = True
                if not ignores_cancel:
                    held.cancel()
                    raise

    def untaken(self) -> list[str]:
        return [
            f"conference fault on {fault.method} was never taken"
            for fault in self._armed
            if not fault.taken
        ]


def bot_joined_at(spec: Mapping[str, Any]) -> datetime | None:
    """``conference.open``'s ``bot_joined_at``: naive when it names no offset."""
    value = spec.get("bot_joined_at")
    return None if value is None else datetime.fromisoformat(value)


def arm_open_faults(faults: SFUFaults, spec: Mapping[str, Any]) -> None:
    """``conference.open``'s ``faults``, armed before the attach."""
    for fault in spec.get("faults", []):
        faults.arm(fault)


def _fixture(harness: Harness, spec: Mapping[str, Any]) -> ConferenceFixture:
    return harness.conferences[spec["channel_id"]]


async def _fault(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    held = getattr(_fixture(harness, spec).backend, "held", None)
    if not isinstance(held, SFUFaults):
        raise UnarmableFaultError(f"conference {spec['channel_id']!r} has no scripted SFU")
    held.arm({key: value for key, value in spec.items() if key not in {"call", "channel_id"}})


async def _mint(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    grants = spec.get("grants")
    access = await _fixture(harness, spec).channel.mint_access(
        spec["room_id"],
        spec["participant_id"],
        grants=None if grants is None else ConferenceGrants(**grants),
        attributes=spec.get("attributes"),
        organization_id=spec.get("organization_id"),
    )
    return to_jsonable_python(access)


async def _close(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    await _fixture(harness, spec).channel.close()


def untaken_faults(fixtures: Iterable[ConferenceFixture]) -> list[str]:
    """The faults no call took, each a problem of the run (README, "Scripted
    conference faults")."""
    problems: list[str] = []
    for fixture in fixtures:
        held = getattr(fixture.backend, "held", None)
        if isinstance(held, SFUFaults):
            problems.extend(held.untaken())
    return problems


def close_work(fixtures: Iterable[ConferenceFixture]) -> list[asyncio.Task[Any]]:
    """What a channel's close left running, an ``idle`` step waits for: the
    steps it abandoned and the closes it deferred until they end."""
    return [
        task
        for fixture in fixtures
        for task in fixture.channel._shutdown._background
        if not task.done()
    ]


FAULT_OPEN = frozenset({"faults", "bot_joined_at"})
FAULT_ARGUMENTS = {
    "conference.fault": {"channel_id"} | _FAULT,
    "conference.mint": {
        "channel_id",
        "room_id",
        "participant_id",
        "grants",
        "attributes",
        "organization_id",
    },
    "conference.close": {"channel_id"},
}
FAULT_CALLS = {
    "conference.fault": _fault,
    "conference.mint": _mint,
    "conference.close": _close,
}
