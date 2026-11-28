"""A channel's trace callbacks (README, Calls: ``channel.on_trace``,
``channel.trace_enabled``; Expectations: ``trace_callbacks``).

A scenario registers a callback of its own with a channel's ``on_trace``, as an
integrator does (RFC §15.6): a sync function that records each trace it is
handed as it is called, or an async one, which the channel schedules as a task,
that holds at its gate when it names one, then records the trace. Before the
run's traces are read, the runner waits for each async callback that started.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Coroutine, Mapping
from typing import TYPE_CHECKING, Any

from roomkit.models import ProtocolTrace
from tests.conformance._calls import Call, UnreadableError
from tests.conformance._script import GATE_BOUND_S

if TYPE_CHECKING:
    from roomkit import RoomKit
    from tests.conformance._harness import Harness


def trace_record(trace: ProtocolTrace) -> dict[str, Any]:
    """The §5.14 record of *trace*, its bytes in base64 (README, ``hooks``)."""
    raw = trace.raw
    return {
        "channel_id": trace.channel_id,
        "room_id": trace.room_id,
        "session_id": trace.session_id,
        "direction": trace.direction,
        "protocol": trace.protocol,
        "summary": trace.summary,
        "raw": {"base64": base64.b64encode(raw).decode("ascii")}
        if isinstance(raw, bytes)
        else raw,
        "metadata": dict(trace.metadata),
        "timestamp": trace.timestamp.isoformat(),
    }


class TraceCallbacks:
    """The callbacks the steps registered, each with the traces it recorded."""

    def __init__(self, hold: Callable[[str], Coroutine[Any, Any, None]]) -> None:
        self._hold = hold
        self.recorded: dict[str, list[dict[str, Any]]] = {}
        self._running: set[asyncio.Task[Any]] = set()

    def callback(self, name: str, spec: Mapping[str, Any]) -> Callable[[ProtocolTrace], Any]:
        """The callback *spec* describes, recording under *name*."""
        if name in self.recorded:
            raise UnreadableError(f"channel.on_trace: callback {name!r} is registered already")
        recorded = self.recorded[name] = []
        if not spec.get("async"):
            if "gate" in spec:
                raise UnreadableError("channel.on_trace: a gate needs an async callback")
            return lambda trace: recorded.append(trace_record(trace))
        gate = spec.get("gate")

        async def observe(trace: ProtocolTrace) -> None:
            task = asyncio.current_task()
            if task is not None:
                self._running.add(task)
            try:
                if gate is not None:
                    await self._hold(gate)
                recorded.append(trace_record(trace))
            finally:
                if task is not None:
                    self._running.discard(task)

        return observe

    async def observe(self) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
        """What each callback recorded, once every async one that started ended."""
        problems: list[str] = []
        running = set(self._running)
        if running:
            _, pending = await asyncio.wait(running, timeout=GATE_BOUND_S)
            if pending:
                problems.append(f"{len(pending)} async trace callback(s) did not end")
        return {name: list(traces) for name, traces in self.recorded.items()}, problems


async def _on_trace(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> None:
    channel = kit.get_channel(spec["channel_id"])
    if channel is None:
        raise UnreadableError(f"channel.on_trace: no registered channel {spec['channel_id']!r}")
    callback = harness.trace_callbacks.callback(spec["callback"], spec)
    channel.on_trace(callback, protocols=spec.get("protocols"))


async def _trace_enabled(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> bool:
    channel = kit.get_channel(spec["channel_id"])
    if channel is None:
        raise UnreadableError(
            f"channel.trace_enabled: no registered channel {spec['channel_id']!r}"
        )
    return channel.trace_enabled


TRACE_CALLS: dict[str, Call] = {
    "channel.on_trace": _on_trace,
    "channel.trace_enabled": _trace_enabled,
}

# Each call's arguments, beside `call` and `organization_id`.
TRACE_ARGUMENTS: dict[str, set[str]] = {
    "channel.on_trace": {"channel_id", "callback", "protocols", "async", "gate"},
    "channel.trace_enabled": {"channel_id"},
}
