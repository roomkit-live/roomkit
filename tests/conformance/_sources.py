"""Scripted sources attached to the kit (roomkit-specs ``conformance/README.md``,
"Setup": ``sources``; "Calls": ``source_health``; "Expectations": ``sources``).

A scripted source's ``start``, ``stop`` and ``healthcheck`` each run their
reaction: its start then reads until it is cut, its health check answers the
source's health. Each call is recorded, and its start's cut.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from roomkit import RoomKit
from roomkit.sources import EmitCallback, SourceHealth, SourceProvider
from tests.conformance._script import Reactions, perform

if TYPE_CHECKING:
    from tests.conformance._harness import Harness


class ScriptedSource(SourceProvider):
    """A source on the source interface (RFC §6.5) whose calls follow the
    scenario's reactions."""

    def __init__(self, spec: Mapping[str, Any], harness: Harness) -> None:
        self._channel_id: str = spec["channel_id"]
        self._harness = harness
        self._start = Reactions(spec.get("start"))
        self._stop = Reactions(spec.get("stop"))
        self._healthcheck = Reactions(spec.get("healthcheck"))

    @property
    def name(self) -> str:
        return self._channel_id

    async def start(self, emit: EmitCallback) -> None:
        self._harness.record_source(self._channel_id, "start")
        try:
            await self._react(self._start)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self._harness.record_source(self._channel_id, "cut")
            raise

    async def stop(self) -> None:
        self._harness.record_source(self._channel_id, "stop")
        await self._react(self._stop)

    async def healthcheck(self) -> SourceHealth:
        self._harness.record_source(self._channel_id, "healthcheck")
        await self._react(self._healthcheck)
        return await super().healthcheck()

    async def _react(self, reactions: Reactions) -> None:
        reaction = reactions.pick({})
        if reaction is not None:
            await perform(self._harness, reaction, {})


async def attach_sources(kit: RoomKit, harness: Harness, specs: list[Mapping[str, Any]]) -> None:
    """Attach each scripted source of the setup to *kit*."""
    for spec in specs:
        options: dict[str, Any] = {"auto_restart": bool(spec.get("auto_restart", False))}
        if "restart_delay_ms" in spec:
            options["restart_delay"] = spec["restart_delay_ms"] / 1000
        await kit.attach_source(spec["channel_id"], ScriptedSource(spec, harness), **options)


async def _source_health(kit: RoomKit, harness: Harness, spec: Mapping[str, Any]) -> Any:
    health = await kit.source_health(spec["channel_id"])
    return None if health is None else str(health.status)


SOURCE_CALLS = {"source_health": _source_health}
SOURCE_ARGUMENTS = {"source_health": {"channel_id"}}
SCRIPTED_SOURCE = {
    "channel_id",
    "auto_restart",
    "restart_delay_ms",
    "start",
    "stop",
    "healthcheck",
}
