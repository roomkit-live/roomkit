"""Independent owners' realtime connections and deterministic presence failures."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from roomkit.realtime.base import EphemeralCallback, EphemeralEvent, RealtimeBackend
from roomkit.turns.base import TurnBackend

if TYPE_CHECKING:
    from roomkit.realtime import InMemoryRealtime
    from roomkit.turns import (
        ChannelControl,
        ChannelResult,
        ExecutorInfo,
        RoomOwner,
        TurnExecutor,
        TurnLink,
        TurnRequest,
    )


class OwnerFaults:
    def __init__(self, reach: Callable[[str], None]) -> None:
        self.reach = reach
        self.gates: dict[str, str] = {}
        self.armed: dict[str, bool] = {}
        self.problems: list[str] = []

    def set(self, operation: str, enabled: bool, gate: str | None) -> None:
        if enabled:
            self.armed[operation] = False
            if gate:
                self.gates[operation] = gate
        elif not self.armed.pop(operation, False):
            self.problems.append(f"presence fault {operation} was never taken")

    def check(self, operation: str) -> None:
        if operation in self.armed:
            self.armed[operation] = True
            if gate := self.gates.pop(operation, None):
                self.reach(gate)
            raise RuntimeError(f"scripted failure of {operation}")

    def untaken(self) -> list[str]:
        return self.problems + [
            f"untaken {op} fault" for op, taken in self.armed.items() if not taken
        ]


class WithoutCatalogue(TurnBackend):
    """Carry the scenario's turns while retaining the base optional catalogue."""

    def __init__(self, backend: TurnBackend) -> None:
        self.inner = backend
        self.timeouts = backend.timeouts

    async def submit(self, request: TurnRequest) -> TurnLink:
        return await self.inner.submit(request)

    async def submit_to(self, request: TurnRequest, executor: ExecutorInfo) -> TurnLink:
        return await self.inner.submit_to(request, executor)

    async def serve(self, executor: TurnExecutor) -> None:
        await self.inner.serve(executor)

    async def control(self, message: ChannelControl) -> ChannelResult | None:
        return await self.inner.control(message)

    async def attend(self, owner: RoomOwner) -> None:
        await self.inner.attend(owner)

    async def claim_presence(self, channel_id: str, room_id: str, online: bool | None) -> bool:
        return await self.inner.claim_presence(channel_id, room_id, online)

    async def close(self) -> None:
        await self.inner.close()


class FaultingPresence(WithoutCatalogue):
    """Forward the real catalogue and interrupt a claim only while its fault is armed."""

    def __init__(self, backend: TurnBackend, faults: OwnerFaults) -> None:
        super().__init__(backend)
        self.faults = faults

    async def executors(self) -> list[ExecutorInfo]:
        return await self.inner.executors()

    async def executors_now(self) -> list[ExecutorInfo]:
        return await self.inner.executors_now()

    def watch_executors(self, listener: Callable[[], Awaitable[None]]) -> None:
        self.inner.watch_executors(listener)

    async def claim_presence(self, channel_id: str, room_id: str, online: bool | None) -> bool:
        if online is not None:
            self.faults.check("claim_presence")
        return await super().claim_presence(channel_id, room_id, online)


class OwnerRealtime(RealtimeBackend):
    """A connection to the scenario bus. Closing it removes its own subscriptions."""

    def __init__(self, bus: InMemoryRealtime, faults: OwnerFaults) -> None:
        self.bus = bus
        self.subscriptions: set[str] = set()
        self.faults = faults

    async def publish(self, channel: str, event: EphemeralEvent) -> None:
        if event.type.value in {"presence_online", "presence_offline"}:
            self.faults.check("publish_presence")
        await self.bus.publish(channel, event)

    async def subscribe(self, channel: str, callback: EphemeralCallback) -> str:
        subscription = await self.bus.subscribe(channel, callback)
        self.subscriptions.add(subscription)
        return subscription

    async def unsubscribe(self, subscription_id: str) -> bool:
        self.subscriptions.discard(subscription_id)
        return await self.bus.unsubscribe(subscription_id)

    async def close(self) -> None:
        for subscription in list(self.subscriptions):
            await self.unsubscribe(subscription)
