"""The run's store, wrapped so that a step makes one of its operations fail or
hold (roomkit-specs ``conformance/README.md``, "Setup": ``kit.store_faults``;
"Steps": ``store_fault``).

Every operation of the store's interface (RFC §14.1) runs as the wrapped
store's own, unless a ``store_fault`` step armed a fault on it: the first call
of one of the fault's operations made after the step, on its ``room_id`` when
it names one, takes the fault. With ``fail`` that call raises
:class:`StoreFaultError` in place of the operation; with ``gate`` it holds at
the gate once the operation answered, or in its place before it fails.

With ``kit.store_owner`` the wrapper is the run's store as its owner closes it:
``kit``, its close closes it and every operation made after fails with
:class:`StoreClosedError`, as a closed store refuses them; ``integrator``, its
close leaves it open, as a pool the integrator handed in.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from roomkit.store.base import ConversationStore

Hold = Callable[[str], Awaitable[None]]


class StoreFaultError(Exception):
    """What an operation a store fault fails raises: the store's own failure."""


class StoreClosedError(StoreFaultError, RuntimeError):
    """What an operation of a store the kit closed raises (``store_owner``
    ``kit``): a closed store's own failure, a ``RuntimeError`` as roomkit's
    stores raise once closed."""


@dataclass
class _Fault:
    operations: frozenset[str]
    room_id: str | None
    fail: str | None
    gate: str | None
    path: str


class StoreFaults:
    """The faults the steps armed, each taken by one call."""

    def __init__(self, hold: Hold) -> None:
        self._hold = hold
        self._armed: list[_Fault] = []
        # Who owns the store (README, ``kit.store_owner``), and whether the
        # kit's close closed it.
        self.owner: str | None = None
        self.closed = False

    def arm(self, step: Mapping[str, Any], path: str, room_id: str | None) -> None:
        """Arms the fault *step* names, for the next call it matches."""
        named = step["store_fault"]
        operations = frozenset([named] if isinstance(named, str) else named)
        if unknown := sorted(operations - _OPERATIONS):
            raise ValueError(f"{path}: no store operation {', '.join(unknown)}")
        self._armed.append(_Fault(operations, room_id, step.get("fail"), step.get("gate"), path))

    def take(self, operation: str, room_id: str | None) -> _Fault | None:
        """The first armed fault this call takes, disarmed; ``None`` for none."""
        for fault in self._armed:
            if operation in fault.operations and fault.room_id in (None, room_id):
                self._armed.remove(fault)
                return fault
        return None

    def untaken(self) -> list[str]:
        """What is left armed, each a problem of its step; disarms it."""
        left = [f"{fault.path}: no store call took the fault" for fault in self._armed]
        self._armed.clear()
        return left

    async def run(self, fault: _Fault, operation: Callable[[], Awaitable[Any]]) -> Any:
        """*operation* as *fault* makes it: held at its gate, then answered or failed."""
        if fault.fail is not None:
            if fault.gate is not None:
                await self._hold(fault.gate)
            raise StoreFaultError(fault.fail)
        result = await operation()
        if fault.gate is not None:
            await self._hold(fault.gate)
        return result


def _public_operations() -> dict[str, inspect.Signature]:
    """The operations of the store's interface, by name, with their signatures."""
    return {
        name: inspect.signature(member)
        for name, member in inspect.getmembers(ConversationStore, inspect.iscoroutinefunction)
        if not name.startswith("_") and name != "close"
    }


_SIGNATURES = _public_operations()
_OPERATIONS = frozenset(_SIGNATURES)


def _room_of(signature: inspect.Signature, *args: Any, **kwargs: Any) -> str | None:
    """The room a call names by its ``room_id`` argument, ``None`` for none;
    a call the interface's signature does not read (a store's own extra
    argument) names it by keyword alone."""
    try:
        return signature.bind(*args, **kwargs).arguments.get("room_id")
    except TypeError:
        return kwargs.get("room_id")


def _faulting(name: str) -> Callable[..., Awaitable[Any]]:
    signature = _SIGNATURES[name]

    async def operation(self: FaultingStore, *args: Any, **kwargs: Any) -> Any:
        if self._faults.closed:
            raise StoreClosedError(f"the store is closed: {name} refused")
        own = getattr(self._inner, name)
        fault = self._faults.take(name, _room_of(signature, self, *args, **kwargs))
        if fault is None:
            return await own(*args, **kwargs)
        return await self._faults.run(fault, lambda: own(*args, **kwargs))

    operation.__name__ = name
    return operation


def _delegating(name: str) -> Callable[..., Awaitable[Any]]:
    async def operation(self: FaultingStore, *args: Any, **kwargs: Any) -> Any:
        return await getattr(self._inner, name)(*args, **kwargs)

    operation.__name__ = name
    return operation


class FaultingStore(ConversationStore):
    """*inner*, each operation of its interface run as its own, unless a fault
    of *faults* takes the call."""

    def __init__(self, inner: ConversationStore, faults: StoreFaults) -> None:
        self._inner = inner
        self._faults = faults

    @property
    def inner(self) -> ConversationStore:
        """The store this one wraps."""
        return self._inner

    @property
    def is_process_local(self) -> bool:  # ty: ignore[invalid-method-override]
        return self._inner.is_process_local

    @property
    def _telemetry(self) -> Any:
        return getattr(self._inner, "_telemetry", None)

    @_telemetry.setter
    def _telemetry(self, telemetry: Any) -> None:
        self._inner._telemetry = telemetry  # ty: ignore[unresolved-attribute]

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[None]:
        async with self._inner.connection():
            yield

    async def close(self) -> None:
        await self._close_as_owned(self._inner.close)

    async def _close_bounded(self) -> None:
        """The kit's close of the store (``close_bounded``): the wrapped
        store's bounded close when it has one, as its owner allows."""
        await self._close_as_owned(getattr(self._inner, "_close_bounded", self._inner.close))

    async def _close_as_owned(self, close: Callable[[], Awaitable[None]]) -> None:
        if self._faults.owner == "integrator":
            # A pool the integrator handed in: the kit's close leaves it open.
            return
        await close()
        if self._faults.owner == "kit":
            self._faults.closed = True

    @property
    def owned_by_integrator(self) -> bool:
        """Whether its owner, not the kit, closes the store it wraps."""
        return self._faults.owner == "integrator"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


for _name in _OPERATIONS:
    setattr(FaultingStore, _name, _faulting(_name))
for _name, _member in inspect.getmembers(ConversationStore, inspect.iscoroutinefunction):
    if _name.startswith("_") and not _name.startswith("__"):
        setattr(FaultingStore, _name, _delegating(_name))
FaultingStore.__abstractmethods__ = frozenset()
