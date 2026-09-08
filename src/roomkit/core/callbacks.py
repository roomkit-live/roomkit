"""Small callback subscriptions with explicit, idempotent removal."""

from __future__ import annotations

from collections.abc import Callable


def subscribe_callback[T](callbacks: list[T], callback: T) -> Callable[[], None]:
    """Append a callback and return a function removing this registration once."""
    callbacks.append(callback)
    removed = False

    def unsubscribe() -> None:
        nonlocal removed
        if not removed:
            removed = True
            if callback in callbacks:
                callbacks.remove(callback)

    return unsubscribe
