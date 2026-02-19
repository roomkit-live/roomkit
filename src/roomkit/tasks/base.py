"""TaskRunner ABC for pluggable background task execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit

OnCompleteCallback = Callable[[DelegatedTaskResult], Awaitable[None]]


class TaskRunner(ABC):
    """ABC for executing delegated tasks in the background.

    Follows the same pluggable-backend pattern as ``ConversationStore``
    and ``RoomLockManager``.
    """

    @abstractmethod
    async def submit(
        self,
        kit: RoomKit,
        task: DelegatedTask,
        *,
        context: dict[str, Any] | None = None,
        on_complete: OnCompleteCallback | None = None,
    ) -> None:
        """Start background execution of *task*.

        Args:
            kit: The RoomKit instance (used to interact with rooms/channels).
            task: The delegated task handle.
            context: Optional context passed to the agent.
            on_complete: Callback invoked when the task finishes.
        """

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """Cancel a running task. Returns True if found and cancelled."""

    @abstractmethod
    async def close(self) -> None:
        """Shutdown the runner, cancelling all in-flight tasks."""
