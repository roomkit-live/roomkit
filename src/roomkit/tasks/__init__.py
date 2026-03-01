"""Background task delegation via child rooms."""

from roomkit.tasks.base import OnCompleteCallback, TaskRunner
from roomkit.tasks.delegate import (
    DELEGATE_TOOL,
    DelegateHandler,
    build_delegate_tool,
    setup_delegation,
)
from roomkit.tasks.delivery import (
    BackgroundTaskDeliveryStrategy,
    ContextOnlyDelivery,
    ImmediateDelivery,
    TaskDeliveryContext,
    WaitForIdleDelivery,
)
from roomkit.tasks.memory import InMemoryTaskRunner
from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

__all__ = [
    "BackgroundTaskDeliveryStrategy",
    "ContextOnlyDelivery",
    "DELEGATE_TOOL",
    "DelegateHandler",
    "DelegatedTask",
    "DelegatedTaskResult",
    "ImmediateDelivery",
    "InMemoryTaskRunner",
    "OnCompleteCallback",
    "TaskDeliveryContext",
    "TaskRunner",
    "WaitForIdleDelivery",
    "build_delegate_tool",
    "setup_delegation",
]
