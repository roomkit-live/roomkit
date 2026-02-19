"""Background task delegation via child rooms."""

from roomkit.tasks.base import OnCompleteCallback, TaskRunner
from roomkit.tasks.delegate import (
    DELEGATE_TOOL,
    DelegateHandler,
    build_delegate_tool,
    setup_delegation,
)
from roomkit.tasks.memory import InMemoryTaskRunner
from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult

__all__ = [
    "DELEGATE_TOOL",
    "DelegateHandler",
    "DelegatedTask",
    "DelegatedTaskResult",
    "InMemoryTaskRunner",
    "OnCompleteCallback",
    "TaskRunner",
    "build_delegate_tool",
    "setup_delegation",
]
