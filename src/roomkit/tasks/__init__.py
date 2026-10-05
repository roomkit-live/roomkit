"""Background task delegation via child rooms."""

from roomkit.tasks.base import OnCompleteCallback, TaskRunner
from roomkit.tasks.cache import CompletedTaskCache
from roomkit.tasks.delegate import (
    DELEGATE_TOOL,
    DelegateHandler,
    build_delegate_tool,
    setup_delegation,
    setup_realtime_delegation,
)
from roomkit.tasks.memory import InMemoryTaskRunner
from roomkit.tasks.models import DelegatedTask, DelegatedTaskResult
from roomkit.tasks.status import TASK_STATUS_TOOL, TaskStatusTool

__all__ = [
    "CompletedTaskCache",
    "DELEGATE_TOOL",
    "DelegateHandler",
    "DelegatedTask",
    "DelegatedTaskResult",
    "InMemoryTaskRunner",
    "OnCompleteCallback",
    "TASK_STATUS_TOOL",
    "TaskRunner",
    "TaskStatusTool",
    "build_delegate_tool",
    "setup_delegation",
    "setup_realtime_delegation",
]
