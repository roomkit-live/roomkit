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
from roomkit.tasks.status import (
    CANCEL_TASK_TOOL,
    TASK_STATUS_TOOL,
    CancelTaskTool,
    TaskStatusTool,
    post_task_progress,
)

__all__ = [
    "CANCEL_TASK_TOOL",
    "CancelTaskTool",
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
    "post_task_progress",
    "setup_delegation",
    "setup_realtime_delegation",
]
