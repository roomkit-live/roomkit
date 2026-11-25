"""Task and observation models."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from roomkit.models._time import UtcDatetime
from roomkit.models.enums import TaskStatus


class Task(BaseModel):
    """A task assigned within a room."""

    id: str
    room_id: str
    title: str
    description: str | None = None
    assigned_to: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """An observation produced by an intelligence channel."""

    id: str
    room_id: str
    channel_id: str
    content: str
    category: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: UtcDatetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


TASK_END_WORDS: Mapping[str, str] = MappingProxyType(
    {
        "max_rounds": "it reached its limit of tool rounds before giving an answer",
        "timeout": "it ran out of time before giving an answer",
        "budget_exceeded": "it used up its token budget before giving an answer",
        "truncated": "its answer was cut at its model's output limit",
        "max_tokens": "its answer was cut at its model's output limit",
        "empty_response": "it gave no answer",
        "unfinished": "it stopped before doing what it had announced",
        "force_stopped": "it was stopped before giving an answer",
        "max_turn_requests": "it reached its limit of model requests before giving an answer",
        "refusal": "its model refused to answer",
        "interrupted": "it was interrupted before giving an answer",
        "cancelled": "it was cancelled before giving an answer",
        "error": "an error ended it",
    }
)
"""Why a delegated task did not complete, as a model may read it (RFC §23.3):
the closed list of its end reasons, each with the runtime's words for it. A
turn cut short ends by its ``loop_end_reason`` or an ACP agent's stop reason,
a task cancelled from outside ``cancelled``, one that failed with no error (its
worker's turn answered nothing) ``empty_response``, and any other failure
``error``, whose detail is for the logs and the hooks alone."""

END_REASON_KEYS = ("end_reason", "end_reason_text")
"""What a task's line and a call's answer say of why a task did not complete:
its code, then the runtime's words for it."""


def end_reason_fields(reason: Any) -> dict[str, str]:
    """*reason* as a task's line and a call's answer give it, by its code and
    in the runtime's words (:data:`END_REASON_KEYS`): nothing for a reason
    :data:`TASK_END_WORDS` does not name, so a reason written from elsewhere
    gives a model no words of its own (RFC §23.4)."""
    if not isinstance(reason, str) or reason not in TASK_END_WORDS:
        return {}
    return dict(zip(END_REASON_KEYS, (reason, TASK_END_WORDS[reason]), strict=True))


def failed_words(reason: str) -> str:
    """That a task failed and why, as a model reads it: *reason*, one of
    :data:`TASK_END_WORDS`, in the runtime's words then by its code."""
    return f"failed: {TASK_END_WORDS[reason]} ({reason})"


class DelegatedTaskResult(BaseModel):
    """Result of a completed delegated task."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    child_room_id: str
    parent_room_id: str
    agent_id: str
    status: TaskStatus = TaskStatus.COMPLETED
    output: str | None = None
    error: str | None = None
    exception: SkipJsonSchema[Exception | None] = Field(default=None, exclude=True)
    """The failure itself, its type kept (what the worker's turn raised),
    for a caller that hands it on; held in memory, never serialized (RFC
    §23.3 step 6)."""
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
