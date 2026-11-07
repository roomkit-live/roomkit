"""Long work started as a task: a turn talks, the work runs apart (RFC §23.4).

An agent given ``work_in_background`` starts long work (going through many
records, files or traces, an analysis of several tool calls) as a task, says
in a sentence that it is under way, and ends its turn; in a discussion the
floor moves on while the work runs (RFC §19.7.5 rule 19). The task is a
delegation (RFC §23.3): it runs on the agent's worker, the agent itself by
default or any intelligence channel the host names, handed the room's last
messages, and its result comes back to the agent, relayed or posted as the
host set it up. The host starts the same task with ``kit.start_task``, and a
person through whatever the host gives them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roomkit._text import fence, identifier
from roomkit.channels._discussion_turn import agent_handle
from roomkit.models.enums import ChannelCategory
from roomkit.providers.ai.base import AITool
from roomkit.tasks.brief import TASK_TAG
from roomkit.tasks.posting import HAND_BACKS
from roomkit.tools.context import current_tool_room

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.tasks.models import DelegatedTask

BACKGROUND_TOOL = "work_in_background"

BACKGROUND_CONVERSATION = 12
"""How many of the room's last messages a background task is handed."""

BACKGROUND_TOOL_DEF = AITool(
    name=BACKGROUND_TOOL,
    description=(
        "Start long work as a background task, so the conversation goes on: going "
        "through many records, logs, traces or files, or an analysis that takes several "
        "tool calls. It runs away from the conversation, handed its last messages, and "
        "its result comes back when it is done. Say once, in one short sentence, that the "
        "work is under way (before the call or after it, never both), and end your turn. "
        "A quick look you answer now, without it."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What to do and what to report, precisely.",
            },
        },
        "required": ["task"],
    },
)


@dataclass(frozen=True)
class BackgroundSetup:
    """How an agent's background tasks run: on *worker* (the agent itself
    when None), their result relayed or posted (*hand_back*)."""

    worker: str | None = None
    hand_back: str = "relay"

    def __post_init__(self) -> None:
        if self.hand_back not in HAND_BACKS:
            raise ValueError(f"hand_back must be one of {HAND_BACKS}, not {self.hand_back!r}")


def worker_brief(task: str, agent_name: str, hand_back: str) -> str:
    """What a background task's worker is told: the task, set apart as it was
    asked, then how to work and where its findings go. *agent_name* is the
    agent the task is for, as the room addresses it: its handle in a
    discussion, else its channel id. A model, the host or a person wrote the
    task, and the runtime's text and the room's conversation follow it: set
    apart, it cannot read as either (RFC §6.4)."""
    agent = identifier(agent_name, "agent")
    where = (
        f"posted to the conversation as it is, as @{agent}'s message, so write it for "
        "the people there"
        if hand_back == "post"
        else f"handed back to @{agent}, who shares it in the conversation"
    )
    return (
        f"Your task, as it was asked:\n{fence(TASK_TAG, task)}\n\n"
        f"You are doing this task in the background for @{agent}, away from the "
        "conversation: nobody there sees this work, and you address no one from here. "
        "Do the work now with your tools, then end with what you found, in a few short "
        f"lines: it is {where}."
    )


async def start_background_task(
    kit: RoomKit,
    room_id: str,
    agent_id: str,
    task: str,
    *,
    setup: BackgroundSetup,
    hand_back: str | None = None,
) -> DelegatedTask:
    """Start *task* for *agent_id*, an agent of *room_id*, on its worker, as
    *setup* says (*hand_back* overriding its own).

    Raises:
        ValueError: *agent_id* is no intelligence channel attached to the
            room, *task* is blank, or *hand_back* is unknown.
    """
    if not task.strip():
        raise ValueError("A background task needs a task")
    channel = kit.get_channel(agent_id)
    binding = await kit.store.get_binding(room_id, agent_id)
    if channel is None or channel.category != ChannelCategory.INTELLIGENCE or binding is None:
        raise ValueError(f"{agent_id!r} is not an agent of room {room_id!r}")
    mode = hand_back or setup.hand_back
    room = await kit.store.get_room(room_id)
    agent_name = agent_handle(room.metadata if room is not None else None, agent_id)
    return await kit.delegate(
        room_id,
        setup.worker or agent_id,
        task,
        brief=worker_brief(task, agent_name, mode),
        notify=agent_id,
        conversation=BACKGROUND_CONVERSATION,
        hand_back=mode,
    )


_NESTED = (
    "You are already working on a task in the background: do the work now with your "
    "tools and end with what you found. A task starts no task of its own."
)


def background_tool_server(kit: RoomKit, agent_id: str) -> Any:
    """What serves *agent_id*'s ``work_in_background`` calls, in the room of
    the call: ``(room_id, name, arguments)`` to the tool's answer.

    The task is always *agent_id*'s own: a model never names another agent's
    worker, which may hold more than it does (a workspace, its tools), nor
    has a result handed back or posted in another agent's name. It asks that
    agent by name instead; the host's ``start_task`` reaches any agent."""

    async def serve(room_id: str, _name: str, arguments: dict[str, Any]) -> str:
        room = current_tool_room()
        if room is not None and (room.metadata or {}).get("task_id"):
            return json.dumps({"error": _NESTED})
        task = str(arguments.get("task") or "").strip()
        try:
            started = await kit.start_task(room_id, agent_id, task)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        agent = identifier(agent_handle(room.metadata if room else None, agent_id), "agent")
        back = (
            f"its result is posted as {agent}'s message"
            if started.hand_back == "post"
            else f"its result comes back to {agent} as a turn of its own"
        )
        return json.dumps(
            {
                "task_id": started.id,
                "status": "started",
                "message": (
                    f"The work runs in the background and {back} when it is done. If "
                    "this turn has not said so yet, say in one short sentence that it is "
                    "under way; then end your turn, without saying it again."
                ),
            }
        )

    return serve
