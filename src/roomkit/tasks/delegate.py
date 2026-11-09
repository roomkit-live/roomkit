"""Tool integration for agent delegation.

Mirrors the handoff pattern in ``orchestration/handoff.py``:
tool definition, handler class, and ``setup_delegation()`` wiring.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator, Mapping
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import ToolEntry, orchestration_tool
from roomkit.channels.declared import id_of, owner_tools_of
from roomkit.core.exceptions import ChannelNotRegisteredError
from roomkit.models.enums import ChannelCategory
from roomkit.orchestration._call_room import RoomToolServe, in_call_room
from roomkit.providers.ai.base import AITool
from roomkit.tasks.brief import context_block, previous_tasks_block, task_brief
from roomkit.tasks.cache import CompletedTaskCache
from roomkit.tasks.handback import MAX_RESULT_CHARS, check_max_result_chars
from roomkit.tasks.models import DelegatedTask, TaskStatus

if TYPE_CHECKING:
    from roomkit.channels.ai import AIChannel
    from roomkit.channels.declared import DeclaredChannel
    from roomkit.channels.realtime_voice import RealtimeVoiceChannel
    from roomkit.core.delivery import DeliveryStrategy
    from roomkit.core.framework import RoomKit

logger = logging.getLogger("roomkit.tasks")


# -- Tool definition ----------------------------------------------------------

DELEGATE_TOOL_NAME = "delegate_task"

CONVERSATION_ARGUMENT = "conversation"
"""The argument a handler that allows it adds to the tool."""

DELEGATE_TOOL = AITool(
    name=DELEGATE_TOOL_NAME,
    description=(
        "Delegate a task to a background agent. The agent works in its own "
        "room and you can continue the current conversation. Use when: "
        "a task needs a specialist, the user wants something done in the "
        "background, or you need to run work in parallel."
    ),
    parameters={
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "description": "Target agent ID for the delegated task",
            },
            "task": {
                "type": "string",
                "description": "Clear description of what the agent should do",
            },
            "context": {
                "type": "object",
                "description": (
                    "Optional context to pass to the agent (e.g. a branch, a repo URL, what a "
                    "tool returned): it reads it with the task, as data"
                ),
            },
            "share_channels": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Channel IDs from the current room to share with the background agent"
                ),
            },
        },
        "required": ["agent", "task"],
    },
)


def build_delegate_tool(targets: list[tuple[str, str | None]]) -> AITool:
    """Build a delegate tool with constrained agent enum.

    Args:
        targets: List of ``(agent_id, description_or_none)`` pairs.
            When empty, returns the generic :data:`DELEGATE_TOOL`.

    Returns:
        An :class:`AITool` whose ``agent`` parameter has an ``enum``
        restricting the AI to only the listed agent IDs: the tool
        :func:`setup_delegation` and :func:`setup_realtime_delegation` serve from
        it refuses a call naming another.
    """
    if not targets:
        return DELEGATE_TOOL

    target_ids = [t[0] for t in targets]
    target_lines = []
    for agent_id, desc in targets:
        if desc:
            target_lines.append(f"  - {agent_id}: {desc}")
        else:
            target_lines.append(f"  - {agent_id}")

    description = "Delegate a task to a background agent.\nAvailable agents:\n" + "\n".join(
        target_lines
    )

    params = {
        "type": "object",
        "properties": {
            **DELEGATE_TOOL.parameters["properties"],
            "agent": {
                "type": "string",
                "enum": target_ids,
                "description": "Target agent ID for the delegated task",
            },
        },
        "required": DELEGATE_TOOL.parameters["required"],
    }

    return AITool(
        name="delegate_task",
        description=description,
        parameters=params,
    )


def _served_tool(tool: AITool, max_conversation: int) -> AITool:
    """*tool* as a handler allowing *max_conversation* messages serves it: with
    a ``conversation`` argument up to that bound, without one when it is 0,
    whatever *tool* declares."""
    declared = tool.parameters.get("properties", {})
    if max_conversation == 0 and CONVERSATION_ARGUMENT not in declared:
        return tool
    properties = {name: spec for name, spec in declared.items() if name != CONVERSATION_ARGUMENT}
    if max_conversation > 0:
        properties[CONVERSATION_ARGUMENT] = {
            "type": "integer",
            "minimum": 0,
            "maximum": max_conversation,
            "description": (
                "How many of the room's last messages to hand to the agent, quoted as "
                f"data: at most {max_conversation}. Leave it out to hand over none."
            ),
        }
    parameters = {**tool.parameters, "properties": properties}
    if "required" in parameters:
        parameters["required"] = [
            name for name in parameters["required"] if name != CONVERSATION_ARGUMENT
        ]
    return tool.model_copy(update={"parameters": parameters})


# -- DelegateHandler ----------------------------------------------------------


def _passed_context(passed: Any) -> dict[str, Any]:
    """What a call passed as ``context``, as the object a task's metadata and
    its brief read: a model may send a string where the tool says object, kept
    under ``context``. Null, an empty string and an empty list are no context; 0
    and false are values."""
    if isinstance(passed, Mapping):
        return dict(passed)
    return {} if passed is None or passed in ("", []) else {"context": passed}


def _named_channels(named: Any, default: list[str]) -> list[str]:
    """The channels a call shares: *default* when it names none or an empty
    list, else the texts of its list.

    Raises:
        ValueError: A value that is no list.
    """
    if named is None or named == []:
        return default
    if not isinstance(named, list):
        raise ValueError("share_channels must be a list of channel ids")
    return [channel for channel in named if isinstance(channel, str)]


def _asked_conversation(asked: Any, maximum: int) -> int:
    """How many messages a call hands over: what it asks, an integer as JSON
    Schema reads one (``1.0`` is, ``"1"``, ``1.5`` and ``true`` are not), within
    *maximum* and never below 0. Anything else asks for none."""
    if isinstance(asked, bool) or not isinstance(asked, int | float):
        return 0
    if isinstance(asked, float) and not asked.is_integer():
        return 0
    return max(0, min(int(asked), maximum))


class DelegateHandler:
    """Processes ``delegate_task`` tool calls by calling ``kit.delegate()``.

    Supports optional dedup via :class:`CompletedTaskCache`, a repeat being the
    same agent's call of the same task with the same ``context`` and no
    ``conversation``, and per-room concurrency control via asyncio locks.

    What the call passes as ``context``, and the calling agent's previous tasks
    when a cache holds any and no intelligence channel is shared, reach the worker
    quoted as data, in its brief; its ``task`` stays as written, for the parent
    room and the cache key (RFC §23.4).

    Args:
        kit: The RoomKit instance.
        notify: Channel ID to notify on completion (defaults to calling agent).
        default_share_channels: Channels shared with background agents by default.
        delivery_strategy: How to deliver results back proactively.
        cache: Optional completed-task cache for dedup.
        serialize_per_room: If True, only one delegation per room at a time.
        max_conversation: How many of the room's last messages a call may hand
            to its worker, those both the calling agent and the notified channel
            may read, quoted as data; none when an intelligence channel is
            shared into the child room, which reads them too. The tool gains a
            ``conversation`` argument when it is above 0, and a call is held to
            it whatever it asks. 0, the default: none, and no such argument: the
            worker may be another vendor's, and the model chooses it, so the
            host opens this.
        max_result_chars: How much of the worker's output its result carries
            back to the agent: past it, the output is cut and says so (RFC
            §23.3 step 8).

    Raises:
        ValueError: A *max_conversation* that is no count: negative or no
            integer. A *max_result_chars* that is no count: below 1 or no
            integer.
    """

    def __init__(
        self,
        kit: RoomKit,
        *,
        notify: str | None = None,
        default_share_channels: list[str] | None = None,
        delivery_strategy: DeliveryStrategy | None = None,
        cache: CompletedTaskCache | None = None,
        serialize_per_room: bool = False,
        max_conversation: int = 0,
        max_result_chars: int = MAX_RESULT_CHARS,
    ) -> None:
        if type(max_conversation) is not int or max_conversation < 0:
            raise ValueError("max_conversation must be an integer, 0 or more")
        check_max_result_chars(max_result_chars)
        self._kit = kit
        self._notify = notify
        self._default_share_channels = default_share_channels or []
        # Set the delivery strategy on the framework if provided
        if delivery_strategy is not None:
            kit._delivery_strategy = delivery_strategy
        self._cache = cache
        self._serialize_per_room = serialize_per_room
        self._max_conversation = max_conversation
        self._max_result_chars = max_result_chars
        # A room's lock, and how many delegations hold it or wait for it: the
        # last one out drops it, so a room no longer delegating keeps nothing.
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._room_lock_users: dict[str, int] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()

    @contextlib.asynccontextmanager
    async def _room_turn(self, room_id: str) -> AsyncGenerator[None]:
        lock = self._room_locks.setdefault(room_id, asyncio.Lock())
        self._room_lock_users[room_id] = self._room_lock_users.get(room_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._room_lock_users[room_id] -= 1
            if not self._room_lock_users[room_id]:
                del self._room_lock_users[room_id]
                del self._room_locks[room_id]

    async def handle(
        self,
        room_id: str,
        calling_agent_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a delegate_task tool call.

        Raises:
            ValueError: A ``task`` that is no text, or a ``share_channels``
                that is no list: nothing is made, and the model reads it as
                its call's error.
            ChannelNotRegisteredError: An ``agent`` the kit does not hold, or
                that is no text.
        """
        agent_id = arguments.get("agent", "")
        task_text = arguments.get("task", "")
        if not isinstance(task_text, str):
            raise ValueError("task must be a string")
        if not isinstance(agent_id, str):
            # Only a text names one of the kit's channels.
            raise ChannelNotRegisteredError(f"Agent channel '{agent_id}' not registered")
        share_channels = _named_channels(
            arguments.get("share_channels"), self._default_share_channels
        )
        passed = _passed_context(arguments.get("context"))
        count = _asked_conversation(arguments.get("conversation"), self._max_conversation)

        # Gap 13: Check cache for recently completed matching task. A call that
        # hands the room's last messages over reads them as they stand now.
        if self._cache is not None and count == 0:
            cached = self._cache._answer_for(
                room_id, agent_id, task_text, passed, calling_agent_id
            )
            if cached is not None:
                logger.info(
                    "Returning cached result for task '%s' -> agent '%s'",
                    task_text[:60],
                    agent_id,
                )
                return {**cached, "from_cache": True}

        # Gap 14: Serialize delegations per room
        if self._serialize_per_room:
            async with self._room_turn(room_id):
                return await self._do_delegate(
                    room_id, calling_agent_id, agent_id, task_text, passed, count, share_channels
                )

        return await self._do_delegate(
            room_id, calling_agent_id, agent_id, task_text, passed, count, share_channels
        )

    async def _shares_an_agent(self, room_id: str, share_channels: list[str]) -> bool:
        """Whether one of *share_channels* is bound in the room as an intelligence
        channel: one it does not bind is not shared (RFC §23.3 step 3)."""
        for channel_id in share_channels:
            binding = await self._kit._store.get_binding(room_id, channel_id)
            if binding is not None and binding.category == ChannelCategory.INTELLIGENCE:
                return True
        return False

    async def _do_delegate(
        self,
        room_id: str,
        calling_agent_id: str,
        agent_id: str,
        task_text: str,
        passed: dict[str, Any],
        count: int,
        share_channels: list[str],
    ) -> dict[str, Any]:
        notified = self._notify or calling_agent_id
        recent: list[str] = []
        conversation: str | None = None
        # An intelligence channel shared into the child room reads the worker's
        # description: nothing the framework rebuilds goes there, as it may hold
        # what that channel may not read (RFC §7.5 rule 8). What this call passed
        # stays: the model wrote it, as it wrote the task that channel reads too.
        if (count or self._cache is not None) and not await self._shares_an_agent(
            room_id, share_channels
        ):
            # Gap 15: Inject previous task context from cache: the agent's own,
            # which it may read, not another's, which may hold what it may not.
            if self._cache is not None:
                recent = self._cache._recent_of(room_id, calling_agent_id, limit=3)
            # What both the agent that chose to hand it over and the channel its
            # result goes back to may read.
            if count:
                conversation = await self._kit._room_conversation(
                    room_id, list(dict.fromkeys((calling_agent_id, notified))), count
                )
        task = await self._kit.delegate(
            room_id=room_id,
            agent_id=agent_id,
            task=task_text,
            context={**passed, "previous_tasks": recent} if recent else passed,
            share_channels=share_channels,
            notify=notified,
            brief=task_brief(
                task_text, context_block(passed), previous_tasks_block(recent), conversation
            ),
            max_result_chars=self._max_result_chars,
        )

        result = {
            "status": "delegated",
            "task_id": task.id,
            "child_room_id": task.child_room_id,
            "agent_id": agent_id,
        }

        # Store in cache when task completes (fire-and-forget)
        if self._cache is not None:
            t = asyncio.create_task(
                self._remember_when_completed(
                    self._cache, task, result, task_text, passed, count, calling_agent_id
                )
            )
            self._background_tasks.add(t)
            t.add_done_callback(self._background_tasks.discard)

        return result

    @staticmethod
    async def _remember_when_completed(
        cache: CompletedTaskCache,
        task: DelegatedTask,
        receipt: dict[str, Any],
        task_text: str,
        passed: dict[str, Any],
        count: int,
        caller: str,
    ) -> None:
        """Cache what the task returned once it completes, so a repeat of its
        call gets the result, as ``task_status`` gives it, not the receipt of
        its start. Only a completed task answers a repeat: a failed or
        cancelled one is run again, as a strategy's dispatch is. One handed the
        room's last messages (*count*) answers none: what it returned holds for
        the room as it stood."""
        try:
            ended = await task.wait(timeout=600.0)
        except Exception:
            return  # timeout or error — don't cache
        if ended.status != TaskStatus.COMPLETED:
            return
        answer = {**receipt, "status": str(ended.status), "result": ended.output or ""}
        cache._keep(
            task.parent_room_id,
            receipt["agent_id"],
            task_text,
            answer,
            passed,
            caller,
            answers=not count,
        )


# -- Wiring -------------------------------------------------------------------


def _served_entry(handler: DelegateHandler, calling: str, tool: AITool | None) -> ToolEntry:
    """The registry entry that serves *tool*, the generic one when none, for the
    agent *calling* through *handler*."""
    served = _served_tool(tool or DELEGATE_TOOL, handler._max_conversation)
    return orchestration_tool(
        served,
        in_call_room(DELEGATE_TOOL_NAME, _served_delegation(handler, calling, served)),
        waits=True,
    )


def _served_delegation(handler: DelegateHandler, calling: str, tool: AITool) -> RoomToolServe:
    """What serves a call of *tool*: refused when it names a worker outside the
    list the tool declares, since a list a model can ignore holds nothing (RFC
    §23.4); a tool with no list takes any.

    Raises:
        ValueError: A tool whose list of agents is no list.
    """
    listed = tool.parameters.get("properties", {}).get("agent", {}).get("enum")
    if listed is not None and not isinstance(listed, list):
        raise ValueError("the delegate tool's agent enum must be a list")

    async def delegate(room_id: str, name: str, arguments: dict[str, Any]) -> str:
        if listed is not None and arguments.get("agent") not in listed:
            return json.dumps({"error": f"This tool delegates only to: {', '.join(listed)}"})
        result = await handler.handle(
            room_id=room_id,
            calling_agent_id=calling,
            arguments=arguments,
        )
        return json.dumps(result)

    return delegate


def setup_delegation(
    channel: AIChannel | DeclaredChannel | str,
    handler: DelegateHandler,
    *,
    tool: AITool | None = None,
) -> None:
    """Serve the delegate tool on an AIChannel.

    Declares ``delegate_task`` in the channel's turns and serves its calls with
    *handler*. Same pattern as ``setup_handoff()``: the task is delegated from
    the room of the call, read from the tool call context (RFC §23.4), so one
    agent serving several rooms delegates from each.

    Args:
        channel: The AI channel to wire delegation into, or the id of one the
            kit holds, by its declaration alone included (RFC §10.6.2).
        handler: The delegate handler that processes tool calls.
        tool: Optional custom delegate tool (e.g. from :func:`build_delegate_tool`).

    Raises:
        RuntimeError: Delegation is already set up on the channel.
        ChannelNotRegisteredError: An id names no channel of the kit.
    """
    calling = id_of(channel)
    entry = _served_entry(handler, calling, tool)
    registry = owner_tools_of(handler._kit, channel)
    if registry.lookup(DELEGATE_TOOL_NAME, None) is not None:
        msg = f"setup_delegation() already called for channel '{calling}'"
        raise RuntimeError(msg)

    registry.register(entry, owner=handler)


def setup_realtime_delegation(
    channel: RealtimeVoiceChannel,
    handler: DelegateHandler,
    *,
    tool: AITool | None = None,
) -> None:
    """Serve the delegate tool on a RealtimeVoiceChannel.

    Declares ``delegate_task`` in the channel's sessions and serves its calls
    with *handler*, delegating from the room of the call: the session's room,
    which the channel installs as the tool call context (RFC §21.4, §23.4).

    Args:
        channel: The realtime voice channel to wire delegation into.
        handler: The delegate handler that processes tool calls.
        tool: Optional custom delegate tool (e.g. from :func:`build_delegate_tool`).

    Raises:
        RuntimeError: Delegation is already set up on the channel.
    """
    if channel._registry.lookup(DELEGATE_TOOL_NAME, None) is not None:
        msg = f"setup_realtime_delegation() already called for channel '{channel.channel_id}'"
        raise RuntimeError(msg)

    entry = _served_entry(handler, channel.channel_id, tool)
    channel._registry.register(entry, owner=handler)
