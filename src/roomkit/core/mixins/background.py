"""BackgroundMixin — long work started as a task (RFC §23.4, §19.7.5 rule 19)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_registry import orchestration_tool
from roomkit.channels.declared import id_of, owner_tools_of
from roomkit.orchestration._call_room import in_call_room
from roomkit.providers.ai.base import AITool
from roomkit.tasks.background import (
    BACKGROUND_TOOL,
    BACKGROUND_TOOL_DEF,
    BackgroundSetup,
    background_tool_server,
    start_background_task,
)
from roomkit.tasks.status import TASK_STATUS_TOOL, CancelTaskTool, TaskStatusTool

if TYPE_CHECKING:
    from roomkit.channels.ai import AIChannel
    from roomkit.channels.declared import DeclaredChannel
    from roomkit.tasks.models import DelegatedTask
    from roomkit.turns._registry_tools import OwnerToolStore


class BackgroundMixin:
    """``setup_background`` and ``start_task``: an agent's long work runs as a
    task, off its turn."""

    _background_setups: dict[str, BackgroundSetup]
    _owner_tools: OwnerToolStore

    # Cross-mixin method — attribute annotation avoids MRO shadowing
    get_room: Any  # RoomLifecycleMixin

    def setup_background(
        self,
        agent: AIChannel | DeclaredChannel | str,
        *,
        worker: str | None = None,
        hand_back: str = "relay",
    ) -> None:
        """Give *agent* the ``work_in_background`` tool (RFC §23.4): it starts
        long work as a task and ends its turn, the work running apart.

        Args:
            agent: The AI channel given the tool, in every room it serves, or
                the id of one the kit holds, by its declaration alone included
                (RFC §10.6.2).
            worker: The channel that does its tasks (another agent, an
                external agent over ACP); the agent itself when None.
            hand_back: How a task's result reaches the room: ``"relay"``, the
                agent says it in a turn of its own, or ``"post"``, it is
                posted as it is, as the agent's message.

        Raises:
            ValueError: An unknown *hand_back*.
            RuntimeError: The tool is already set up on the agent.
            ChannelNotRegisteredError: An id names no channel of the kit.
        """
        setup = BackgroundSetup(worker=worker, hand_back=hand_back)
        registry = owner_tools_of(self, agent)  # ty: ignore[invalid-argument-type]
        channel_id = id_of(agent)
        if registry.lookup(BACKGROUND_TOOL, None) is not None:
            msg = f"setup_background() already called for channel '{channel_id}'"
            raise RuntimeError(msg)
        serve = background_tool_server(self, channel_id)  # ty: ignore[invalid-argument-type]
        registry.register(
            orchestration_tool(BACKGROUND_TOOL_DEF, in_call_room(BACKGROUND_TOOL, serve)),
            owner=self,
        )
        self._background_setups[channel_id] = setup

    def setup_task_tools(self, agent: AIChannel | DeclaredChannel | str) -> None:
        """Give *agent* ``task_status`` and ``cancel_task`` (RFC §23.4), served
        by the room's owner wherever the agent's turns run (RFC §10.6.5): the
        tasks of the room of the call, and a cancel of one of them, never
        another room's. ``TaskStatusTool(kit)`` and ``CancelTaskTool(kit)``
        are the same tools as an agent's own, for an agent run in this
        process.

        Raises:
            RuntimeError: The tools are already set up on the agent.
            ToolNameCollisionError: The agent already serves a tool of one of
                those names (its own ``CancelTaskTool``, say): neither is set up.
            ChannelNotRegisteredError: An id names no channel of the kit.

        *agent* may be the id of a channel the kit holds, by its declaration
        alone included (RFC §10.6.2).
        """
        registry = owner_tools_of(self, agent)  # ty: ignore[invalid-argument-type]
        channel_id = id_of(agent)
        if registry.lookup(TASK_STATUS_TOOL, None) is not None:
            msg = f"setup_task_tools() already called for channel '{channel_id}'"
            raise RuntimeError(msg)
        entries = []
        for tool in (TaskStatusTool(self), CancelTaskTool(self)):  # ty: ignore[invalid-argument-type]
            definition = AITool(**tool.definition)
            entries.append(orchestration_tool(definition, _served_by(tool, definition.name)))
        # Both or neither: an agent never keeps half of them.
        registry.register_all(entries, owner=self)

    async def start_task(
        self,
        room_id: str,
        agent_id: str,
        task: str,
        *,
        hand_back: str | None = None,
        organization_id: str | None = None,
    ) -> DelegatedTask:
        """Start *task* for *agent_id*, an agent of *room_id*, as its
        ``work_in_background`` call would (RFC §23.4): on its worker, handed
        the room's last messages, its result relayed or posted back to it.
        The host's door, and a person's through what the host gives them; an
        agent with no background set up does the work itself and has its
        result relayed.

        Args:
            room_id: The room the task is for.
            agent_id: The agent the task is for: an intelligence channel
                attached to the room, an external agent included.
            task: What to do and report.
            hand_back: ``"relay"`` or ``"post"``; the agent's setup's when
                ``None`` or an empty string.
            organization_id: The organization the caller acts for (RFC
                §17.2): the room is read with it before anything is started.

        Raises:
            RoomNotFoundError: *organization_id* is set and the room is
                missing or another organization's; nothing is started.
            ValueError: *agent_id* is no agent of the room, *task* is blank,
                or *hand_back* is unknown.
        """
        if organization_id is not None:
            await self.get_room(room_id, organization_id=organization_id)
        setup = self._background_setups.get(agent_id) or BackgroundSetup()
        return await start_background_task(
            self,  # ty: ignore[invalid-argument-type]
            room_id,
            agent_id,
            task,
            setup=setup,
            hand_back=hand_back,
        )


def _served_by(tool: TaskStatusTool | CancelTaskTool, name: str) -> Any:
    """What serves calls to *name* with *tool*'s handler, which reads the room
    of the call itself."""

    async def serve(arguments: dict[str, Any]) -> str:
        return await tool.handler(name, arguments)

    return serve
