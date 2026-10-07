"""A conversation pipeline driving a speech-to-speech session (RFC §19.5).

On a realtime channel the active agent is the session's configuration: its
prompt (with its identity block), its voice and its tools, the handoff tool
among them. The active agent is the room's: a session starts with its room's,
and a handoff reconfigures that room's sessions to the next agent's, leaving
the channel and the other rooms as they are. A call to the handoff tool hands
the room off; a call to one of the active agent's own tools is served by the
handler the agent was given.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Collection
from typing import TYPE_CHECKING, Any

from roomkit.channels._agent_features import UnservedFeature, unserved_on_realtime
from roomkit.channels._tool_registry import SessionConfig, orchestration_tool
from roomkit.core.exceptions import UnservedToolCallError
from roomkit.orchestration.handoff import (
    HANDOFF_TOOL,
    HANDOFF_TOOL_NAME,
    HandoffHandler,
    build_handoff_tool,
)
from roomkit.orchestration.state import get_conversation_state
from roomkit.tools.context import _current_turn_chain_depth, current_tool_room_id
from roomkit.tools.result import declined_answer

if TYPE_CHECKING:
    from roomkit.channels.agent import Agent
    from roomkit.channels.ai import ToolResult
    from roomkit.channels.realtime_voice import RealtimeVoiceChannel
    from roomkit.core.framework import RoomKit
    from roomkit.models.room import Room
    from roomkit.orchestration.pipeline import PipelineStage
    from roomkit.providers.ai.base import AITool

_DEFAULT_GREETING = (
    "Handoff complete. You are now the active agent. "
    "Please introduce yourself briefly to the caller."
)


logger = logging.getLogger("roomkit.orchestration.pipeline")


class RealtimePipeline:
    """A pipeline's agents on one realtime channel: each agent's session
    configuration, the calls its sessions make, and its handoffs."""

    def __init__(
        self,
        kit: RoomKit,
        rtv: RealtimeVoiceChannel,
        agents: list[Agent],
        stages: list[PipelineStage],
        handler: HandoffHandler,
        default_agent_id: str,
        *,
        greet_on_handoff: bool,
        greeting_prompt: str | None,
    ) -> None:
        self._kit = kit
        self._rtv = rtv
        self._handler = handler
        self._agent_map: dict[str, Agent] = {a.channel_id: a for a in agents}
        self._default_agent_id = default_agent_id
        self._greet_on_handoff = greet_on_handoff
        self._greeting_prompt = greeting_prompt
        self.agent_configs: dict[str, dict[str, Any]] = {
            agent.channel_id: self._agent_config(agent, stages) for agent in agents
        }
        # The agents' tools the pipeline serves, fixed at the install.
        self._served: frozenset[str] = frozenset()

    def install(self) -> None:
        """Serve the handoff and the agents' own tools on the channel, and start
        each new session with its room's active agent."""
        registry = self._rtv._registry
        # Declared by each agent's configuration, with its own targets: never
        # hidden by Tool Search, served by the handoff handler.
        handoff = orchestration_tool(
            HANDOFF_TOOL,
            self.serve_handoff,
            declared_as=[config["handoff"] for config in self.agent_configs.values()],
            always_declared=False,
        )
        served = self._agent_tools()
        self._served = frozenset(served)
        # Two agents may give one name two definitions: each agent's session
        # declares its own, and every one of them is the entry's.
        agent_tools = [
            orchestration_tool(
                first,
                functools.partial(self.serve_agent_tool, name),
                declared_as=others,
                always_declared=False,
                deferrable=True,
            )
            for name, (first, *others) in served.items()
        ]
        registry.register_all([handoff, *agent_tools], owner=self)
        registry.set_session_source(self.session_config, owner=self)

    def _agent_tools(self) -> dict[str, list[AITool]]:
        """The agents' own tools the pipeline serves, each name with every
        agent's definition of it: a name the channel carries is the
        channel's, the agent's tool under it neither declared nor served
        (RFC §19.5, §21.1)."""
        carried = _carried_tool_names(self._rtv)
        tools: dict[str, list[AITool]] = {}
        for agent in self._agent_map.values():
            for tool in agent._user_tools:
                if tool.name in carried:
                    logger.warning(
                        "Agent %s's tool %r shares its name with a tool of channel %s: "
                        "the name is the channel's, the agent's is neither declared nor served",
                        agent.channel_id,
                        tool.name,
                        self._rtv.channel_id,
                    )
                elif tool.name != HANDOFF_TOOL_NAME:
                    tools.setdefault(tool.name, []).append(tool)
        return tools

    async def session_config(self, room_id: str) -> SessionConfig | None:
        """What a new session of *room_id* starts with: its active agent's."""
        room = await self._kit.get_room(room_id)
        state = get_conversation_state(room)
        agent_id = state.active_agent_id or self._default_agent_id
        config = self.agent_configs.get(agent_id)
        if config is None:
            return None
        return SessionConfig(
            system_prompt=self._prompt_for(agent_id, room),
            voice=config["voice"],
            tools=self._session_tools(agent_id),
            tool_policy=self._agent_map[agent_id]._tool_policy,
        )

    def _session_tools(self, agent_id: str) -> list[dict[str, Any]]:
        """The tools *agent_id*'s sessions declare, read from the channel's
        tools as they are now: a channel configured after the install keeps
        its tools under every agent (RFC §19.5)."""
        return _agent_session_tools(
            self._rtv,
            self._agent_map[agent_id],
            self.agent_configs[agent_id]["handoff"],
            self._served,
        )

    def _prompt_for(self, agent_id: str, room: Room) -> str | None:
        """*agent_id*'s prompt in *room*, its identity in the room's language."""
        prompt = self.agent_configs[agent_id]["system_prompt"]
        lang = self._handler.get_room_language(room, agent_id)
        agent = self._agent_map.get(agent_id)
        if lang and agent is not None:
            base = getattr(agent, "system_prompt", None) or ""
            identity = agent.build_identity_block(language=lang)
            prompt = (base + identity) if identity else prompt
        return prompt

    def _agent_config(self, agent: Agent, stages: list[PipelineStage]) -> dict[str, Any]:
        """The prompt, voice and handoff tool *agent*'s sessions run with."""
        prompt = agent.system_prompt or ""
        identity = agent.build_identity_block()
        if identity:
            prompt = prompt + identity
        return {
            "system_prompt": prompt or None,
            "voice": agent.voice,
            "handoff": self._handoff_tool(agent, stages),
        }

    def _handoff_tool(self, agent: Agent, stages: list[PipelineStage]) -> AITool:
        """The handoff tool *agent* declares, its targets the stages it reaches."""
        stage = next((s for s in stages if s.agent_id == agent.channel_id), None)
        if stage is None:
            return build_handoff_tool([])
        reachable: set[str] = set()
        if stage.next:
            reachable.add(stage.next)
        reachable.update(stage.can_return_to)
        targets: list[tuple[str, str | None]] = []
        for s in stages:
            if s.phase in reachable and s.agent_id != agent.channel_id:
                ta = self._agent_map.get(s.agent_id)
                desc = ta.description if ta else None
                if desc is None:
                    desc = s.description
                targets.append((s.agent_id, desc))
        return build_handoff_tool(targets)

    def greeting(self, agent_id: str, language: str | None = None) -> str:
        """What *agent_id* is told to say when a handoff makes it active."""
        if self._greeting_prompt:
            msg = self._greeting_prompt
        else:
            target = self._agent_map.get(agent_id)
            role = target.role if target else None
            if role:
                msg = (
                    f"Handoff complete. You are now the {role}. "
                    f"Your previous identity in this conversation no longer "
                    f"applies — introduce yourself in your new role."
                )
            else:
                msg = _DEFAULT_GREETING
        lang = language
        if not lang and agent_id in self._agent_map:
            lang = getattr(self._agent_map[agent_id], "language", None)
        if lang:
            msg = f"[Respond in {lang}] {msg}"
        return msg

    async def serve_handoff(self, arguments: dict[str, Any]) -> ToolResult:
        """Hand the room of the call off to the agent *arguments* name."""
        # Lazy import to avoid circular dependency
        from roomkit.channels.realtime_voice import get_current_voice_session

        kit = self._kit
        session = get_current_voice_session()
        session_id = session.id if session else None
        room_id = self._rtv.session_rooms.get(session_id) if session_id else None
        if not room_id:
            return json.dumps({"error": "No room context for this session"})

        room = await kit.get_room(room_id)
        state = get_conversation_state(room)
        calling_agent = state.active_agent_id or self._default_agent_id

        result = await self._handler.handle(
            room_id=room_id,
            calling_agent_id=calling_agent,
            arguments=arguments,
        )

        output = result.model_dump()
        if result.accepted and self._greet_on_handoff:
            target = arguments.get("target", "")
            # Re-read room for current language
            room = await kit.get_room(room_id)
            lang = self._handler.get_room_language(room, target)
            output["message"] = self.greeting(target, language=lang)
        return json.dumps(output)

    async def serve_agent_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Serve a call to an agent's tool (RFC §19.5): the active agent's own
        by the handler the agent was given, and one the agent has no handler
        for, or does not declare, by the channel's."""
        channel_handler = self._rtv._tool_handler
        agent = await self._active_agent()
        agent_handler = agent._user_tool_handler if agent is not None else None
        agent_tools = agent._user_tools if agent is not None else []
        # Each answer read as every channel reads a handler's: the "not mine"
        # envelope is a call nothing served, not a result (RFC §21.4).
        if agent_handler is not None and any(t.name == name for t in agent_tools):
            return declined_answer(await agent_handler(name, arguments), name)
        if channel_handler is not None:
            return declined_answer(await channel_handler(name, arguments), name)
        raise UnservedToolCallError(f"tool {name!r} is not served here")

    async def _active_agent(self) -> Agent | None:
        """The agent the call's room is talking to, by its conversation state."""
        room_id = current_tool_room_id()
        if room_id is None:
            return None
        state = get_conversation_state(await self._kit.get_room(room_id))
        return self._agent_map.get(state.active_agent_id or self._default_agent_id)

    async def on_handoff_complete(self, room_id: str, result: Any) -> None:
        """Reconfigure *room_id*'s sessions to the agent a handoff made active."""
        new_id = result.new_agent_id
        if not new_id or new_id not in self.agent_configs:
            return
        config = self.agent_configs[new_id]
        rtv = self._rtv
        room = await self._kit.get_room(room_id)
        prompt = self._prompt_for(new_id, room)
        lang = self._handler.get_room_language(room, new_id)
        tools = self._session_tools(new_id)
        policy = self._agent_map[new_id]._tool_policy
        sessions = rtv.get_room_sessions(room_id)

        # The new agent's policy holds on every session of the room before any
        # declares its tools, each read for its participant's role.
        for session in sessions:
            await rtv._use_agent_policy(session, policy)
        for session in sessions:
            await rtv.reconfigure_session(
                session,
                system_prompt=prompt,
                voice=config["voice"],
                tools=tools,
            )

            if self._greet_on_handoff:
                # Session resumption doesn't preserve pending function-
                # call state, so the tool result alone won't trigger a
                # response.  Inject a language-aware instruction to give
                # the new agent a turn to speak in its new role. It
                # directs the model, so it carries the system intent: a
                # full-duplex provider voices a user injection instead
                # of following it (RFC §12.4).
                msg = self.greeting(new_id, language=lang)
                # At the handing-off call's depth: the new agent's greeting
                # continues that chain, it does not open one (RFC §8.3).
                await rtv.inject_text(
                    session, msg, role="system", chain_depth=_current_turn_chain_depth()
                )


def _agent_session_tools(
    rtv: RealtimeVoiceChannel, agent: Agent, handoff: AITool, served: Collection[str]
) -> list[dict[str, Any]]:
    """The tools an agent's realtime session declares (RFC §19.5).

    The channel's own tools as they are now, which stay declared under every
    agent, the agent's that the pipeline serves (*served*), then the handoff
    tool. One schema, one server (RFC §21.1): an agent tool that a channel
    tool shadowed at the install is served by nothing of the agent's, so it
    is not declared, even once the channel dropped its own.
    """
    host = [dict(t) for t in rtv._tools or []]
    own = {t.name: t.model_dump() for t in agent._user_tools if t.name in served}
    return [*host, *own.values(), handoff.model_dump()]


def _carried_tool_names(rtv: RealtimeVoiceChannel) -> set[str]:
    """Every name the channel carries now, declared in a session or not: its
    host definitions, every name its human-input tools serve, the tools it
    serves itself (Tool Search's, the skills') and those its reasoning
    backend answers itself."""
    return (
        set(rtv._host_tool_names())
        | rtv._human_input.names
        | rtv._channel_tool_names()
        | rtv._backend_served_names()
    )


def refuse_agents_with_unserved(agents: list[Agent], channel_id: str) -> None:
    """Refuse an agent that carries what a realtime session never serves for
    it (skills, a human-input handler, planning, a sandbox, an external tool
    handler), each cause named: its gated tools would run without their skill,
    and its other ones would be called by a model never told of them (RFC
    §19.5). Its own host tools stay served."""
    causes = [
        _unserved_cause(agent.channel_id, feature, channel_id)
        for agent in agents
        for feature in unserved_on_realtime(agent)
    ]
    if causes:
        raise ValueError(" ".join(causes))


def _unserved_cause(agent_id: str, feature: UnservedFeature, channel_id: str) -> str:
    """Why *agent_id* is refused for *feature*, and what serves it instead."""
    where = (
        f"RealtimeVoiceChannel({channel_id!r}, {feature.instead}...) serves "
        f"{feature.short} in its sessions"
        if feature.instead is not None
        else f"a realtime session never serves an agent's {feature.short}"
    )
    return (
        f"Agent {agent_id!r} carries {feature.what}, which a realtime pipeline on channel "
        f"{channel_id!r} does not serve for it (RFC §19.5): {where}."
    )
