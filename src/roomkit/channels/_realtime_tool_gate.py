"""The pre-execution gate of a RealtimeVoiceChannel's tool calls (RFC §12.4).

What a session declares (its catalogue, what orchestration adds for its room,
the channel's own tools, each name once), what its tool policy admits for its
participant's role, and the gate a call passes before anything serves it, in
RFC §12.4's order: the declared tool, the policy, skill gating, its schema (a
flattened hub call folded back first), BEFORE_TOOL_USE, whose arguments are
validated again. A refused tool never has its arguments read.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Container
from typing import TYPE_CHECKING, Any

from roomkit.channels._ai_policy import policy_admits
from roomkit.channels._served_tools import CollisionLog, declared_once, dict_tool_name
from roomkit.channels._tool_registry import ChannelRegistry, ToolSource, tool_dict
from roomkit.models.enums import ChannelType
from roomkit.models.tool_call import ToolCallEvent
from roomkit.tools.policy import policy_refusal
from roomkit.tools.result import (
    GateRefusal,
    gated_tool_refusal,
    pre_execution_denial,
    unknown_tool_error,
)
from roomkit.tools.validation import (
    fold_hoisted_arguments,
    rewritten_arguments_error,
    validate_tool_arguments,
)

if TYPE_CHECKING:
    from roomkit.core.framework import RoomKit
    from roomkit.models.context import RoomContext
    from roomkit.tools._human_input_channel import ChannelHumanInput
    from roomkit.tools.policy import ToolPolicy
    from roomkit.voice.base import VoiceSession
    from roomkit.voice.realtime.provider import RealtimeVoiceProvider

logger = logging.getLogger("roomkit.channels.realtime_voice")


class RealtimeToolGateMixin:
    """The catalogue, the policy and the pre-execution gate of a realtime session."""

    _state_lock: threading.Lock
    _session_rooms: dict[str, str]
    _session_tools: dict[str, Any]
    _tools: Any
    _skill_support: Any
    _tool_policy: ToolPolicy | None
    _session_agent_policies: dict[str, ToolPolicy]
    _session_roles: dict[str, str | None]
    _sessions: dict[str, VoiceSession]
    _room_session_config: Any  # RealtimeVoiceChannel — cross-mixin
    _collisions: CollisionLog
    _human_input: ChannelHumanInput
    _registry: ChannelRegistry
    _tool_search_support: Any
    _provider: RealtimeVoiceProvider
    _framework: RoomKit | None
    channel_id: str

    def _session_base_tools(self, session_id: str) -> list[dict[str, Any]]:
        """The session's authorized catalogue, read under the state lock."""
        with self._state_lock:
            return self._session_tools.get(session_id, self._tools or [])

    def _session_catalogue(self, session_id: str) -> list[dict[str, Any]]:
        """Every tool the session declares beside the channel's own: its base
        catalogue, what orchestration set up for its room (RFC §19.7), then
        the person's tools (RFC §9.3). What a call is checked, validated and
        recovered against, on every door."""
        with self._state_lock:
            base = list(self._session_tools.get(session_id, self._tools or []))
            room_id = self._session_rooms.get(session_id)
        orchestration = self._orchestration_dicts(room_id, {dict_tool_name(t) for t in base})
        return base + orchestration + self._human_input_dicts()

    def _human_input_dicts(self) -> list[dict[str, Any]]:
        """The declarations of the person's tools, served by the channel on
        every door and in every session (RFC §9.3)."""
        return [tool_dict(tool) for tool in self._human_input.definitions]

    def _human_input_names(self) -> frozenset[str]:
        """The names the person's tools declare, which no other tool takes."""
        return self._human_input.declared_names

    def _orchestration_dicts(
        self, room_id: str | None, skip: Container[str | None] = ()
    ) -> list[dict[str, Any]]:
        """The declarations of the tools orchestration declares in *room_id*'s
        sessions, bar the names in *skip*."""
        return [
            tool_dict(entry.definition)
            for entry in self._registry.entries(room_id, source=ToolSource.ORCHESTRATION)
            if entry.traits.always_declared and entry.name not in skip
        ]

    def _tool_parameters(self, name: str, session: VoiceSession) -> dict[str, Any] | None:
        """Return the declared ``parameters`` schema for realtime tool *name*.

        ``None`` when the tool's schema is unknown (skips argument validation).
        """
        if self._tool_search_support and self._tool_search_support.is_search_tool(name):
            for tool in self._tool_search_support.search_tool_dicts():
                if tool["name"] == name:
                    params = tool.get("parameters")
                    return params if isinstance(params, dict) else None
        if self._skill_support and self._skill_support.is_skill_tool(name):
            for tool in self._skill_support.skill_tool_dicts():
                if tool["name"] == name:
                    params = tool.get("parameters")
                    return params if isinstance(params, dict) else None
        for t in self._session_catalogue(session.id):
            if isinstance(t, dict) and t.get("name") == name:
                params = t.get("parameters")
                return params if isinstance(params, dict) else None
        # A tool orchestration serves is checked against its server's schema
        # though the session declares none of its own, as on a text turn
        # (RFC §21.1): the catalogue holds no other tool under its name.
        entry = self._registry.lookup(name, self._session_room_id(session.id))
        if entry is not None and entry.source is ToolSource.ORCHESTRATION:
            return entry.definition.parameters
        return None

    def _session_room_id(self, session_id: str) -> str | None:
        """The room *session_id* belongs to, read under the state lock."""
        with self._state_lock:
            return self._session_rooms.get(session_id)

    def _is_declared_realtime_tool(
        self, name: str, session: VoiceSession, served: Container[str] | None = None
    ) -> bool:
        """Return whether *name* is in a non-empty session tool catalogue.

        An empty catalogue retains the historical hook-only/dynamic-handler
        mode. Once declarations exist, however, a provider cannot invent an
        undeclared name and reach a generic dispatcher.

        Infrastructure tools (Tool Search, skills) are declared by the channel
        rather than by the caller's catalogue, so they answer for themselves
        without appearing in it, when the session declares them: Tool
        Search's only while it hides the session's catalogue (RFC §12.4).
        """
        if name in (self._channel_tool_names() if served is None else served):
            return self._channel_declares(name, session.id)
        tools = self._session_catalogue(session.id)
        if not tools:
            return True
        return any(isinstance(tool, dict) and tool.get("name") == name for tool in tools)

    def _admitted_catalogue(self, session_id: str) -> list[dict[str, Any]]:
        """Every tool the session declares (its catalogue, the channel's own)
        that its tool policies admit, skill gating aside: what a skill's
        ``requires`` is checked against, as on a text turn, and the schemas an
        activation may hand over (RFC §24.3)."""
        return [
            tool
            for tool in self._session_declared_tools(session_id)
            if (name := dict_tool_name(tool)) and self._session_admits(session_id, name)
        ]

    def _session_declared_tools(self, session_id: str) -> list[dict[str, Any]]:
        """Every tool the session can call: its catalogue, then the channel's
        own it declares (Tool Search's while active, the skills')."""
        own: list[dict[str, Any]] = []
        if self._tool_search_support is not None:
            own += self._tool_search_support.search_tool_dicts()
        if self._skill_support is not None:
            own += self._skill_support.skill_tool_dicts()
        declared = [t for t in own if self._channel_declares(t["name"], session_id)]
        return self._session_catalogue(session_id) + declared

    def _channel_declares(self, name: str, session_id: str) -> bool:
        """Whether the session declares the channel's own tool *name*: Tool
        Search's while it hides the session's catalogue, the skills' it
        offers (``run_skill_script`` only with an executor)."""
        search = self._tool_search_support
        if search is not None and search.is_search_tool(name):
            return search.active(session_id)
        skills = self._skill_support
        if skills is not None and skills.is_skill_tool(name):
            return any(tool["name"] == name for tool in skills.skill_tool_dicts())
        return True

    def _channel_tool_names(self) -> frozenset[str]:
        """The tools this channel serves itself: Tool Search's and the skills'."""
        return frozenset(e.name for e in self._registry.entries(None, source=ToolSource.CHANNEL))

    def _exempt_tool_names(self) -> frozenset[str]:
        """The channel's own tools that escape the policy and skill gating (RFC §21.1)."""
        return frozenset(self._registry.names(None, lambda traits: traits.exempt))

    def _door_exempt(self, channel_serves: bool) -> frozenset[str]:
        """What escapes the policy and skill gating on a door: the channel's
        exempt tools where it serves its own tools, nothing where it does not
        (a reasoning backend, a call recovered from speech), whose calls name
        no tool of the channel's (RFC §21.1)."""
        return self._exempt_tool_names() if channel_serves else frozenset()

    def _declared_once(
        self, tools: list[dict[str, Any]], room_id: str | None
    ) -> list[dict[str, Any]]:
        """A session's host tools in *room_id*: none under a name the channel or
        orchestration declares itself there, each name once (RFC §21.1,
        :func:`declared_once`). Those are composed in afterwards. A tool
        orchestration serves without declaring it always (a pipeline agent's,
        its handoff) comes through the session's catalogue: only its own
        declaration does, never another tool given under its name."""
        served = (
            self._channel_tool_names()
            | self._human_input_names()
            | self._registry.names(room_id, lambda traits: traits.always_declared)
        )
        own = [tool for tool in tools if self._declares_its_server(tool, room_id)]
        return declared_once(own, dict_tool_name, served, self._collisions)

    def _declares_its_server(self, tool: dict[str, Any], room_id: str | None) -> bool:
        """Whether *tool* may be declared under its name in *room_id*: no
        orchestration entry serves the name there, or *tool* is that entry's
        own declaration. Another tool under the name is dropped, and said once
        (RFC §21.1): the gate would check one schema, the entry serve another."""
        name = dict_tool_name(tool)
        entry = self._registry.lookup(name, room_id) if name else None
        if entry is None or entry.source is not ToolSource.ORCHESTRATION:
            return True
        if entry.declares(tool.get("description") or "", tool.get("parameters") or {}):
            return True
        self._collisions.served(str(name))
        return False

    def _tool_reachable(self, name: str, session_id: str) -> bool:
        """Whether the session may call *name*: its tool policy and skill gating.

        What Tool Search may name in its results and listings (RFC §21.1); the
        pre-execution gate enforces the same rule on the call itself.
        """
        return self._access_cause(name, session_id) is None

    def _session_policies(self, session_id: str) -> list[ToolPolicy]:
        """The tool policies the session answers to, each resolved for its
        participant: the channel's, and its active agent's when a pipeline set
        one (RFC §12.4, §19.5)."""
        role = self._session_roles.get(session_id)
        policies = (self._tool_policy, self._session_agent_policies.get(session_id))
        return [policy.resolve(role) for policy in policies if policy is not None]

    def _session_admits(
        self, session_id: str, name: str, exempt: Container[str] | None = None
    ) -> bool:
        """Whether every policy the session answers to admits *name*, *exempt*
        (by default the channel's own exempt tools) passing (RFC §21.1): the
        one reading of a session's policy, for its declaration, its gate, Tool
        Search and the names a handler reads."""
        passes = self._exempt_tool_names() if exempt is None else exempt
        return all(policy_admits(p, name, passes) for p in self._session_policies(session_id))

    def _session_policy_check(self, session_id: str) -> Callable[[str], bool] | None:
        """:meth:`_session_admits` for one session as it stands now, or ``None``
        when no policy applies to it: a call's toolset is the one it started
        with, whatever a handoff or the session's end does during it."""
        policies = self._session_policies(session_id)
        if not policies:
            return None
        passes = self._exempt_tool_names()
        return lambda name: all(policy_admits(p, name, passes) for p in policies)

    def _policy_filter(self, session_id: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The part of *tools* the session's policies admit; a provider's
        native tool, which has no name for a policy to name, is kept."""
        if not self._session_policies(session_id):
            return tools
        return [
            t
            for t in tools
            if not (name := dict_tool_name(t)) or self._session_admits(session_id, name)
        ]

    async def _use_agent_policy(self, session: VoiceSession, policy: ToolPolicy | None) -> None:
        """Make *policy* the one of the pipeline agent *session* now speaks as
        (``None`` when that agent has none), with its participant's role read
        for it (RFC §19.5). A session that ended keeps nothing."""
        self._set_agent_policy(session, policy)
        await self._refresh_session_role(session, session.room_id)

    def _set_agent_policy(self, session: VoiceSession, policy: ToolPolicy | None) -> None:
        """Record *policy* as the one of the agent *session* speaks as, while
        the session lives."""
        with self._state_lock:
            if session.id not in self._sessions:
                return
            if policy is None:
                self._session_agent_policies.pop(session.id, None)
            else:
                self._session_agent_policies[session.id] = policy

    async def _refresh_session_policies(self, session: VoiceSession, room_id: str | None) -> None:
        """Read again what a call is judged by: the agent the room talks to
        now, the one that serves the call (RFC §19.5), and the participant's
        role, so a role changed during the session holds from the next call
        on (RFC §12.4)."""
        if self._registry.session_source is not None:
            config = await self._room_session_config(room_id or session.room_id)
            self._set_agent_policy(session, config.tool_policy if config is not None else None)
        await self._refresh_session_role(session, room_id)

    async def _refresh_session_role(self, session: VoiceSession, room_id: str | None) -> None:
        """Read the participant's role again, so a role changed during the
        session holds at the gate from the next call on (RFC §12.4)."""
        if not self._reads_roles(session.id) or not (self._framework and room_id):
            return
        role = await self._resolve_session_role(room_id, session.participant_id, session.id)
        if session.id in self._session_roles:
            self._session_roles[session.id] = role

    def _reads_roles(self, session_id: str) -> bool:
        """Whether a policy the session answers to has role overrides to read."""
        policies = (self._tool_policy, self._session_agent_policies.get(session_id))
        return any(policy is not None and policy.role_overrides for policy in policies)

    async def _resolve_session_role(
        self, room_id: str | None, participant_id: str, session_id: str
    ) -> str | None:
        """The session participant's role, where a policy has overrides to read."""
        if not self._reads_roles(session_id) or not (self._framework and room_id):
            return None
        # Under the framework's lease, like every store read a channel makes:
        # a call landing while the kit closes must not read a closing store.
        with self._framework._resource_lease():
            participant = await self._framework.store.get_participant(room_id, participant_id)
        return participant.role if participant is not None else None

    async def _authorize_realtime_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        call_id: str,
        room_id: str | None,
        session: VoiceSession,
        *,
        channel_serves: bool = True,
        can_activate: bool = True,
    ) -> tuple[dict[str, Any], GateRefusal | None, RoomContext | None]:
        """Pre-execution gate for realtime tool calls (parity with the classic
        AI path), in RFC §12.4's order.

        *channel_serves* says whether this entry serves the channel's own
        tools (Tool Search, skills): the provider's function calls do; a
        reasoning backend's calls and a recovered spoken call reach the
        handler only, so on them no name is the channel's (RFC §21.1).
        *can_activate* is false for a model that cannot activate a skill
        itself (a reasoning backend), whose skill refusal says so.

        Checks the tool is declared, applies the tool policy and skill gating,
        folds a flattened hub-tool call back into ``params`` and validates the
        arguments against the declared schema, and runs BEFORE_TOOL_USE so a
        block prevents the side effect rather than only hiding the result.
        Hooks may replace the arguments through ``metadata["arguments"]``; the
        replacement is validated before it can reach the handler.

        Returns the effective arguments, an optional denial result, and the
        room context this gate built — ``None`` when it built none. The caller
        hands that context to ON_TOOL_CALL's judgement as ``carrying`` so one
        tool call deserialises the room history once instead of twice.
        """
        served = self._channel_tool_names() if channel_serves else frozenset()
        if not self._is_declared_realtime_tool(name, session, served):
            logger.warning("Realtime provider requested undeclared tool %s", name)
            # The search hint only for a model that can call find_tools here.
            search = self._tool_search_support
            searching = channel_serves and search is not None and search.active(session.id)
            undeclared = json.dumps(unknown_tool_error(name, searching=searching))
            return arguments, GateRefusal(undeclared), None
        # Access before the arguments: a refused tool never names its schema
        # (RFC §21.1).
        await self._refresh_session_policies(session, room_id)
        exempt = self._door_exempt(channel_serves)
        cause = self._access_cause(name, session.id, exempt, can_activate=can_activate)
        if cause is not None:
            logger.warning("Realtime tool %s refused: %s", name, cause)
            return arguments, GateRefusal(json.dumps({"error": cause})), None
        params = self._tool_parameters(name, session)
        arguments, invalid = self._validated_realtime_arguments(name, arguments, params)
        if invalid is not None:
            return arguments, GateRefusal(invalid), None
        return await self._before_realtime_tool_use(
            name, arguments, params, call_id, room_id, session
        )

    def _validated_realtime_arguments(
        self, name: str, arguments: dict[str, Any], params: dict[str, Any] | None
    ) -> tuple[dict[str, Any], str | None]:
        """The model's arguments checked against the declared schema (fail-closed),
        after repairing a hub tool's flattened ``params``: same gate, same order
        as the classic AI path."""
        if params is None:
            return arguments, None
        folded, fold_error = fold_hoisted_arguments(params, arguments)
        if fold_error is not None:
            logger.warning("Realtime tool %s arguments ambiguous: %s", name, fold_error)
            return arguments, json.dumps(
                {"error": f"Invalid arguments for '{name}': {fold_error}"}
            )
        if folded is not None:
            logger.info(
                "Realtime tool %s: folded hoisted arguments %s into its container "
                "(provider=%s, model=%s)",
                name,
                sorted(set(arguments) - set(folded)),
                self._provider.name,
                self._provider.model_name,
            )
            arguments = folded
        arg_error = validate_tool_arguments(params, arguments)
        if arg_error is not None:
            logger.warning("Realtime tool %s arguments rejected: %s", name, arg_error)
            return arguments, json.dumps({"error": f"Invalid arguments for '{name}': {arg_error}"})
        return arguments, None

    def _access_cause(
        self,
        name: str,
        session_id: str,
        exempt: Container[str] | None = None,
        *,
        can_activate: bool = True,
    ) -> str | None:
        """Why the session may not call *name*, in the words every gate uses
        (RFC §21.1): its tool policies, resolved for its participant, then
        skill gating, as on the classic path; *exempt* as
        :meth:`_session_admits` reads it, for both. *can_activate* is false for
        a model that cannot activate a skill itself (a reasoning backend)."""
        if not self._session_admits(session_id, name, exempt):
            return policy_refusal(name)
        # Hiding a gated tool from the catalogue is not enforcement — the model
        # may still name one it saw before the skill was deactivated.
        support = self._skill_support
        if support is not None and support.is_gated(name, session_id, exempt=exempt):
            closed = support.is_closed_for_good(name)
            return gated_tool_refusal(name, can_activate=can_activate, closed=closed)
        return None

    async def _before_realtime_tool_use(
        self,
        name: str,
        arguments: dict[str, Any],
        params: dict[str, Any] | None,
        call_id: str,
        room_id: str | None,
        session: VoiceSession,
    ) -> tuple[dict[str, Any], GateRefusal | None, RoomContext | None]:
        """BEFORE_TOOL_USE as every channel runs it, which needs a framework and
        a room to run room hooks; the arguments it leaves are validated again."""
        framework = self._framework
        if framework is None or not room_id:
            return arguments, None, None
        pre_event = ToolCallEvent(
            channel_id=self.channel_id,
            channel_type=ChannelType.REALTIME_VOICE,
            tool_call_id=call_id,
            name=name,
            arguments=arguments,
            result=None,
            room_id=room_id,
            session=session,
        )
        decision, context = await framework._decide_before_tool_use(pre_event, self.channel_id)
        if not decision:
            logger.info("Realtime tool %s denied by BEFORE_TOOL_USE hook", name)
            denial = json.dumps({"error": pre_execution_denial(name, decision.reason)})
            return arguments, GateRefusal(denial, decision.detail), context
        arguments, invalid = _rewritten_arguments(name, arguments, params, decision.arguments)
        return arguments, GateRefusal(invalid) if invalid is not None else None, context


def _rewritten_arguments(
    name: str,
    arguments: dict[str, Any],
    params: dict[str, Any] | None,
    rewritten: dict[str, Any] | None,
) -> tuple[dict[str, Any], str | None]:
    """The arguments BEFORE_TOOL_USE left, returned or edited in place, checked
    against the schema again."""
    effective = rewritten if rewritten is not None else arguments
    invalid = rewritten_arguments_error(name, params, effective)
    if invalid is not None:
        logger.warning("Realtime tool %s: %s", name, invalid)
        return effective, json.dumps({"error": invalid})
    return effective, None
