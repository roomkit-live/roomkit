"""AIChannel mixin for tool policy enforcement, skill gating, and the declaration
a provider holds unseen, kept from one turn to the next (RFC §6.4)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Container
from typing import TYPE_CHECKING, Any

from roomkit.channels._tool_reopen import (
    REOPENING,
    insert_before,
    reopening_exchanges,
    room_declaration,
)
from roomkit.core._authors import People
from roomkit.models.tool_call import DeclaredTool, ToolDeclarationOrigin
from roomkit.providers.ai.base import AIContext, AIMessage, AITool, AIToolResultPart
from roomkit.tools.policy import (
    ToolPolicy,
    call_admitted,
    call_gated,
    policy_refusal,
)
from roomkit.tools.result import gated_tool_refusal

if TYPE_CHECKING:
    from collections.abc import Iterable

    from roomkit.channels._skill_activation import SkillActivationMemory
    from roomkit.channels._tool_registry import ChannelRegistry
    from roomkit.channels._tool_usage import ToolUsageMemory
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.providers.ai.base import AIProvider
    from roomkit.skills.registry import SkillRegistry
    from roomkit.tools.context import _ToolLoopContext

if TYPE_CHECKING:
    from roomkit.channels._ai_contract import _AIChannelContract
else:
    _AIChannelContract = object

logger = logging.getLogger("roomkit.channels.ai")


def policy_admits(policy: ToolPolicy | None, name: str, exempt: Container[str]) -> bool:
    """Whether a channel's role-resolved *policy* admits *name* (RFC §21.1).

    The one reading of the policy for every channel that carries one, AI or
    realtime, and for a conference: a tool in *exempt* passes, anything else is
    the policy's to allow. *exempt* names the tools the channel serves itself
    that only read or unlock and never act (the ``exempt`` trait of their
    entries): a tool of the host, of MCP, of orchestration or of a hook under
    one of these names is not the channel's, and the policy governs it. A call
    under an MCP alias is judged under both names (:func:`judged_names`).
    """
    return name in exempt or policy is None or call_admitted(policy, name)


def policy_check(
    policy: ToolPolicy | None, exempt: Container[str]
) -> Callable[[str], bool] | None:
    """Whether *policy* admits a name, as :func:`policy_admits` reads it, held
    for a turn; ``None`` when no policy applies (RFC §21.4)."""
    if policy is None:
        return None
    return lambda name: policy_admits(policy, name, exempt)


class AIToolPolicyMixin(_AIChannelContract):
    """Resolves participant roles and enforces tool policy / skill gating.

    What it calls on the other mixins is declared once, in
    :class:`~roomkit.channels._ai_contract._AIChannelContract`, which it
    derives from for the type checker only.
    """

    _tool_policy: ToolPolicy | None
    _skills: SkillRegistry | None
    _skill_activation: SkillActivationMemory
    _tool_search_pinned: set[str]
    _provider: AIProvider
    _registry: ChannelRegistry  # the tools the channel serves, with their traits
    _tool_usage: ToolUsageMemory  # the room's tool memory, its kept declaration included

    def _resolve_participant_role(self, event: RoomEvent, context: RoomContext) -> str | None:
        """The role of the participant *event*'s sender is on its channel."""
        record = People(context.participants).named(event)
        return record.role if record is not None else None

    @property
    def _exempt_tool_names(self) -> set[str]:
        """The channel's own tools that escape the policy and skill gating (RFC §21.1)."""
        return self._registry.names(None, lambda traits: traits.exempt)

    @property
    def _effective_tool_policy(self) -> ToolPolicy | None:
        """Return the tool policy resolved for the current participant role."""
        if self._tool_policy is None:
            return None
        return self._tool_policy.resolve(self._get_loop_ctx().current_participant_role)

    @property
    def _gated_tool_names(self) -> set[str]:
        """Collect tool names gated by skills that have NOT been activated yet.

        An activation counts for the whole conversation, not just the turn that
        made it: the loop context is only this turn's view, so it is unioned
        with the room's activation record. Without that, a skill activated in
        turn N would see its gated tools disappear again in turn N+1 — while
        its instructions, now carried by the system prompt, still tell the
        model to use them. A standalone turn reads none of the room's working
        state (RFC §10.1.1): only what it activates itself opens a gate.
        """
        skills = self._turn_skills()
        if not skills:
            return set()
        return skills.gated_tool_names(self._activated_skill_names())

    def _turn_skills(self) -> SkillRegistry | None:
        """The skills this turn reads (RFC §24.3): the registry its
        configuration gave it, else the channel's."""
        given = self._get_loop_ctx().skills
        return given if given is not None else self._skills

    def _activated_skill_names(self) -> set[str]:
        """The skills this conversation activated: this turn's, and the room's
        unless the turn is standalone (RFC §10.1.1)."""
        loop_ctx = self._get_loop_ctx()
        room = (
            set() if loop_ctx.standalone else self._skill_activation.active_names(loop_ctx.room_id)
        )
        return loop_ctx.activated_skills | room

    def _record_declared_tools(
        self, loop_ctx: _ToolLoopContext, tools: list[AITool] | None
    ) -> None:
        """Record the toolset one generation round hands the provider.

        Called with the ``AIContext.tools`` of every provider call of the turn,
        by the loop that makes the call, and with the held tools a result
        references (``_reference_shown``). Not from ``_apply_tool_filters``: its
        output is not the round's declaration (a ``BEFORE_AI_GENERATION`` hook
        may edit the tools, the eviction tool is added per round after them)
        and it also serves as a single-tool probe. A name is recorded once per
        turn, on its first round, with the reason Tool Search let it through as
        it stood then. The turn's
        ``AIResponseEvent.declared_tools`` reports the union.
        """
        if not tools:
            return
        declared = loop_ctx.declared_tools
        for tool in tools:
            # Held unseen: reported once a result references it (RFC §6.4).
            if tool.defer_loading:
                continue
            if tool.name not in declared:
                origin = self._declaration_origin(tool.name, loop_ctx)
                declared[tool.name] = DeclaredTool.from_tool(tool, origin)

    def _never_deferred(self, loop_ctx: _ToolLoopContext) -> set[str]:
        """The tools Tool Search never defers, declared at every round.

        The host's pinned, the channel's own, what orchestration injected for
        the room, and what the generation hook added (RFC §6.4, §21.1): the
        keep-set's floor, and what ``find_tools`` never names.
        """
        return self._tool_search_pinned | self._never_held(loop_ctx)

    def _never_held(self, loop_ctx: _ToolLoopContext) -> set[str]:
        """The tools a room's kept declaration always shows (RFC §6.4): what
        Tool Search never hides (the channel's own, what orchestration
        injected) and what a hook added. A tool the host pins is not among
        them: a skill's gating can hold it, and once a turn opened it, it is
        reopened like any other."""
        return self._never_hidden(loop_ctx.room_id) | loop_ctx.hook_pinned

    def _holds_tools(self, loop_ctx: _ToolLoopContext) -> bool:
        """Whether this turn declares tools held unseen: a loop built with its
        context, on a provider that can hold a tool (RFC §6.4)."""
        return loop_ctx.all_context_tools is not None and self._provider.supports_deferred_tools

    def _declaration_origin(self, name: str, loop_ctx: _ToolLoopContext) -> ToolDeclarationOrigin:
        """Why a tool is in this round's declaration (``ToolDeclarationOrigin``).

        The keep-set of ``_apply_tool_filters`` is ``pinned | revealed |
        sticky`` plus the tools the collapse never touches; this names which
        term admitted the tool, pinned before sticky before revealed: the
        earliest reason it was visible. Outside Tool Search, and for a tool
        the collapse never touches (an infrastructure tool, one orchestration
        injected, one a hook added), there is no such reason (RFC §21.1).
        """
        if not loop_ctx.tool_search_active:
            return "always"
        if name in self._tool_search_pinned:
            return "pinned"
        if name in self._never_held(loop_ctx):
            return "always"
        if name in loop_ctx.sticky_tools:
            return "sticky"
        if name in loop_ctx.revealed_tools:
            return "revealed"
        return "always"

    def _policy_allows(self, name: str) -> bool:
        """Whether the turn's role-resolved policy admits *name* (RFC §21.1).

        Skill gating aside: what the prompt may describe as available, a gated
        tool included, since activating its skill opens it.
        """
        return policy_admits(self._effective_tool_policy, name, self._exempt_tool_names)

    def _gate_refusal(self, name: str) -> dict[str, str] | None:
        """Why the policy or skill gating refuses a call to *name*, or ``None``.

        The execution guard's reading of the listing filter's rule (RFC
        §21.1): the same exempt names, the same role-resolved policy, the same
        glob-aware gating (RFC §24.2).
        """
        exempt = self._exempt_tool_names
        if name in exempt:
            return None
        if not policy_admits(self._effective_tool_policy, name, exempt):
            logger.warning("Tool %s blocked by policy", name)
            return {"error": policy_refusal(name)}
        if call_gated(name, self._gated_tool_names):
            logger.warning("Tool %s blocked by skill gating", name)
            skills = self._turn_skills()
            closed = skills is not None and call_gated(name, skills.unopenable_tool_names())
            return {"error": gated_tool_refusal(name, closed=closed)}
        return None

    def _reachable_tools(self, tools: Iterable[AITool]) -> list[AITool]:
        """The tools the policy and skill gating let this turn reach (RFC §21.1).

        What ``find_tools`` and ``list_tools`` search and hint from: a name the
        model could never call is a false promise, and naming it discloses
        what the policy hides. Tool Search's reveal window does not apply.
        """
        policy = self._effective_tool_policy
        gated = self._gated_tool_names
        exempt = self._exempt_tool_names
        return [tool for tool in tools if self._is_reachable(tool.name, policy, gated, exempt)]

    def _hook_toolset(self, loop_ctx: _ToolLoopContext) -> list[str]:
        """The turn's toolset as a tool hook reads it: what the policy and
        skill gating let it reach, as ``BEFORE_AI_GENERATION`` sees it, Tool
        Search's whole catalogue included; a withdrawn tool has left it
        (RFC §6.4)."""
        return [tool.name for tool in self._reachable_tools(loop_ctx.all_context_tools or ())]

    @staticmethod
    def _is_reachable(
        name: str, policy: ToolPolicy | None, gated: set[str], exempt: Container[str]
    ) -> bool:
        """Whether the role-resolved *policy* and skill gating admit *name*.

        *exempt* names the channel's own tools that escape both (RFC §21.1).
        """
        if name in exempt:
            return True
        if not policy_admits(policy, name, exempt):
            return False
        # ``gated`` holds ToolPolicy globs, not names (RFC §24.2): an
        # exact-membership test would let ``search_*`` gate nothing at all.
        return not call_gated(name, gated)

    def _apply_tool_filters(self, tools: list[AITool]) -> list[AITool]:
        """Apply tool policy, skill gating, and Tool Search to a list of tools.

        The policy-exempt tools (the ``exempt`` trait: skill activation
        and reference reading, the eviction re-read, Tool Search's discovery
        tools) always pass: they ARE the skill/discovery mechanism and must
        stay reachable while the discretionary catalogue is hidden. Every
        other tool, the ones the channel injects included, passes the
        role-aware policy and skill gating first (RFC §21.1).

        When Tool Search is active for the turn (``loop_ctx.tool_search_active``,
        set in ``_build_context``), the discretionary catalogue is collapsed to
        the pinned set plus the tools already revealed by ``find_tools`` this
        loop. The re-filter runs every round, so a tool revealed in round N
        becomes visible in round N+1 — the same mechanism as skill gating.
        Channel-managed tools (``run_skill_script``, ``plan_tasks``) are never
        deferred, nor are the tools orchestration injected for this room (RFC
        §21.1): the agent is told to call them. Sandbox tools are: a sandbox
        can expose ~10 tools, which on a small model would crowd the context
        window, so they wait behind ``find_tools`` like any other
        discretionary tool unless the host pins them via ``tool_search_pinned``.
        """
        gated = self._gated_tool_names
        policy = self._effective_tool_policy
        exempt = self._exempt_tool_names
        loop_ctx = self._get_loop_ctx()
        # ``None`` = Tool Search inactive (no collapse). Otherwise the set of
        # discretionary tool names that stay visible this round.
        keep: set[str] | None = None
        if loop_ctx.tool_search_active:
            # What is never deferred + revealed (find_tools this loop) + sticky
            # (tools already used this conversation, re-exposed so they stay
            # callable).
            keep = self._never_deferred(loop_ctx) | loop_ctx.revealed_tools | loop_ctx.sticky_tools
        result: list[AITool] = []
        for tool in tools:
            name = tool.name
            if not self._is_reachable(name, policy, gated, exempt):
                continue
            if keep is not None and name not in keep:
                continue
            result.append(tool)
        return result

    def _held_declaration(self, loop_ctx: _ToolLoopContext, shown: list[AITool]) -> list[AITool]:
        """The round's declaration where the provider holds a tool unseen: what
        the turn's first round showed, then every other tool the policy admits,
        held unseen (RFC §6.4); *shown* where it cannot.

        Held from the first round on, what Tool Search hides and what a
        skill's gating keeps closed stay declared as they were: a result that
        makes one callable references it (``_reference_shown``) instead of
        declaring it, and the declaration does not change within the turn.
        The policy still decides what is declared: a tool it denies is not.
        """
        if not self._holds_tools(loop_ctx):
            return shown
        if loop_ctx.first_shown is None:
            # A standalone turn: its own declaration, the room's untouched.
            loop_ctx.first_shown = frozenset(t.name for t in shown)
        first, held = loop_ctx.first_shown, self._held_names(loop_ctx)
        return [
            *(t for t in shown if t.name in first),
            *(
                t.model_copy(update={"defer_loading": True})
                for t in loop_ctx.all_context_tools or ()
                if t.name in held
            ),
        ]

    def _open_turn_declaration(
        self, context: AIContext, loop_ctx: _ToolLoopContext, shown: list[AITool]
    ) -> None:
        """Fix the turn's declaration from the room's, and reopen before the
        turn's input what it would show beyond it (RFC §6.4).

        A provider that cannot hold a tool unseen keeps the turn's own
        declaration, and so does a standalone turn, which reads none of the
        room's working state (RFC §10.1.1).
        """
        if not self._holds_tools(loop_ctx) or loop_ctx.standalone:
            return
        room_id = loop_ctx.room_id
        kept = self._tool_usage.declaration(room_id)
        first = room_declaration(kept, {t.name for t in shown}, self._never_held(loop_ctx))
        opened = [t for t in shown if t.name not in first]
        left = self._reopen(context, loop_ctx, opened, first) if opened else []
        # What nothing could reopen is shown, as a turn would show it.
        first = first | {t.name for t in left}
        loop_ctx.first_shown = first
        self._tool_usage.keep_declaration(room_id, first)
        reopened = [t for t in opened if t not in left]
        if reopened:
            # The round's own declaration first, then what the exchange opened.
            self._record_declared_tools(loop_ctx, [t for t in shown if t.name in first])
            self._mark_referenced(loop_ctx, reopened)
            logger.debug("Turn reopens %s in room %s", [t.name for t in reopened], room_id)

    def _reopen(
        self,
        context: AIContext,
        loop_ctx: _ToolLoopContext,
        opened: list[AITool],
        first: frozenset[str],
    ) -> list[AITool]:
        """Place before the turn's input the exchanges that reopen *opened*;
        the tools they do not reach."""
        exchanges, left = reopening_exchanges(
            opened,
            first,
            self._skill_activation.active_names(loop_ctx.room_id),
            self._turn_skills(),
            f"reopen_{loop_ctx.loop_id[:12]}",
        )
        if not exchanges or not insert_before(context.messages, loop_ctx.turn_input, exchanges):
            return opened
        return left

    def _mark_referenced(self, loop_ctx: _ToolLoopContext, tools: list[AITool]) -> None:
        """Record *tools*, held unseen, as referenced once, and as declared."""
        loop_ctx.referenced |= {t.name for t in tools}
        self._record_declared_tools(loop_ctx, tools)

    def _held_names(self, loop_ctx: _ToolLoopContext) -> set[str]:
        """The tools the turn holds unseen: declared from its first round, not
        shown there, and admitted by the policy. Empty where nothing is held."""
        if loop_ctx.first_shown is None:
            return set()
        policy, exempt = self._effective_tool_policy, self._exempt_tool_names
        return {
            t.name
            for t in loop_ctx.all_context_tools or ()
            if t.name not in loop_ctx.first_shown and policy_admits(policy, t.name, exempt)
        }

    def _reference_shown(self, loop_ctx: _ToolLoopContext) -> list[str]:
        """The held tools the turn now shows, each referenced once: what a
        provider that cannot hold a tool would have seen declared after the
        call just served (a reveal, an activation, a call-time recovery).

        Recorded in the turn's declaration (RFC §6.4) and returned for the
        call's result, which makes them callable.
        """
        held = self._held_names(loop_ctx) - loop_ctx.referenced
        if not held:
            return []
        shown = [
            t for t in self._apply_tool_filters(loop_ctx.all_context_tools or []) if t.name in held
        ]
        self._mark_referenced(loop_ctx, shown)
        return [t.name for t in shown]

    def _show_summarized_references(self, summarized: list[AIMessage]) -> None:
        """Show the held tools whose references a compaction summarized away:
        the references were what made them callable (RFC §6.4)."""
        loop_ctx = self._get_loop_ctx()
        lost = _references_in(summarized)
        if lost and loop_ctx.first_shown is not None:
            loop_ctx.first_shown = loop_ctx.first_shown | lost


def declared_for(provider: Any, context: AIContext) -> AIContext:
    """*context* as *provider* can take it: where it cannot hold a tool unseen,
    the held tools are dropped and those a result referenced declared plainly
    (a fallback provider receives what the turn made visible, RFC §6.4), and
    each result goes as its text, without the references only such a
    provider reads (a provider behind a gateway that does not take them). The
    exchange that reopened tools at the turn's start is left out: it only
    stood for their references, and the tools are declared instead."""
    if provider.supports_deferred_tools:
        return context
    referenced = _references_in(context.messages)
    if not referenced and not any(t.defer_loading for t in context.tools):
        return context
    tools = [
        t.model_copy(update={"defer_loading": False}) if t.defer_loading else t
        for t in context.tools
        if not t.defer_loading or t.name in referenced
    ]
    messages = [_without_references(m) for m in context.messages if not m.metadata.get(REOPENING)]
    return context.model_copy(update={"tools": tools, "messages": messages})


def _without_references(message: AIMessage) -> AIMessage:
    """*message* with its tool results carrying their text alone."""
    if not isinstance(message.content, list) or not any(
        isinstance(part, AIToolResultPart) and part.references for part in message.content
    ):
        return message
    content = [
        part.model_copy(update={"references": []})
        if isinstance(part, AIToolResultPart) and part.references
        else part
        for part in message.content
    ]
    return message.model_copy(update={"content": content})


def _references_in(messages: list[AIMessage]) -> set[str]:
    """The tools the tool results of *messages* reference."""
    return {
        name
        for message in messages
        if isinstance(message.content, list)
        for part in message.content
        if isinstance(part, AIToolResultPart)
        for name in part.references
    }
