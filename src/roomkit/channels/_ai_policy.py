"""AIChannel mixin for tool policy enforcement and skill gating."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from roomkit.channels._skill_constants import SKILL_INFRA_TOOL_NAMES
from roomkit.channels._tool_search_constants import TOOL_SEARCH_INFRA_TOOL_NAMES
from roomkit.providers.ai.base import AITool
from roomkit.sandbox.tools import SANDBOX_TOOL_PREFIX
from roomkit.tools.policy import ToolPolicy

if TYPE_CHECKING:
    from roomkit.channels.ai import _ToolLoopContext
    from roomkit.models.context import RoomContext
    from roomkit.models.event import RoomEvent
    from roomkit.skills.registry import SkillRegistry


@runtime_checkable
class ToolPolicyHost(Protocol):
    """Contract: capabilities a host class must provide for AIToolPolicyMixin.

    Attributes provided by the host's ``__init__``:
        _tool_policy: Global tool access policy (may contain per-role overrides).
        _skills: Skill registry for gated tool resolution.

    Methods provided by AISteeringMixin (or equivalent):
        _get_loop_ctx: Return the current tool-loop context (activated skills,
            participant role, steering queue).
    """

    _tool_policy: ToolPolicy | None
    _skills: SkillRegistry | None
    _tool_search_pinned: set[str]

    def _get_loop_ctx(self) -> _ToolLoopContext: ...


class AIToolPolicyMixin:
    """Resolves participant roles and enforces tool policy / skill gating.

    Host contract: :class:`ToolPolicyHost`.
    """

    _tool_policy: ToolPolicy | None
    _skills: SkillRegistry | None
    _tool_search_pinned: set[str]
    _get_loop_ctx: Callable[[], _ToolLoopContext]

    def _resolve_participant_role(self, event: RoomEvent, context: RoomContext) -> str | None:
        """Look up the participant role for the event source."""
        pid = event.source.participant_id
        if not pid:
            return None
        for p in context.participants:
            if p.id == pid:
                return p.role
        return None

    @property
    def _effective_tool_policy(self) -> ToolPolicy | None:
        """Return the tool policy resolved for the current participant role."""
        if self._tool_policy is None:
            return None
        return self._tool_policy.resolve(self._get_loop_ctx().current_participant_role)

    # Infrastructure tool names — never filtered by policy or gating.
    # Includes skill tools and channel-managed tools (eviction, planning).
    _SKILL_INFRA_TOOLS: frozenset[str] = SKILL_INFRA_TOOL_NAMES | frozenset(
        {"read_stored_result", "plan_tasks"}
    )

    @property
    def _gated_tool_names(self) -> set[str]:
        """Collect tool names gated by skills that have NOT been activated yet."""
        if not self._skills:
            return set()
        activated = self._get_loop_ctx().activated_skills
        gated: set[str] = set()
        for meta in self._skills.all_metadata():
            if meta.name in activated:
                continue
            gated.update(meta.gated_tool_names)
        return gated

    def _apply_tool_filters(self, tools: list[AITool]) -> list[AITool]:
        """Apply tool policy, skill gating, and Tool Search to a list of tools.

        Skill infrastructure tools and Tool Search discovery tools are *never*
        filtered — they ARE the skill/discovery mechanism and must stay
        reachable while the discretionary catalogue is hidden (the model still
        needs ``activate_skill`` and ``find_tools``).

        Sandbox tools are channel-attached rather than user-selected, so they
        are exempt from the user **tool policy** and skill **gating** — but they
        still participate in **Tool Search**. A sandbox can expose ~10 tools
        (read/write/edit/ls/grep/find/git/diff/delete/bash); on a small model
        those would otherwise crowd the context window, so they are deferred
        behind ``find_tools`` like any other discretionary tool and revealed on
        demand. A host that wants an always-visible entry point pins specific
        sandbox tool names via ``tool_search_pinned``.

        When Tool Search is active for the turn (``loop_ctx.tool_search_active``,
        set in ``_build_context``), the discretionary catalogue is collapsed to
        the pinned set plus the tools already revealed by ``find_tools`` this
        loop. The re-filter runs every round, so a tool revealed in round N
        becomes visible in round N+1 — the same mechanism as skill gating.

        Uses ``_effective_tool_policy`` which incorporates role-based overrides.
        """
        gated = self._gated_tool_names
        policy = self._effective_tool_policy
        loop_ctx = self._get_loop_ctx()
        # ``None`` = Tool Search inactive (no collapse). Otherwise the set of
        # discretionary tool names that stay visible this round.
        keep: set[str] | None = None
        if loop_ctx.tool_search_active:
            # pinned (config) + revealed (find_tools this loop) + sticky (tools
            # already used this conversation, re-exposed so they stay callable).
            keep = self._tool_search_pinned | loop_ctx.revealed_tools | loop_ctx.sticky_tools
        result: list[AITool] = []
        for tool in tools:
            name = tool.name
            # Skill infra + Tool Search discovery tools always pass
            if name in self._SKILL_INFRA_TOOLS or name in TOOL_SEARCH_INFRA_TOOL_NAMES:
                result.append(tool)
                continue
            # Sandbox tools are channel-attached, not user-managed: skip the
            # user tool policy + skill gating. They still flow through the Tool
            # Search ``keep`` filter below so a large sandbox toolset can be
            # deferred behind find_tools on small models.
            if not name.startswith(SANDBOX_TOOL_PREFIX):
                # Tool policy filter (role-aware)
                if policy and not policy.is_allowed(name):
                    continue
                # Skill gating filter
                if name in gated:
                    continue
            # Tool Search: hide the discretionary catalogue (sandbox included)
            # behind find_tools, exposing only pinned + already-revealed tools.
            if keep is not None and name not in keep:
                continue
            result.append(tool)
        return result
