"""Code a served skill would cause to run, approved by the host (RFC §24.7).

While a skill an MCP server serves is active in a conversation, its content
may direct the model to run code: a sandbox command, a skill's script, a tool
the host names as running code. Such a call is served only when the host's
``approve_execution`` approves every served skill active there. An approval
holds for the conversation, the skill's fingerprint and the source that gave
it only: an entry read again with another set of files is asked again, a
dynamic skill, which has no fingerprint, is asked every time, and the same
entry served by another source (another registry's server, as a turn may give
its own registry) asks that source.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Container, Iterable

from roomkit.channels._skill_constants import TOOL_RUN_SCRIPT
from roomkit.skills.mcp.approval import SkillExecutionRequest
from roomkit.skills.mcp.skill import MCPSkill
from roomkit.skills.registry import SkillRegistry
from roomkit.tools.policy import matches_any_pattern

logger = logging.getLogger("roomkit.channels.skills")

_MAX_APPROVALS = 1000
"""Approvals a channel remembers, oldest dropped first: one forgotten is asked again."""


class ServedExecutionGate:
    """The approvals a host gave, per conversation, source, skill and fingerprint."""

    def __init__(self, max_approvals: int = _MAX_APPROVALS) -> None:
        self._max = max_approvals
        # Keyed by the source's identity, each entry holding the source itself:
        # while an approval is kept, no other source can take that identity.
        self._approved: OrderedDict[tuple[str | None, int, str, str], object] = OrderedDict()

    def forget(self, room_id: str) -> None:
        """Let go of the approvals given for *room_id*'s conversation, as the
        channel lets the room go (RFC §10.6.2): a room joined again under
        that id is asked again."""
        for key in [key for key in self._approved if key[0] == room_id]:
            del self._approved[key]

    async def refusal(
        self,
        skills: SkillRegistry | None,
        active: Iterable[str],
        room_id: str | None,
        tool: str,
        *,
        sandbox_tools: Container[str] = (),
    ) -> str | None:
        """Why *tool* may not run now, or ``None`` when it may.

        *active* are the skills active in the conversation; *sandbox_tools*
        the sandbox commands the channel serves, by exact name.
        """
        if skills is None or not skills.has_served:
            return None
        # Every served name the conversation holds active, loaded or not: a
        # session may still carry the body of a skill the registry dropped,
        # and one no approval can bind is refused, never skipped.
        served = [name for name in sorted(active) if skills.is_served(name)]
        if not served or not _runs_code(skills, tool, sandbox_tools):
            return None
        for name in served:
            if not await self._approves(skills, name, room_id, tool):
                return (
                    f"Tool {tool!r} runs code, and skill {name!r}, served by an MCP server, "
                    "is active: the host has not approved code run on that skill's behalf."
                )
        return None

    async def _approves(
        self, skills: SkillRegistry, name: str, room_id: str | None, tool: str
    ) -> bool:
        """Whether the host approves code run while *name* is active."""
        skill, origin = skills.get_skill(name), skills.origin(name)
        if not isinstance(skill, MCPSkill) or origin is None:
            return False
        fingerprint = skill.held.fingerprint
        # Kept only for a conversation, a fingerprint and the source that gave
        # it: without the first two, an approval would cover what it never saw,
        # and another source's entry under the same name is that source's to
        # approve (a turn's registry may hold it, RFC §24.3), so it is asked.
        key = (room_id, id(origin), name, fingerprint) if fingerprint and room_id else None
        if key is not None and key in self._approved:
            self._approved.move_to_end(key)
            return True
        if origin.approve_execution is None:
            return False
        request = SkillExecutionRequest(
            name=name,
            label=origin.label,
            uri=skill.held.uri,
            fingerprint=fingerprint,
            room_id=room_id,
            tool=tool,
        )
        try:
            approved = await origin.approve_execution(request)
        except Exception:
            logger.exception("approve_execution failed for skill %s; the call is refused", name)
            return False
        if approved is not True:
            return False
        if key is not None:
            self._approved[key] = origin
            while len(self._approved) > self._max:
                self._approved.popitem(last=False)
        return True


def _runs_code(skills: SkillRegistry, tool: str, sandbox_tools: Container[str]) -> bool:
    """Whether *tool* runs code: a sandbox command, a skill's script, or a
    tool a host named so when it registered a server."""
    if tool in sandbox_tools or tool == TOOL_RUN_SCRIPT:
        return True
    return matches_any_pattern(tool, list(skills.execution_tool_patterns))
