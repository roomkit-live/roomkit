"""Skill support for RealtimeVoiceChannel.

Handles skill tool definitions, prompt injection, per-session activation
tracking, and tool gating for realtime voice sessions.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from roomkit.channels._skill_constants import (
    ACTIVATE_SKILL_SCHEMA,
    READ_REFERENCE_SCHEMA,
    RUN_SCRIPT_SCHEMA,
    SKILL_INFRA_TOOL_NAMES,
    SKILLS_NO_SCRIPTS_NOTE,
    SKILLS_PREAMBLE,
    TOOL_ACTIVATE_SKILL,
    TOOL_READ_REFERENCE,
    TOOL_RUN_SCRIPT,
)
from roomkit.channels._skill_handlers import (
    handle_activate_skill,
    handle_read_reference,
    handle_run_script,
)
from roomkit.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from roomkit.skills.executor import ScriptExecutor

logger = logging.getLogger("roomkit.channels.realtime_voice")


class RealtimeSkillSupport:
    """Skill infrastructure for RealtimeVoiceChannel.

    Unlike AIChannel (which rebuilds the tool list per request), realtime
    channels set tools once at ``provider.connect()`` and must call
    ``provider.reconfigure()`` to push newly-visible tools after a skill
    activation reveals gated tools.

    Activation state is tracked per session since each session has its own
    AI conversation.
    """

    def __init__(
        self,
        skills: SkillRegistry,
        script_executor: ScriptExecutor | None = None,
    ) -> None:
        self._skills = skills
        self._script_executor = script_executor
        # session_id -> set of activated skill names
        self._activated_skills: dict[str, set[str]] = {}

    # -- Tool definitions (as dicts, the format provider.connect expects) --

    def skill_tool_dicts(self) -> list[dict[str, Any]]:
        """Return skill infrastructure tool definitions as plain dicts."""
        tools: list[dict[str, Any]] = [ACTIVATE_SKILL_SCHEMA, READ_REFERENCE_SCHEMA]
        if self._script_executor:
            tools.append(RUN_SCRIPT_SCHEMA)
        return tools

    # -- System prompt injection --

    def inject_skills_prompt(self, system_prompt: str | None) -> str:
        """Append skills preamble and available-skills XML to the prompt."""
        preamble = SKILLS_PREAMBLE
        if not self._script_executor:
            preamble += SKILLS_NO_SCRIPTS_NOTE
        skills_xml = self._skills.to_prompt_xml()
        skill_block = f"\n\n{preamble}\n\n{skills_xml}"
        return (system_prompt or "") + skill_block

    # -- Per-session activation tracking --

    def init_session(self, session_id: str) -> None:
        """Initialize activation state for a new session."""
        self._activated_skills[session_id] = set()

    def cleanup_session(self, session_id: str) -> None:
        """Remove activation state when a session ends."""
        self._activated_skills.pop(session_id, None)

    # -- Tool dispatch --

    def is_skill_tool(self, name: str) -> bool:
        """Return True if *name* is a skill infrastructure tool."""
        return name in SKILL_INFRA_TOOL_NAMES

    async def handle_tool_call(self, name: str, arguments: dict[str, Any], session_id: str) -> str:
        """Dispatch a skill tool call and return the JSON result string."""
        if name == TOOL_ACTIVATE_SKILL:
            return await self._handle_activate_skill(arguments, session_id)
        if name == TOOL_READ_REFERENCE:
            return await self._handle_read_reference(arguments)
        if name == TOOL_RUN_SCRIPT:
            return await self._handle_run_script(arguments)
        return json.dumps({"error": f"Unknown skill tool: {name}"})

    # -- Tool gating --

    def _gated_tool_names(self, session_id: str) -> set[str]:
        """Collect tool names gated by skills not yet activated in this session."""
        activated = self._activated_skills.get(session_id, set())
        gated: set[str] = set()
        for meta in self._skills.all_metadata():
            if meta.name in activated:
                continue
            gated.update(meta.gated_tool_names)
        return gated

    def get_visible_tools(
        self, all_tools: list[dict[str, Any]], session_id: str
    ) -> list[dict[str, Any]]:
        """Filter tool list, removing gated tools but keeping infra tools."""
        gated = self._gated_tool_names(session_id)
        if not gated:
            return all_tools
        return [
            t
            for t in all_tools
            if t.get("name") in SKILL_INFRA_TOOL_NAMES or t.get("name") not in gated
        ]

    def newly_visible_after_activation(
        self,
        all_tools: list[dict[str, Any]],
        session_id: str,
        skill_name: str,
    ) -> list[dict[str, Any]] | None:
        """Return updated tool list if activation revealed new tools, else None."""
        meta = self._skills.get_metadata(skill_name)
        if not meta or not meta.gated_tool_names:
            return None
        # Re-filter with the now-activated skill
        return self.get_visible_tools(all_tools, session_id)

    # -- Internal handlers --

    async def _handle_activate_skill(self, arguments: dict[str, Any], session_id: str) -> str:
        """Load and return full skill instructions, tracking activation."""
        result_str, skill_name = await handle_activate_skill(arguments, self._skills)
        # Track activation so gated tools become visible
        activated = self._activated_skills.get(session_id)
        if activated is not None:
            activated.add(skill_name)
        return result_str

    async def _handle_read_reference(self, arguments: dict[str, Any]) -> str:
        """Read a reference file from a skill."""
        return await handle_read_reference(arguments, self._skills)

    async def _handle_run_script(self, arguments: dict[str, Any]) -> str:
        """Execute a script via the configured ScriptExecutor."""
        return await handle_run_script(arguments, self._skills, self._script_executor)
