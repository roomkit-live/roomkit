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

    Skill body delivery on realtime channels uses ``system_instruction``
    (refreshed via ``provider.reconfigure``), NOT the activate_skill tool
    result. Returning a multi-KB skill body through ``submit_tool_result``
    reliably tipped Gemini Live (and similarly long realtime returns
    on OpenAI Realtime) into "narrate the script" mode — the model
    treated the long return as conversational data and stopped emitting
    function calls for the rest of the session. Routing the body to
    system_instruction keeps it as binding rules and leaves the tool
    surface intact.
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
        # session_id -> ordered list of (skill_name, instructions) tuples
        # for skills activated so far in this session. Concatenated into
        # the system_instruction on the next reconfigure_session call.
        # Order matches activation sequence so a chained flow can layer
        # later skills on top of earlier ones.
        self._activated_bodies: dict[str, list[tuple[str, str]]] = {}

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
        self._activated_bodies[session_id] = []

    def cleanup_session(self, session_id: str) -> None:
        """Remove activation state when a session ends."""
        self._activated_skills.pop(session_id, None)
        self._activated_bodies.pop(session_id, None)

    def activated_skills_prompt(self, session_id: str) -> str | None:
        """Return concatenated bodies of skills activated in this session.

        Used by the channel's tool dispatcher: after activate_skill
        runs we call ``provider.reconfigure(system_prompt=base + this)``
        so the skill content lives as binding rules in
        ``system_instruction`` rather than as a giant tool result that
        derails realtime function calling.

        Returns ``None`` when no skills have been activated yet so the
        caller can decide whether a reconfigure is even needed.
        """
        bodies = self._activated_bodies.get(session_id) or []
        if not bodies:
            return None
        sections = [
            f"## Active skill: {name}\n{instructions.strip()}"
            for name, instructions in bodies
            if instructions and instructions.strip()
        ]
        return "\n\n".join(sections) if sections else None

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
        """Acknowledge skill activation; buffer the body for system_instruction.

        Returns a small ACK as the tool result instead of the full skill
        instructions. The full body is buffered on
        ``self._activated_bodies[session_id]`` so the channel's tool
        dispatcher can fold it into ``system_instruction`` via
        ``provider.reconfigure``. See class docstring for the realtime
        rationale (long tool results derail Gemini Live function calling).
        """
        skill_name = arguments.get("name", "")
        skill = self._skills.get_skill(skill_name)
        if skill is None:
            return json.dumps(
                {
                    "error": f"Skill {skill_name!r} not found",
                    "available_skills": self._skills.skill_names,
                }
            )

        activated = self._activated_skills.get(session_id)
        if activated is not None and skill_name not in activated:
            activated.add(skill_name)
            bodies = self._activated_bodies.setdefault(session_id, [])
            bodies.append((skill.name, skill.instructions))

        payload: dict[str, Any] = {
            "ok": True,
            "name": skill.name,
            "_note": (
                "Skill instructions are now active in your system rules. "
                "Follow them. Use other tools to act — do NOT narrate "
                "tool calls without invoking them."
            ),
        }
        scripts = skill.list_scripts()
        if scripts:
            payload["scripts"] = scripts
        refs = skill.list_references()
        if refs:
            payload["references"] = refs
        return json.dumps(payload)

    async def _handle_read_reference(self, arguments: dict[str, Any]) -> str:
        """Read a reference file from a skill."""
        return await handle_read_reference(arguments, self._skills)

    async def _handle_run_script(self, arguments: dict[str, Any]) -> str:
        """Execute a script via the configured ScriptExecutor."""
        return await handle_run_script(arguments, self._skills, self._script_executor)
